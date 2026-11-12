"""Frozen-backbone Faster R-CNN localization, TBX11K and CANDID-PTX.
Use COCO bbox AP50:95 and AP50, selecting epochs on a validation split only.
TBX's official labeled val is never used for model selection.
"""
import argparse, collections, contextlib, copy, hashlib, io, json, os, random, sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'requirements'))
import numpy as np
import pydicom
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ark_finetune.extract_embeddings import load_backbone

TB='/data/jliang12/nuislam/data_files_splits/tbx11k'
CB='/data/jliang12/nuislam/FoundationX_localization_bbox_annotation_collections/CANDID_PTX'
TB_IMG='/data/jliang12/shared/dataset/TBX11/TBX11K/imgs'
CB_IMG='/data/jliang12/shared/dataset/CANDID-PTX/dataset'
NORM={'imagenet':([.485,.456,.406],[.229,.224,.225]),'raddino':([.5307]*3,[.2583]*3)}
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for x in iter(lambda:f.read(1048576),b''): h.update(x)
 return h.hexdigest()
def read_json(path):
 with open(path) as f: return json.load(f)
def subset(d,ids):
 ids=set(ids); out={k:copy.deepcopy(v) for k,v in d.items() if k not in ('images','annotations')}
 out['images']=[x for x in d['images'] if x['id'] in ids];out['annotations']=[x for x in d['annotations'] if x['image_id'] in ids]
 out.setdefault('info',{});out.setdefault('licenses',[]);return out
def tb_split(d):
 # Stable across models/seeds; stratify positive/negative and source folder.
 pos={a['image_id'] for a in d['annotations'] if not a.get('iscrowd',0)}; groups=collections.defaultdict(list)
 for im in d['images']: groups[(im['file_name'].split('/')[0],im['id'] in pos)].append(im)
 train=[]; val=[]
 for key,g in sorted(groups.items()):
  g.sort(key=lambda x:hashlib.sha256(('tbx-split-100:'+x['file_name']).encode()).hexdigest()); n=round(len(g)*.15)
  val.extend(x['id'] for x in g[:n]);train.extend(x['id'] for x in g[n:])
 assert not(set(train)&set(val)) and len(train)+len(val)==len(d['images'])
 return train,val
class BoxDataset(Dataset):
 def __init__(self,data,dataset):
  self.data=data;self.images=data['images'];self.dataset=dataset;self.anns=collections.defaultdict(list)
  for a in data['annotations']:
   if not a.get('iscrowd',0): self.anns[a['image_id']].append(a)
 def __len__(self):return len(self.images)
 def __getitem__(self,i):
  im=self.images[i];w,h=im['width'],im['height']
  if self.dataset=='tbx11k':
   with Image.open(os.path.join(TB_IMG,im['file_name'])) as p: img=p.convert('RGB')
   if img.size!=(w,h):raise ValueError('image size mismatch '+im['file_name'])
  else:
   name=im['file_name'];assert name.endswith('.png') and '/' not in name
   a=pydicom.dcmread(os.path.join(CB_IMG,name[:-4])).pixel_array.astype(np.float32)
   lo,hi=float(a.min()),float(a.max()); a=np.zeros(a.shape,dtype=np.uint8) if lo==hi else ((a-lo)*255/(hi-lo)).astype(np.uint8)
   img=Image.fromarray(a).resize((w,h),Image.Resampling.BILINEAR).convert('RGB')
  boxes=[]
  for a in self.anns[im['id']]:
   x,y,bw,bh=a['bbox'];x1=max(float(x),0);y1=max(float(y),0);x2=min(float(x+bw),w);y2=min(float(y+bh),h)
   if x2>x1 and y2>y1: boxes.append([x1,y1,x2,y2])
  return to_tensor(img),{'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),'labels':torch.ones(len(boxes),dtype=torch.int64),'image_id':torch.tensor(im['id'])}
def collate(batch):return tuple(zip(*batch))
class Backbone(nn.Module):
 out_channels=256
 def __init__(self,model,weights):
  super().__init__();self.vit=load_backbone(model,weights)
  for p in self.vit.parameters():p.requires_grad_(False)
  self.vit.eval();self.projection=nn.Conv2d(1024 if model=='vit_large_raydino' else 768,256,1)
 def train(self,mode=True):super().train(mode);self.vit.eval();return self
 def forward(self,x):
  with torch.no_grad(): y=self.vit.get_intermediate_layers(x,n=1,reshape=True)[0]
  return self.projection(y)
def evaluate(model,loader,data,device):
 model.eval(); pred=[]
 with torch.no_grad():
  for images,targets in loader:
   outputs=model([x.to(device) for x in images])
   for o,t in zip(outputs,targets):
    for box,score in zip(o['boxes'].cpu().tolist(),o['scores'].cpu().tolist()):
     x1,y1,x2,y2=box;pred.append({'image_id':int(t['image_id']),'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'score':score})
 coco=COCO();coco.dataset=copy.deepcopy(data);coco.dataset.setdefault('info',{});coco.dataset.setdefault('licenses',[])
 with contextlib.redirect_stdout(io.StringIO()):
  coco.createIndex()
  if not pred:return {'ap50_95':0.,'ap50':0.,'images':len(data['images']),'annotations':len(data['annotations'])}
  dt=coco.loadRes(pred); ev=COCOeval(coco,dt,'bbox');ev.params.imgIds=[x['id'] for x in data['images']];ev.evaluate();ev.accumulate();ev.summarize()
 return {'ap50_95':float(ev.stats[0]),'ap50':float(ev.stats[1]),'images':len(data['images']),'annotations':len(data['annotations'])}
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['tbx11k','candidptx']);p.add_argument('--model',required=True,choices=['vit_base_dinov3','vit_base_raddino','vit_large_raydino']);p.add_argument('--weights',required=True);p.add_argument('--normalization',required=True,choices=list(NORM));p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=10);p.add_argument('--batch',type=int,default=4);p.add_argument('--workers',type=int,default=8);p.add_argument('--out',required=True);p.add_argument('--manifest',required=True);a=p.parse_args()
 if os.path.exists(a.out):raise RuntimeError('output already exists: '+a.out)
 torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);assert torch.cuda.is_available();device=torch.device('cuda')
 if a.dataset=='tbx11k':
  ftrain=TB+'/all_train_2.json';ftest=TB+'/all_val_2.json';raw=read_json(ftrain);train_ids,val_ids=tb_split(raw);tr=subset(raw,train_ids);va=subset(raw,val_ids);te=read_json(ftest)
  manifest={'source':ftrain,'source_sha256':digest(ftrain),'algorithm':'sha256(tbx-split-100:filename), stratified by folder and annotation presence; 15% per group','train_filenames':sorted(x['file_name'] for x in tr['images']),'selection_filenames':sorted(x['file_name'] for x in va['images'])}
  if os.path.exists(a.manifest):
   if read_json(a.manifest)!=manifest:raise RuntimeError('existing split manifest differs')
  else:
   os.makedirs(os.path.dirname(a.manifest),exist_ok=True)
   with open(a.manifest,'x') as f:json.dump(manifest,f,indent=2)
 else:
  ftrain=CB+'/CANDID_PTX_train_1.json';fval=CB+'/CANDID_PTX_valid_1.json';ftest=CB+'/CANDID_PTX_test_1.json';tr=read_json(ftrain);va=read_json(fval);te=read_json(ftest)
 for name,d in [('train',tr),('selection',va),('heldout',te)]:
  assert len(d['images']) and len(d['annotations']),name
  print(name,len(d['images']),len(d['annotations']),flush=True)
 os.makedirs(a.out)
 ds=[BoxDataset(d,a.dataset) for d in (tr,va,te)]
 dl=[DataLoader(d,batch_size=a.batch,shuffle=(i==0),num_workers=a.workers,collate_fn=collate) for i,d in enumerate(ds)]
 size=518 if a.model=='vit_base_raddino' else 512
 b=Backbone(a.model,a.weights);anchors=AnchorGenerator(sizes=((16,32,64,128,256),),aspect_ratios=((.5,1.,2.),))
 mean,std=NORM[a.normalization];model=FasterRCNN(b,num_classes=2,min_size=size,max_size=size,image_mean=mean,image_std=std,rpn_anchor_generator=anchors,box_detections_per_img=100,box_score_thresh=.001).to(device)
 model.transform.size_divisible=14 if a.model=='vit_base_raddino' else 16
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.0001);best=-1;history=[]
 for epoch in range(1,a.epochs+1):
  model.train();losses=[]
  for ims,targets in dl[0]:
   loss=sum(model([x.to(device) for x in ims],[{k:v.to(device) for k,v in t.items()} for t in targets]).values())
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
  metrics=evaluate(model,dl[1],va,device);history.append({'epoch':epoch,'loss':float(np.mean(losses)),'validation':metrics});print(history[-1],flush=True)
  if metrics['ap50_95']>best:
   best=metrics['ap50_95'];torch.save({'model':model.state_dict(),'epoch':epoch},os.path.join(a.out,'best.pt'))
 ck=torch.load(os.path.join(a.out,'best.pt'),map_location=device,weights_only=False);model.load_state_dict(ck['model']);score=evaluate(model,dl[2],te,device)
 result={'args':vars(a),'input_size':size,'weight_sha256':digest(a.weights),'train_sha256':digest(ftrain),'test_sha256':digest(ftest),'selected_epoch':ck['epoch'],'history':history,'heldout':score,'warning':'TBX val contains known train/pretraining duplicates; full val retained per request' if a.dataset=='tbx11k' else None}
 with open(os.path.join(a.out,'result.json'),'x') as f:json.dump(result,f,indent=2)
 print('heldout',score,flush=True)
if __name__=='__main__':main()
