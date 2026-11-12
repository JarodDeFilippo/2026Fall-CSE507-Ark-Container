"""TBX11K frozen DINOv3 single-scale Faster R-CNN baseline.
Train on lab train COCO JSON. Divide lab val deterministically into validation and
held-out evaluation subsets. Lab test annotations are empty and are not used.
"""
import os, json, random, argparse, hashlib
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou
from ark_finetune.extract_embeddings import load_backbone

ROOT='/data/jliang12/shared/dataset/TBX11/TBX11K/imgs'
SPLITS='/data/jliang12/nuislam/data_files_splits/tbx11k'
class TBX(Dataset):
 def __init__(self, split, subset=None):
  d=json.load(open(f'{SPLITS}/all_{split}_2.json'))
  self.items=d['images']; self.ann=defaultdict(list)
  for a in d['annotations']:
   if a['category_id']==1 and not a.get('iscrowd',0): self.ann[a['image_id']].append(a['bbox'])
  if subset is not None:
   # Split each positive/negative group deterministically; no image in both groups.
   groups=[[x for x in self.items if bool(self.ann[x['id']]) == positive] for positive in (False,True)]
   rng=random.Random(100)
   for g in groups: g.sort(key=lambda x:x['file_name']); rng.shuffle(g)
   self.items=[x for g in groups for x in (g[:len(g)//2] if subset=='val' else g[len(g)//2:])]
  assert self.items and any(self.ann[x['id']] for x in self.items)
 def __len__(self): return len(self.items)
 def __getitem__(self, idx):
  item=self.items[idx]; path=os.path.join(ROOT,item['file_name'])
  with Image.open(path) as im: image=to_tensor(im.convert('RGB'))
  h,w=image.shape[-2:]; assert w==item['width'] and h==item['height'],path
  boxes=[]
  for x,y,bw,bh in self.ann[item['id']]:
   x1=max(0.,float(x)); y1=max(0.,float(y)); x2=min(w,float(x+bw)); y2=min(h,float(y+bh))
   if x2>x1 and y2>y1: boxes.append([x1,y1,x2,y2])
  return image, {'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4), 'labels':torch.ones(len(boxes),dtype=torch.int64),'image_id':torch.tensor(item['id'])}
def collate(batch): return tuple(zip(*batch))
class FrozenViT(nn.Module):
 out_channels=768
 def __init__(self, weights):
  super().__init__(); self.vit=load_backbone('vit_base_dinov3',weights)
  for p in self.vit.parameters(): p.requires_grad_(False)
  self.vit.eval()
 def train(self,mode=True):
  super().train(mode); self.vit.eval(); return self
 def forward(self,x):
  with torch.no_grad(): return self.vit.get_intermediate_layers(x,n=1,reshape=True)[0]
def evaluate(model,loader,device):
 model.eval(); predictions=[]; ground={}; total=0
 with torch.no_grad():
  for images,targets in loader:
   outputs=model([im.to(device) for im in images])
   for output,target in zip(outputs,targets):
    i=int(target['image_id']); ground[i]=target['boxes']; total+=len(target['boxes'])
    for box,score in zip(output['boxes'].cpu(),output['scores'].cpu()): predictions.append((float(score),i,box))
 predictions.sort(key=lambda x:x[0],reverse=True); seen={i:set() for i in ground}; tp=[]; fp=[]
 for score,i,box in predictions:
  g=ground[i]; j=-1
  if len(g):
   vals=box_iou(box[None],g)[0]; best=int(vals.argmax())
   if vals[best]>=0.5 and best not in seen[i]: j=best
  if j>=0: seen[i].add(j)
  tp.append(int(j>=0)); fp.append(int(j<0))
 if not total: raise ValueError('No annotated boxes in evaluation subset')
 tp=np.cumsum(tp); fp=np.cumsum(fp); rec=tp/total; prec=tp/np.maximum(tp+fp,1)
 ap=float(np.mean([max(prec[rec>=r],default=0.) for r in np.linspace(0,1,101)]))
 return {'ap50':ap,'ground_truth_boxes':total,'images':len(ground),'detections':len(predictions)}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--weights',required=True); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch',type=int,default=8); p.add_argument('--workers',type=int,default=8); a=p.parse_args()
 torch.manual_seed(100); np.random.seed(100); random.seed(100)
 if os.path.exists(a.out): raise RuntimeError('Output directory already exists: '+a.out)
 os.makedirs(a.out); device=torch.device('cuda'); assert torch.cuda.is_available()
 train=TBX('train'); val=TBX('val','val'); test=TBX('val','holdout')
 assert not ({x['file_name'] for x in val.items}&{x['file_name'] for x in test.items})
 loaders=[DataLoader(d,batch_size=a.batch,shuffle=(i==0),num_workers=a.workers,collate_fn=collate) for i,d in enumerate((train,val,test))]
 backbone=FrozenViT(a.weights); anchors=AnchorGenerator(sizes=((16,32,64,128,256),),aspect_ratios=((0.5,1.,2.),))
 model=FasterRCNN(backbone,num_classes=2,min_size=224,max_size=224,rpn_anchor_generator=anchors,box_detections_per_img=100,box_score_thresh=0.001).to(device)
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.0001,weight_decay=0.0001); best=-1; history=[]
 for epoch in range(1,a.epochs+1):
  model.train(); losses=[]
  for imgs,targets in loaders[0]:
   imgs=[x.to(device) for x in imgs]; targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
   l=model(imgs,targets); loss=sum(l.values());
   if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
   opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
  metrics=evaluate(model,loaders[1],device); history.append({'epoch':epoch,'loss':float(np.mean(losses)),'validation':metrics})
  print('epoch',epoch,history[-1],flush=True)
  if metrics['ap50']>best:
   best=metrics['ap50']; torch.save({'model':model.state_dict(),'epoch':epoch,'validation':metrics},os.path.join(a.out,'best.pt'))
 ck=torch.load(os.path.join(a.out,'best.pt'),map_location=device,weights_only=False); model.load_state_dict(ck['model']); result=evaluate(model,loaders[2],device)
 with open(os.path.join(a.out,'result.json'),'w') as f: json.dump({'weights':a.weights,'protocol':'lab train; lab validation split stratified by box presence seed 100 into selection and held-out; lab test has zero boxes', 'history':history,'selected_epoch':ck['epoch'],'heldout':result},f,indent=2)
 print('heldout',result,flush=True)
if __name__=='__main__': main()
