"""One batch from training data only; verifies forward, gradient, COCO evaluation."""
import argparse,torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from .compare_detectors import TB,CB,read_json,subset,BoxDataset,Backbone,collate,evaluate,NORM
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['tbx11k','candidptx']);p.add_argument('--model',required=True,choices=['vit_base_dinov3','vit_base_raddino','vit_large_raydino']);p.add_argument('--weights',required=True);p.add_argument('--normalization',required=True,choices=list(NORM));a=p.parse_args()
f=TB+'/all_train_2.json' if a.dataset=='tbx11k' else CB+'/CANDID_PTX_train_1.json';raw=read_json(f);pos={x['image_id'] for x in raw['annotations']};chosen=[x['id'] for x in raw['images'] if x['id'] in pos][:2]+[x['id'] for x in raw['images'] if x['id'] not in pos][:2];data=subset(raw,chosen);data.pop('info',None); loader=DataLoader(BoxDataset(data,a.dataset),batch_size=2,num_workers=0,collate_fn=collate)
size=518 if a.model=='vit_base_raddino' else 512;mean,std=NORM[a.normalization];anchors=AnchorGenerator(sizes=((16,32,64,128,256),),aspect_ratios=((.5,1.,2.),));model=FasterRCNN(Backbone(a.model,a.weights),num_classes=2,min_size=size,max_size=size,image_mean=mean,image_std=std,rpn_anchor_generator=anchors,box_score_thresh=.001).cuda()
model.transform.size_divisible=14 if a.model=='vit_base_raddino' else 16
model.train();images,targets=next(iter(loader));losses=model([x.cuda() for x in images],[{k:v.cuda() for k,v in t.items()} for t in targets]);loss=sum(losses.values());assert torch.isfinite(loss);loss.backward();assert model.backbone.projection.weight.grad is not None;assert all(p.grad is None for p in model.backbone.vit.parameters());print('forward/backward OK',float(loss),flush=True);print('coco OK',evaluate(model,loader,data,torch.device('cuda')),flush=True)
