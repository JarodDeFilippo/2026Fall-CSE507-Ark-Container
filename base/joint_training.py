"""Reference implementation copied from the original Ark+ concurrent code.

This module is intentionally not imported by the current training entry point.
The copied sections will be adapted in later commits.
"""

import cv2
import numpy as np
import random
import time
import torch
import wandb
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from dataloader import (
    build_transform_classification,
    build_ts_transformations,
    dict_dataloarder,
)
from utils import MetricLogger, ProgressLogger


# Copied from https://github.com/jlianglab/Ark/tree/main/Ark_Plus/AblationStudy/Concurrent
# Source: Concurrent/dataloader.py, OmniPretrainingDatasets.
class OmniPretrainingDatasets(Dataset):
  def __init__(self, datasets_config, dataset_list = ["ChestXray14"], crop_size=224, resize=256, augment=None):
    self.dataset_list = dataset_list
    self.datasets_config = datasets_config
    self.dataset_image_list = []
    self.dataset_label_list = []
    self.dataset_index_list = []

    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)
    
    self.num_classes_list = []
    for idx, dataset in enumerate(dataset_list):
        dataset_loaded = dict_dataloarder[dataset](images_path=self.datasets_config[dataset]['data_dir'], file_path=self.datasets_config[dataset]['train_list'], augment=None)
        self.dataset_image_list.extend(dataset_loaded.img_list)
        self.dataset_label_list.extend(dataset_loaded.img_label)
        self.dataset_index_list.extend([idx for _ in range(len(dataset_loaded.img_list))])
        self.num_classes_list.append(len(self.datasets_config[dataset]['diseases']))
  
    max_class_num = max(self.num_classes_list)
    print("max_class_num", max_class_num)
    label_padding = []
    for label in self.dataset_label_list:
        if len(label) < max_class_num:
          label.extend([0 for _ in range(max_class_num - len(label))])
          assert len(label) == max_class_num
        label_padding.append(label)


  def __getitem__(self, index):
    cv2.setNumThreads(0)

    image_path = self.dataset_image_list[index]
    imageData = Image.open(image_path).convert('RGB').resize((self.resize,self.resize))
    imageLabel = self.dataset_label_list[index]
    imageLabel = torch.FloatTensor(imageLabel)
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.     
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
  
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')

    return student_img, teacher_img, imageLabel, self.dataset_index_list[index]

  def __len__(self):
    return len(self.dataset_image_list)


# Copied from https://github.com/jlianglab/Ark/tree/main/Ark_Plus/AblationStudy/Concurrent
# Source: Concurrent/dataloader.py, OmniPretrainingDatasets_EqualSampling.
class OmniPretrainingDatasets_EqualSampling(Dataset):
  def __init__(self, datasets_config, dataset_list = ["ChestXray14"], normalization = "imagenet"):
    self.dataset_list = dataset_list
    self.datasets_config = datasets_config
    self.dataset_image_lists = []
    self.dataset_label_lists = []
    self.num_classes_list = []
    for dataset in dataset_list:
        dataset_loaded = dict_dataloarder[dataset](images_path=self.datasets_config[dataset]['data_dir'], file_path=self.datasets_config[dataset]['train_list'], augment=None)
        self.dataset_image_lists.append(dataset_loaded.img_list)
        self.dataset_label_lists.append(dataset_loaded.img_label)
        self.num_classes_list.append(len(self.datasets_config[dataset]['diseases']))
    self.data_number_list = [len(im_list) for im_list in self.dataset_image_lists]
    self.prime_length = max(self.data_number_list) # set the dataset with the most number of data to be prime
    self.augment = build_transform_classification(normalize = normalization, mode="train")

  def __getitem__(self, index):
    # dataset_list ["ChestXray14", "CheXpert", "VinDrCXR"] 
    # data_number_list [70K, 220K, 15K]
    image_data_list = []
    image_label_list = []
    # return one image from each dataset to assemle a batch
    for i in range(len(self.dataset_list)):
      # this is the prime dataset with the most number of data
      if self.data_number_list[i] == self.prime_length: 
        reindex = index
      #  when self.data_number_list[i] < self.prime_length, need to deal with the index
      elif self.data_number_list[i] < self.prime_length: 
        reindex = index % self.data_number_list[i]
        if reindex == 0:
          random.Random(0).shuffle(self.dataset_image_lists[i])
          random.Random(0).shuffle(self.dataset_label_lists[i])
          
      image_path = self.dataset_image_lists[i][reindex]
      image_data = Image.open(image_path).convert('RGB')
      image_label = torch.FloatTensor(self.dataset_label_lists[i][reindex])
      if self.augment != None: image_data = self.augment(image_data)
      image_data_list.append(image_data)
      image_label_list.append(image_label)

    return image_data_list, image_label_list

  def __len__(self):
    return self.prime_length


# Copied from https://github.com/jlianglab/Ark/tree/main/Ark_Plus/AblationStudy/Concurrent
# Source: Concurrent/trainer.py, train_one_epoch and ema_update_teacher.
def train_one_epoch(model, data_loader_train, num_classes_list, device, criterion, optimizer, epoch, ema_mode, teacher, momentum_schedule, it):
    batch_time = MetricLogger('Time', ':6.3f')
    losses_cls = MetricLogger('Loss_cls', ':.4e')
    losses_mse = MetricLogger('Loss_mse', ':.4e')
    progress = ProgressLogger(
        len(data_loader_train),
        [batch_time, losses_cls, losses_mse],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    MSE = torch.nn.MSELoss()
    coff = (momentum_schedule[it] - 0.9) * 5
    end = time.time()
    for i, (samples1, samples2, targets, dataset_index) in enumerate(data_loader_train):
        samples1, samples2, targets = samples1.float().to(device), samples2.float().to(device), targets.float().to(device)


        feat_t, _ = teacher(samples2)
        feat_s, pred_s_lst = model(samples1)
        loss_const = MSE(feat_s, feat_t)

        loss_cls = 0
        for j, di in enumerate(dataset_index):
            l = criterion(pred_s_lst[di][j], targets[j][:num_classes_list[di]])
            loss_cls += l
        loss_cls = loss_cls/len(dataset_index)

        loss = (1-coff) * loss_cls + coff * loss_const

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_cls.update(loss_cls.item(), samples1.size(0))
        losses_mse.update(loss_const.item(), samples1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)
        if ema_mode == "iteration":
            ema_update_teacher(model, teacher, momentum_schedule, it)
            it += 1
            
    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1
    
    wandb.log({"train_loss_cls": losses_cls.avg})
    wandb.log({"train_loss_mse": losses_mse.avg})


def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
