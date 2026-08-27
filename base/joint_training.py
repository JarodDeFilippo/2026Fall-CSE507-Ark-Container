"""Joint-training helpers based on the original Ark+ concurrent code."""

from bisect import bisect_right
import os
import random
import time
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataloader import (
    build_transform_classification,
    dict_dataloarder,
)
from utils import MetricLogger, save_image


# Based on the copied implementation from:
# https://github.com/jlianglab/Ark/tree/main/Ark_Plus/AblationStudy/Concurrent
#
# The original class reparsed datasets_config and rebuilt each dataset. This
# version consumes the already-created current dataset objects so it preserves
# their transforms, labels, and dataset-specific options.
class OmniPretrainingDatasets(Dataset):
  def __init__(self, dataset_train_list, num_classes_list):
    self.datasets = dataset_train_list
    self.num_classes_list = list(num_classes_list)
    if len(self.datasets) != len(self.num_classes_list):
      raise ValueError("Expected one class count for each training dataset")

    self.cumulative_sizes = []
    cumulative_size = 0
    for dataset in self.datasets:
      cumulative_size += len(dataset)
      self.cumulative_sizes.append(cumulative_size)

    if not self.cumulative_sizes:
      raise ValueError("At least one training dataset is required")
    self.max_class_num = max(self.num_classes_list)

  def __getitem__(self, index):
    dataset_index = bisect_right(self.cumulative_sizes, index)
    previous_size = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
    sample_index = index - previous_size
    student_img, teacher_img, image_label = self.datasets[dataset_index][sample_index]
    image_label = torch.as_tensor(image_label, dtype=torch.float32)
    if image_label.shape[0] < self.max_class_num:
      image_label = torch.cat(
          (
              image_label,
              torch.zeros(
                  self.max_class_num - image_label.shape[0],
                  dtype=image_label.dtype,
              ),
          )
      )

    return student_img, teacher_img, image_label, dataset_index

  def __len__(self):
    return self.cumulative_sizes[-1]


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


# Adapted from the copied implementation above and the Ark+ Concurrent
# joint-training dataset. It keeps the current dataset objects and returns
# the same per-sample interface as OmniPretrainingDatasets.
class OmniPretrainingDatasetsEqualSampling(Dataset):
  def __init__(self, dataset_train_list, num_classes_list):
    self.datasets = dataset_train_list
    self.num_classes_list = list(num_classes_list)
    if len(self.datasets) != len(self.num_classes_list):
      raise ValueError("Expected one class count for each training dataset")
    if not self.datasets:
      raise ValueError("At least one training dataset is required")

    self.dataset_sizes = [len(dataset) for dataset in self.datasets]
    if any(size == 0 for size in self.dataset_sizes):
      raise ValueError("Equal sampling requires non-empty training datasets")
    self.max_dataset_size = max(self.dataset_sizes)
    self.max_class_num = max(self.num_classes_list)

  def __getitem__(self, index):
    dataset_index = index % len(self.datasets)
    sample_index = (index // len(self.datasets)) % self.dataset_sizes[dataset_index]
    student_img, teacher_img, image_label = self.datasets[dataset_index][sample_index]
    image_label = torch.as_tensor(image_label, dtype=torch.float32)
    if image_label.shape[0] < self.max_class_num:
      image_label = torch.cat(
          (
              image_label,
              torch.zeros(
                  self.max_class_num - image_label.shape[0],
                  dtype=image_label.dtype,
              ),
          )
      )

    return student_img, teacher_img, image_label, dataset_index

  def __len__(self):
    return self.max_dataset_size * len(self.datasets)


# Based on the copied implementation from:
# https://github.com/jlianglab/Ark/tree/main/Ark_Plus/AblationStudy/Concurrent
#
# The dataset-index head routing and loss balance are retained. This version
# uses the current model API, task-specific criteria, and distributed loss
# reduction.
def train_one_epoch_joint(
        model,
        data_loader_train,
        num_classes_list,
        task_types,
        device,
        optimizer,
        epoch,
        ema_mode,
        teacher,
        momentum_schedule,
        it,
        accelerator=None,
        is_main_process=True,
        global_step=0,
        momentum=None,
        train_log=None,
        loss_writer=None,
        loss_file=None,
        snapshot_directory=None,
):
    batch_time = MetricLogger('Time', ':6.3f')
    losses_cls = MetricLogger('Loss_cls', ':.4e')
    losses_mse = MetricLogger('Loss_mse', ':.4e')
    losses_total = MetricLogger('Loss_total', ':.4e')
    model.train()
    MSE = torch.nn.MSELoss()
    criteria = [
        torch.nn.CrossEntropyLoss()
        if task_type == "multi-class classification"
        else torch.nn.BCEWithLogitsLoss()
        for task_type in task_types
    ]
    if momentum is None:
        momentum = momentum_schedule[it]
    coff = (momentum - 0.9) * 5
    end = time.time()
    for i, (samples1, samples2, targets, dataset_index) in enumerate(data_loader_train):
        samples1, samples2, targets = samples1.float().to(device), samples2.float().to(device), targets.float().to(device)
        dataset_index = dataset_index.to(device)

        if is_main_process and snapshot_directory is not None and i == 0:
            save_image(
                samples1[0].cpu().numpy().transpose(1, 2, 0),
                os.path.join(snapshot_directory, "student"),
            )
            save_image(
                samples2[0].cpu().numpy().transpose(1, 2, 0),
                os.path.join(snapshot_directory, "teacher"),
            )

        with torch.no_grad():
            feat_t = teacher(samples2, return_features=True)
        feat_s, pred_s_lst = model(samples1, return_all=True)
        loss_const = MSE(feat_s, feat_t)

        loss_cls = torch.zeros((), dtype=loss_const.dtype, device=device)
        for dataset_index_value, criterion in enumerate(criteria):
            dataset_mask = dataset_index == dataset_index_value
            if dataset_mask.any():
                dataset_loss = criterion(
                    pred_s_lst[dataset_index_value][dataset_mask],
                    targets[dataset_mask, :num_classes_list[dataset_index_value]],
                )
                loss_cls += dataset_loss * dataset_mask.sum()
            else:
                # Every head is returned by the forward pass. Keep absent
                # heads connected to the loss graph so DDP observes a zero
                # gradient instead of waiting for a missing reduction.
                loss_cls += pred_s_lst[dataset_index_value].sum() * 0.0
        loss_cls = loss_cls / targets.shape[0]

        loss = (1-coff) * loss_cls + coff * loss_const

        optimizer.zero_grad()
        loss.backward()
        if accelerator is not None:
            accelerator.mark_step()
        optimizer.step()
        if accelerator is not None:
            accelerator.mark_step()

        loss_stats = torch.tensor(
            [
                (1 - coff) * loss_cls.item() * samples1.size(0),
                coff * loss_const.item() * samples1.size(0),
                loss.item() * samples1.size(0),
                samples1.size(0),
            ],
            dtype=torch.float32,
            device=device,
        )
        if accelerator is not None and accelerator.distributed:
            accelerator.all_reduce(loss_stats)
        global_batch_size = int(loss_stats[3].item())
        global_cls_value = (loss_stats[0] / loss_stats[3]).item()
        global_mse_value = (loss_stats[1] / loss_stats[3]).item()
        global_total_value = (loss_stats[2] / loss_stats[3]).item()
        losses_cls.update(global_cls_value, global_batch_size)
        losses_mse.update(global_mse_value, global_batch_size)
        losses_total.update(global_total_value, global_batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        total_loss = losses_total.avg
        if total_loss != 0:
            cls_percent = 100 * losses_cls.avg / total_loss
            mse_percent = 100 * losses_mse.avg / total_loss
        else:
            cls_percent = 0
            mse_percent = 0
        current_total = global_total_value
        if current_total != 0:
            current_cls_percent = 100 * global_cls_value / current_total
            current_mse_percent = 100 * global_mse_value / current_total
        else:
            current_cls_percent = 0
            current_mse_percent = 0

        if is_main_process and loss_writer is not None:
            loss_writer.writerow([
                epoch + 1,
                epoch,
                global_step + i,
                i + 1,
                "joint",
                global_batch_size,
                global_cls_value,
                global_mse_value,
                global_total_value,
                current_cls_percent,
                current_mse_percent,
                optimizer.param_groups[0]["lr"],
                momentum,
            ])
            if loss_file is not None:
                loss_file.flush()

        if is_main_process and ((i + 1) % 50 == 0 or i + 1 == len(data_loader_train)):
            message = (
                "Cycle {:04d} | Dataset joint | Batch {:04d}/{:04d} | "
                "classification={:.4e} ({:.1f}%) | "
                "consistency={:.4e} ({:.1f}%) | total={:.4e} (100.0%)"
            ).format(
                epoch + 1,
                i + 1,
                len(data_loader_train),
                losses_cls.avg,
                cls_percent,
                losses_mse.avg,
                mse_percent,
                total_loss,
            )
            if train_log is not None:
                train_log.info(message)
            else:
                print(message)
        if ema_mode == "iteration":
            ema_update_teacher(model, teacher, momentum_schedule, it, accelerator)
            it += 1
            
    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it, accelerator)
        it += 1

    return it, {
        "classification_loss": losses_cls.avg,
        "consistency_loss": losses_mse.avg,
        "total_loss": losses_total.avg,
    }


def ema_update_teacher(model, teacher, momentum_schedule, it, accelerator=None):
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
    if accelerator is not None:
        accelerator.mark_step()
