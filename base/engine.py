
import os
import sys
import shutil
import time
import csv
import glob
import re
import tempfile
import json
import logging
import numpy as np
from optparse import OptionParser
import copy


from accelerator import Accelerator, DistributedEvaluationSampler
from models import build_omni_model, save_checkpoint
from utils import metric_AUROC, cosine_scheduler
from sklearn.metrics import accuracy_score

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import train_one_epoch, test_classification, evaluate
#import segmentation_models_pytorch as smp
from utils import cosine_anneal_schedule,dice,mean_dice_coef

from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from functools import partial
import torch.nn as nn

# import wandb

sys.setrecursionlimit(40000)


def _print_and_log(message, log_file=None):
    if log_file is not None:
        log_file.info(message)
    else:
        print(message)


def _configure_logger(log_file=None):
    logger = logging.getLogger("ark")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _close_logger(logger):
    if logger is None:
        return
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def _copy_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {
            key: _copy_to_cpu(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_to_cpu(item) for item in value)
    return value


def _dataset_sample_signature(dataset):
    image_paths = getattr(dataset, 'img_list', None)
    labels = getattr(dataset, 'img_label', None)
    if image_paths is None or labels is None or len(image_paths) != len(labels):
        return None

    records = []
    for image_path, label in zip(image_paths, labels):
        label_array = np.asarray(label)
        records.append((
            os.path.normcase(os.path.realpath(image_path)),
            label_array.dtype.str,
            tuple(label_array.shape),
            label_array.tobytes(),
        ))
    return sorted(records)


def _datasets_have_identical_samples(validation_dataset, test_dataset):
    if len(validation_dataset) != len(test_dataset):
        return False
    validation_signature = _dataset_sample_signature(validation_dataset)
    test_signature = _dataset_sample_signature(test_dataset)
    return (
        validation_signature is not None
        and validation_signature == test_signature
    )


def _append_evaluation_rows(file_path, rows):
    evaluation_header = [
        "cycle",
        "epoch",
        "evaluation_point",
        "metric",
        "student",
        "teacher",
    ]
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    if file_exists:
        with open(file_path, 'r', newline='') as file_descriptor:
            existing_rows = list(csv.reader(file_descriptor))
        if existing_rows and existing_rows[0] == [
                "cycle",
                "epoch",
                "metric",
                "student",
                "teacher",
        ]:
            with tempfile.NamedTemporaryFile(
                    mode='w',
                    dir=os.path.dirname(file_path),
                    newline='',
                    delete=False) as temporary_file:
                writer = csv.writer(temporary_file)
                writer.writerow(evaluation_header)
                for existing_row in existing_rows[1:]:
                    if len(existing_row) == 5:
                        existing_row = [
                            existing_row[0],
                            existing_row[1],
                            "",
                            existing_row[2],
                            existing_row[3],
                            existing_row[4],
                        ]
                    writer.writerow(existing_row)
                temporary_path = temporary_file.name
            os.replace(temporary_path, file_path)
    with open(file_path, 'a', newline='') as file_descriptor:
        writer = csv.writer(file_descriptor)
        if not file_exists:
            writer.writerow(evaluation_header)
        writer.writerows(rows)


def _evaluate_test_sets(
        student_model,
        teacher,
        dataset_list,
        datasets_config,
        data_loader_list_test,
        device,
        accelerator,
        evaluation_directory,
        cycle,
        epoch,
        evaluation_point,
        shared_val_test_splits,
        train_log):
    student_means = []
    teacher_means = []
    if accelerator.is_main_process:
        _print_and_log(
            "Evaluating student and teacher test sets at {} in Cycle {}"
            .format(evaluation_point, cycle),
            train_log,
        )

    for dataset_index, dataset in enumerate(dataset_list):
        diseases = datasets_config[dataset]['diseases']
        multiclass = (
            datasets_config[dataset]['task_type']
            == "multi-class classification"
        )
        y_student, p_student = test_classification(
            student_model,
            dataset_index,
            data_loader_list_test[dataset_index],
            device,
            multiclass,
            len(diseases),
            accelerator=accelerator,
        )
        y_teacher, p_teacher = test_classification(
            teacher,
            dataset_index,
            data_loader_list_test[dataset_index],
            device,
            multiclass,
            len(diseases),
            accelerator=accelerator,
        )
        y_student = torch.cat(accelerator.gather_tensor(y_student), 0)
        p_student = torch.cat(accelerator.gather_tensor(p_student), 0)
        y_teacher = torch.cat(accelerator.gather_tensor(y_teacher), 0)
        p_teacher = torch.cat(accelerator.gather_tensor(p_teacher), 0)
        if not accelerator.is_main_process:
            continue

        if dataset == "CheXpert":
            performance_diseases = datasets_config[dataset]['test_diseases_name']
            performance_indices = [
                diseases.index(name) for name in performance_diseases
            ]
            y_student = y_student[:, performance_indices]
            p_student = p_student[:, performance_indices]
            y_teacher = y_teacher[:, performance_indices]
            p_teacher = p_teacher[:, performance_indices]
        else:
            performance_diseases = diseases

        student_auc = metric_AUROC(
            y_student,
            p_student,
            len(performance_diseases),
        )
        teacher_auc = metric_AUROC(
            y_teacher,
            p_teacher,
            len(performance_diseases),
        )
        rows = []
        accuracy_message = ""
        if multiclass:
            student_accuracy = accuracy_score(
                np.argmax(y_student.cpu().numpy(), axis=1),
                np.argmax(p_student.cpu().numpy(), axis=1),
            )
            teacher_accuracy = accuracy_score(
                np.argmax(y_teacher.cpu().numpy(), axis=1),
                np.argmax(p_teacher.cpu().numpy(), axis=1),
            )
            rows.append([
                cycle,
                epoch,
                evaluation_point,
                "accuracy",
                student_accuracy,
                teacher_accuracy,
            ])
            accuracy_message = (
                " student_accuracy={:.6f} teacher_accuracy={:.6f}"
                .format(student_accuracy, teacher_accuracy)
            )

        rows.extend([
            [
                cycle,
                epoch,
                evaluation_point,
                "AUC_{}".format(disease),
                student_value,
                teacher_value,
            ]
            for disease, student_value, teacher_value in zip(
                performance_diseases,
                student_auc,
                teacher_auc,
            )
        ])
        student_mean = _mean_defined_metrics(student_auc)
        teacher_mean = _mean_defined_metrics(teacher_auc)
        rows.append([
            cycle,
            epoch,
            evaluation_point,
            "mAUC",
            student_mean,
            teacher_mean,
        ])
        _append_evaluation_rows(
            os.path.join(
                evaluation_directory,
                dataset,
                "test_performance.csv",
            ),
            rows,
        )
        if shared_val_test_splits[dataset_index]:
            _append_evaluation_rows(
                os.path.join(
                    evaluation_directory,
                    dataset,
                    "val_performance.csv",
                ),
                rows,
            )
        _print_and_log(
            "Test {} {}: student_mean_auroc={:.6f} "
            "teacher_mean_auroc={:.6f}{}"
            .format(
                evaluation_point,
                dataset,
                student_mean,
                teacher_mean,
                accuracy_message,
            ),
            train_log,
        )
        student_means.append(student_mean)
        teacher_means.append(teacher_mean)

    if accelerator.is_main_process:
        _print_and_log(
            "Test summary {} Cycle={}: Student={} Teacher={}".format(
                evaluation_point,
                cycle,
                student_means,
                teacher_means,
            ),
            train_log,
        )
    accelerator.barrier()


def _evaluate_validation_sets(
        student_model,
        teacher,
        dataset_list,
        datasets_config,
        data_loader_list_val,
        val_sampler_list,
        device,
        accelerator,
        evaluation_directory,
        cycle,
        epoch,
        evaluation_point,
        shared_val_test_splits,
        train_log):
    student_val_losses = []
    teacher_val_losses = []
    student_mean_aurocs = []
    teacher_mean_aurocs = []
    if accelerator.is_main_process:
        _print_and_log(
            "Evaluating student and teacher validation sets at {} in Cycle {}"
            .format(evaluation_point, cycle),
            train_log,
        )

    for dataset_index, dataset in enumerate(dataset_list):
        if shared_val_test_splits[dataset_index]:
            student_val_losses.append(None)
            teacher_val_losses.append(None)
            if accelerator.is_main_process:
                _print_and_log(
                    "Skipping separate validation evaluation for {} at {}; "
                    "validation and test samples are identical"
                    .format(dataset, evaluation_point),
                    train_log,
                )
            continue

        diseases = datasets_config[dataset]['diseases']
        multiclass = (
            datasets_config[dataset]['task_type']
            == "multi-class classification"
        )
        criterion = (
            torch.nn.CrossEntropyLoss()
            if multiclass else torch.nn.BCEWithLogitsLoss()
        )
        val_loss, y_val, p_val = evaluate(
            student_model,
            dataset_index,
            data_loader_list_val[dataset_index],
            device,
            criterion,
            dataset,
            return_outputs=True,
            multiclass=multiclass,
            num_classes=len(diseases),
            accelerator=accelerator,
        )
        teacher_val_loss, y_teacher_val, p_teacher_val = evaluate(
            teacher,
            dataset_index,
            data_loader_list_val[dataset_index],
            device,
            criterion,
            dataset,
            return_outputs=True,
            multiclass=multiclass,
            num_classes=len(diseases),
            accelerator=accelerator,
        )
        if accelerator.distributed:
            sample_count = len(val_sampler_list[dataset_index])
            val_loss_stats = torch.tensor(
                [
                    val_loss * sample_count,
                    teacher_val_loss * sample_count,
                    sample_count,
                ],
                dtype=torch.float32,
                device=device,
            )
            accelerator.all_reduce(val_loss_stats)
            val_loss = (val_loss_stats[0] / val_loss_stats[2]).item()
            teacher_val_loss = (
                val_loss_stats[1] / val_loss_stats[2]
            ).item()
        y_val = torch.cat(accelerator.gather_tensor(y_val), 0)
        p_val = torch.cat(accelerator.gather_tensor(p_val), 0)
        y_teacher_val = torch.cat(
            accelerator.gather_tensor(y_teacher_val), 0
        )
        p_teacher_val = torch.cat(
            accelerator.gather_tensor(p_teacher_val), 0
        )
        student_val_losses.append(val_loss)
        teacher_val_losses.append(teacher_val_loss)

        if not accelerator.is_main_process:
            continue

        val_auc_values = metric_AUROC(y_val, p_val, len(diseases))
        teacher_val_auc_values = metric_AUROC(
            y_teacher_val,
            p_teacher_val,
            len(diseases),
        )
        val_mean_auc = (
            _mean_defined_metrics(val_auc_values)
            if val_auc_values else float("nan")
        )
        teacher_val_mean_auc = (
            _mean_defined_metrics(teacher_val_auc_values)
            if teacher_val_auc_values else float("nan")
        )
        val_rows = [[
            cycle,
            epoch,
            evaluation_point,
            "validation_loss",
            val_loss,
            teacher_val_loss,
        ]]
        val_rows.extend([
            [
                cycle,
                epoch,
                evaluation_point,
                "AUC_{}".format(disease),
                student_auc,
                teacher_auc,
            ]
            for disease, student_auc, teacher_auc in zip(
                diseases,
                val_auc_values,
                teacher_val_auc_values,
            )
        ])
        if val_auc_values and teacher_val_auc_values:
            val_rows.append([
                cycle,
                epoch,
                evaluation_point,
                "mAUC",
                val_mean_auc,
                teacher_val_mean_auc,
            ])
        _append_evaluation_rows(
            os.path.join(
                evaluation_directory,
                dataset,
                "val_performance.csv",
            ),
            val_rows,
        )
        _print_and_log(
            "Validation {} {}: student_loss={:.6f} teacher_loss={:.6f} "
            "student_mean_auroc={:.6f} teacher_mean_auroc={:.6f}"
            .format(
                evaluation_point,
                dataset,
                val_loss,
                teacher_val_loss,
                val_mean_auc,
                teacher_val_mean_auc,
            ),
            train_log,
        )
        student_mean_aurocs.append(val_mean_auc)
        teacher_mean_aurocs.append(teacher_val_mean_auc)

    if accelerator.is_main_process:
        _print_and_log(
            "Validation summary {} Cycle={}: student_losses={} "
            "teacher_losses={} student_mean_aurocs={} teacher_mean_aurocs={}"
            .format(
                evaluation_point,
                cycle,
                student_val_losses,
                teacher_val_losses,
                student_mean_aurocs,
                teacher_mean_aurocs,
            ),
            train_log,
        )
    accelerator.barrier()
    return student_val_losses


def _checkpoint_cycle(file_path):
    match = re.fullmatch(r"cycle_(\d+)(?:_[^/]+)?\.pth\.tar", os.path.basename(file_path))
    return int(match.group(1)) if match else None


def _checkpoint_stem(args, cycle):
    return "cycle_{:04d}_{}_seed_{}".format(cycle, args.exp_name, args.seed)


def _run_metadata(args, dataset_list, num_classes_list, cycle):
    return {
        'cycle': cycle,
        'dataset_list': list(dataset_list),
        'num_classes_list': list(num_classes_list),
        'model_name': args.model_name,
        'projector_features': args.projector_features,
        'use_mlp': bool(args.use_mlp),
        'run_name': args.exp_name,
        'seed': args.seed,
    }


def _find_checkpoint_for_cycle(checkpoint_directory, cycle):
    checkpoint_paths = [
        path for path in glob.glob(os.path.join(checkpoint_directory, "cycle_*.pth.tar"))
        if _checkpoint_cycle(path) == cycle
    ]
    if not checkpoint_paths:
        raise FileNotFoundError(
            "No checkpoint found for cycle {} in {}".format(cycle, checkpoint_directory)
        )
    checkpoint_paths.sort()
    return checkpoint_paths[0]


def _load_checkpoint_metadata(file_path):
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    required_keys = {'epoch', 'lossMIN', 'state_dict', 'teacher', 'optimizer', 'scheduler'}
    missing_keys = required_keys.difference(checkpoint.keys())
    if missing_keys:
        raise ValueError(
            "Checkpoint '{}' is missing keys {}".format(
                file_path,
                sorted(missing_keys),
            )
        )
    filename_cycle = _checkpoint_cycle(file_path)
    checkpoint_cycle = checkpoint.get('cycle', filename_cycle)
    if filename_cycle is None or checkpoint_cycle != filename_cycle:
        raise ValueError("Checkpoint '{}' has an invalid cycle number".format(file_path))
    return checkpoint


def _find_latest_valid_checkpoint(checkpoint_directory):
    checkpoint_paths = [
        path for path in glob.glob(os.path.join(checkpoint_directory, "cycle_*.pth.tar"))
        if _checkpoint_cycle(path) is not None
    ]
    checkpoint_paths.sort(key=_checkpoint_cycle, reverse=True)
    for checkpoint_path in checkpoint_paths:
        try:
            checkpoint = _load_checkpoint_metadata(checkpoint_path)
        except Exception:
            continue
        return checkpoint_path, int(checkpoint['cycle'])
    raise FileNotFoundError(
        "No valid cycle checkpoint found in {}".format(checkpoint_directory)
    )


def _checkpoint_run_directory(checkpoint_path):
    checkpoint_directory = os.path.dirname(checkpoint_path)
    if os.path.basename(checkpoint_directory) != "checkpoints" or os.path.basename(os.path.dirname(checkpoint_directory)) != "models":
        raise ValueError(
            "Checkpoint must be inside <run>/models/checkpoints: {}".format(checkpoint_path)
        )
    return os.path.dirname(os.path.dirname(checkpoint_directory))


def _scheduler_config(args):
    return {
        'schedule': 'linear_warmup_cosine_decay',
        'pretrain_epochs': args.pretrain_epochs,
        'peak_lr': args.lr,
        'warmup_lr': args.warmup_lr,
        'min_lr': args.min_lr,
        'warmup_epochs': args.warmup_epochs,
    }


class WarmupCosineScheduler:
    """Cycle-based linear warmup followed by cosine learning-rate decay."""

    def __init__(self, optimizer, total_cycles, warmup_cycles, start_lr,
                 peak_lr, end_lr):
        if total_cycles < 3:
            raise ValueError("The learning-rate schedule requires at least 3 cycles")
        if warmup_cycles < 2 or warmup_cycles >= total_cycles:
            raise ValueError(
                "warmup_epochs must be at least 2 and less than pretrain_epochs"
            )
        if start_lr > peak_lr or end_lr > peak_lr:
            raise ValueError(
                "warmup_lr and min_lr must not exceed the peak learning rate"
            )

        self.optimizer = optimizer
        self.total_cycles = total_cycles
        self.warmup_cycles = warmup_cycles
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.last_cycle = -1

    def learning_rate(self, cycle):
        if cycle < 0 or cycle >= self.total_cycles:
            raise ValueError(
                "Cycle {} is outside the configured range [0, {})".format(
                    cycle, self.total_cycles
                )
            )

        if cycle < self.warmup_cycles:
            warmup_progress = cycle / float(self.warmup_cycles - 1)
            return self.start_lr + (
                self.peak_lr - self.start_lr
            ) * warmup_progress

        decay_cycles = self.total_cycles - self.warmup_cycles
        decay_progress = (
            cycle - self.warmup_cycles + 1
        ) / float(decay_cycles)
        return self.end_lr + 0.5 * (
            self.peak_lr - self.end_lr
        ) * (1.0 + np.cos(np.pi * decay_progress))

    def step(self, cycle):
        learning_rate = self.learning_rate(cycle)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        self.last_cycle = cycle
        return learning_rate

    def state_dict(self):
        return {
            'last_cycle': self.last_cycle,
            'total_cycles': self.total_cycles,
            'warmup_cycles': self.warmup_cycles,
            'start_lr': self.start_lr,
            'peak_lr': self.peak_lr,
            'end_lr': self.end_lr,
        }

    def load_state_dict(self, state_dict):
        expected = self.state_dict()
        for key in (
                'total_cycles', 'warmup_cycles', 'start_lr', 'peak_lr',
                'end_lr'):
            if state_dict.get(key) != expected[key]:
                raise ValueError(
                    "Learning-rate scheduler state has incompatible {}".format(key)
                )
        self.last_cycle = state_dict.get('last_cycle', -1)


def _validate_checkpoint_total_cycles(checkpoint, file_path, args):
    total_cycles = checkpoint.get('total_cycles')
    if total_cycles is None:
        raise ValueError(
            "Checkpoint '{}' does not record the total cycle count".format(file_path)
        )
    if total_cycles != args.pretrain_epochs:
        raise ValueError(
            "Checkpoint '{}' was configured for {} cycles, but this run requests {}"
            .format(file_path, total_cycles, args.pretrain_epochs)
        )


def _rewrite_csv_through_cycle(file_path, max_cycle):
    if not os.path.isfile(file_path):
        return
    with open(file_path, 'r', newline='') as file_descriptor:
        rows = list(csv.reader(file_descriptor))
    if not rows:
        return
    filtered_rows = [rows[0]]
    for row in rows[1:]:
        try:
            cycle = int(row[0])
        except (IndexError, ValueError):
            continue
        if cycle <= max_cycle:
            filtered_rows.append(row)
    with tempfile.NamedTemporaryFile(
            mode='w',
            newline='',
            dir=os.path.dirname(file_path),
            delete=False) as temporary_file:
        writer = csv.writer(temporary_file)
        writer.writerows(filtered_rows)
        temporary_path = temporary_file.name
    os.replace(temporary_path, file_path)


def _trim_train_log_through_cycle(file_path, max_cycle):
    if not os.path.isfile(file_path):
        return
    with open(file_path, 'r') as file_descriptor:
        lines = file_descriptor.readlines()
    filtered_lines = []
    for line in lines:
        match = re.match(
            r"^(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO )?"
            r"Cycle\s+(\d+)(?::|\s+\|)",
            line,
        )
        if match and int(match.group(1)) > max_cycle:
            break
        filtered_lines.append(line)
    with tempfile.NamedTemporaryFile(
            mode='w',
            dir=os.path.dirname(file_path),
            delete=False) as temporary_file:
        temporary_file.writelines(filtered_lines)
        temporary_path = temporary_file.name
    os.replace(temporary_path, file_path)


def _remove_cycle_directories_after(directory, prefix, max_cycle):
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        match = re.fullmatch(r"{}_([0-9]+)".format(prefix), name)
        if match and int(match.group(1)) > max_cycle:
            path = os.path.join(directory, name)
            if os.path.isdir(path):
                shutil.rmtree(path)


def _remove_checkpoints_after(directory, max_cycle):
    if not os.path.isdir(directory):
        return
    for path in glob.glob(os.path.join(directory, "cycle_*.pth.tar")):
        cycle = _checkpoint_cycle(path)
        if cycle is not None and cycle > max_cycle:
            os.remove(path)


def _remove_stale_latest_checkpoints(directory, current_cycle, checkpoint_frequency):
    if not os.path.isdir(directory):
        return
    for path in glob.glob(os.path.join(directory, "cycle_*.pth.tar")):
        cycle = _checkpoint_cycle(path)
        if (
                cycle is not None
                and cycle != current_cycle
                and cycle % checkpoint_frequency != 0):
            os.remove(path)


def _discard_outputs_after_cycle(run_directory, max_cycle):
    _remove_cycle_directories_after(
        os.path.join(run_directory, "snapshots"),
        "cycle",
        max_cycle,
    )
    _remove_cycle_directories_after(
        os.path.join(run_directory, "models", "weights"),
        "epoch",
        max_cycle,
    )
    _remove_checkpoints_after(
        os.path.join(run_directory, "models", "checkpoints"),
        max_cycle,
    )
    _rewrite_csv_through_cycle(os.path.join(run_directory, "loss.csv"), max_cycle)
    evaluation_directory = os.path.join(run_directory, "evaluation")
    if os.path.isdir(evaluation_directory):
        for dataset in os.listdir(evaluation_directory):
            dataset_directory = os.path.join(evaluation_directory, dataset)
            if os.path.isdir(dataset_directory):
                for file_name in os.listdir(dataset_directory):
                    if file_name.endswith('.csv'):
                        _rewrite_csv_through_cycle(
                            os.path.join(dataset_directory, file_name),
                            max_cycle,
                        )
    _trim_train_log_through_cycle(
        os.path.join(run_directory, "train.log"),
        max_cycle,
    )


def _copy_cycle_directories(source_directory, target_directory, prefix, max_cycle):
    if not os.path.isdir(source_directory):
        return
    for name in os.listdir(source_directory):
        match = re.fullmatch(r"{}_([0-9]+)".format(prefix), name)
        if not match or int(match.group(1)) > max_cycle:
            continue
        source_path = os.path.join(source_directory, name)
        target_path = os.path.join(target_directory, name)
        if os.path.isdir(source_path):
            os.makedirs(target_directory, exist_ok=True)
            shutil.copytree(source_path, target_path)
        else:
            os.makedirs(target_directory, exist_ok=True)
            shutil.copy2(source_path, target_path)


def _copy_checkpoint_files(source_directory, target_directory, max_cycle):
    if not os.path.isdir(source_directory):
        return
    for source_path in glob.glob(os.path.join(source_directory, "cycle_*.pth.tar")):
        cycle = _checkpoint_cycle(source_path)
        if cycle is None or cycle > max_cycle:
            continue
        os.makedirs(target_directory, exist_ok=True)
        shutil.copy2(source_path, os.path.join(target_directory, os.path.basename(source_path)))


def _copy_run_through_cycle(source_directory, target_directory, max_cycle):
    os.makedirs(target_directory, exist_ok=True)
    for file_name in ("train.log", "loss.csv"):
        source_path = os.path.join(source_directory, file_name)
        target_path = os.path.join(target_directory, file_name)
        if os.path.isfile(source_path):
            shutil.copy2(source_path, target_path)

    _copy_cycle_directories(
        os.path.join(source_directory, "snapshots"),
        os.path.join(target_directory, "snapshots"),
        "cycle",
        max_cycle,
    )
    _copy_cycle_directories(
        os.path.join(source_directory, "models", "weights"),
        os.path.join(target_directory, "models", "weights"),
        "epoch",
        max_cycle,
    )
    _copy_checkpoint_files(
        os.path.join(source_directory, "models", "checkpoints"),
        os.path.join(target_directory, "models", "checkpoints"),
        max_cycle,
    )

    source_evaluation_directory = os.path.join(source_directory, "evaluation")
    if os.path.isdir(source_evaluation_directory):
        for dataset in os.listdir(source_evaluation_directory):
            source_dataset_directory = os.path.join(source_evaluation_directory, dataset)
            if not os.path.isdir(source_dataset_directory):
                continue
            target_dataset_directory = os.path.join(target_directory, "evaluation", dataset)
            os.makedirs(target_dataset_directory, exist_ok=True)
            for file_name in os.listdir(source_dataset_directory):
                if file_name.endswith('.csv'):
                    source_path = os.path.join(source_dataset_directory, file_name)
                    target_path = os.path.join(target_dataset_directory, file_name)
                    shutil.copy2(source_path, target_path)


def _run_main_process_resume_setup(accelerator, device, setup):
    setup_status = torch.zeros(1, dtype=torch.int32, device=device)
    if accelerator.is_main_process:
        try:
            setup()
        except Exception as error:
            print("Resume setup failed: {}".format(error))
        else:
            setup_status[0] = 1
    accelerator.broadcast(setup_status)
    if setup_status.item() == 0:
        raise RuntimeError(
            "Resume setup failed on rank 0; see rank 0 output for details"
        )


def _is_saved_weight_file(file_path):
    return (
        os.path.basename(file_path) in ('student.pth', 'teacher.pth')
        or re.fullmatch(
            r"(?:student|teacher)_cycle_.+\.pth",
            os.path.basename(file_path),
        ) is not None
    )


def _weight_cycle(weights_directory):
    match = re.fullmatch(
        r"epoch_(\d+)",
        os.path.basename(os.path.normpath(weights_directory)),
    )
    return int(match.group(1)) if match else None


def _load_weight_manifest(weights_directory):
    manifest_path = os.path.join(weights_directory, 'manifest.json')
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "Saved weights require manifest.json: {}".format(weights_directory)
        )
    with open(manifest_path, 'r') as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict):
        raise ValueError(
            "Weight manifest '{}' is not a JSON object".format(manifest_path)
        )
    required_keys = {
        'cycle',
        'dataset_list',
        'num_classes_list',
        'model_name',
        'projector_features',
        'use_mlp',
        'run_name',
        'seed',
    }
    missing_keys = required_keys.difference(manifest.keys())
    if missing_keys:
        raise ValueError(
            "Weight manifest '{}' is missing keys {}"
            .format(manifest_path, sorted(missing_keys))
        )
    directory_cycle = _weight_cycle(weights_directory)
    if not isinstance(manifest['cycle'], int) or manifest['cycle'] < 1:
        raise ValueError(
            "Weight manifest '{}' has an invalid cycle".format(manifest_path)
        )
    if directory_cycle is not None and manifest['cycle'] != directory_cycle:
        raise ValueError(
            "Weight manifest '{}' does not match its directory cycle"
            .format(manifest_path)
        )
    if (
            not isinstance(manifest['run_name'], str)
            or manifest['run_name'] in ('.', '..')
            or re.fullmatch(r"[A-Za-z0-9._-]+", manifest['run_name']) is None):
        raise ValueError(
            "Weight manifest '{}' has an unsafe run name"
            .format(manifest_path)
        )
    if not isinstance(manifest['seed'], int):
        raise ValueError(
            "Weight manifest '{}' has an invalid seed".format(manifest_path)
        )
    return manifest


def _test_output_path(output_directory, checkpoint_path):
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_path))
    if os.path.isdir(checkpoint_path):
        manifest = _load_weight_manifest(checkpoint_path)
        return os.path.join(
            output_directory,
            "cycle_{:04d}_{}_seed_{}.csv".format(
                manifest['cycle'],
                manifest['run_name'],
                manifest['seed'],
            ),
        )
    if _is_saved_weight_file(checkpoint_path):
        weights_directory = os.path.dirname(checkpoint_path)
        manifest = _load_weight_manifest(weights_directory)
        return os.path.join(
            output_directory,
            "cycle_{:04d}_{}_seed_{}.csv".format(
                manifest['cycle'],
                manifest['run_name'],
                manifest['seed'],
            ),
        )
    if checkpoint_name.endswith('.pth.tar'):
        result_name = checkpoint_name[:-len('.pth.tar')] + '.csv'
    else:
        result_name = os.path.splitext(checkpoint_name)[0] + '.csv'
    return os.path.join(output_directory, result_name)


def _find_saved_weight(weights_directory, role):
    legacy_path = os.path.join(weights_directory, "{}.pth".format(role))
    if os.path.isfile(legacy_path):
        return legacy_path
    weight_paths = glob.glob(
        os.path.join(weights_directory, "{}_cycle_*.pth".format(role))
    )
    if len(weight_paths) != 1:
        raise FileNotFoundError(
            "Expected one {} weight file in {}".format(role, weights_directory)
        )
    return weight_paths[0]


def _load_test_state_dicts(weights_path):
    if os.path.isfile(weights_path) and _is_saved_weight_file(weights_path):
        weights_path = os.path.dirname(weights_path)
    if os.path.isdir(weights_path):
        manifest = _load_weight_manifest(weights_path)
        student_path = _find_saved_weight(weights_path, 'student')
        teacher_path = _find_saved_weight(weights_path, 'teacher')
        student_state_dict = torch.load(
            student_path,
            map_location='cpu',
            weights_only=False,
        )
        teacher_state_dict = torch.load(
            teacher_path,
            map_location='cpu',
            weights_only=False,
        )
        return student_state_dict, teacher_state_dict, manifest

    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    for key in ('state_dict', 'teacher'):
        if key not in checkpoint:
            raise ValueError(
                "Test checkpoint '{}' is missing '{}'".format(weights_path, key)
            )
    return checkpoint['state_dict'], checkpoint['teacher'], checkpoint


def _validate_test_metadata(metadata, file_path, args, dataset_list, num_classes_list, require_model_metadata=False):
    if not isinstance(metadata, dict):
        raise ValueError(
            "Test weights '{}' do not contain a metadata object".format(file_path)
        )
    required_keys = {'dataset_list', 'num_classes_list'}
    if require_model_metadata:
        required_keys.update({
            'cycle',
            'model_name',
            'projector_features',
            'use_mlp',
            'run_name',
            'seed',
        })
    missing_keys = required_keys.difference(metadata.keys())
    if missing_keys:
        raise ValueError(
            "Test weights '{}' are missing metadata keys {}"
            .format(file_path, sorted(missing_keys))
        )
    if not isinstance(metadata['dataset_list'], (list, tuple)):
        raise ValueError(
            "Test weights '{}' have invalid dataset metadata".format(file_path)
        )
    if not isinstance(metadata['num_classes_list'], (list, tuple)):
        raise ValueError(
            "Test weights '{}' have invalid class-count metadata".format(file_path)
        )
    if list(metadata['dataset_list']) != list(dataset_list):
        raise ValueError(
            "Test weights '{}' were trained for datasets {}, requested {}"
            .format(file_path, metadata['dataset_list'], list(dataset_list))
        )
    if list(metadata['num_classes_list']) != list(num_classes_list):
        raise ValueError(
            "Test weights '{}' have class counts {}, requested {}"
            .format(file_path, metadata['num_classes_list'], num_classes_list)
        )
    artifact_cycle = _checkpoint_cycle(file_path)
    if artifact_cycle is None:
        weights_directory = (
            file_path
            if os.path.isdir(file_path)
            else os.path.dirname(file_path)
        )
        artifact_cycle = _weight_cycle(weights_directory)
    if artifact_cycle is not None and metadata.get('cycle') != artifact_cycle:
        raise ValueError(
            "Test weights '{}' have metadata cycle {}, but the artifact is cycle {}"
            .format(file_path, metadata.get('cycle'), artifact_cycle)
        )
    for key, expected_value in (
            ('model_name', args.model_name),
            ('projector_features', args.projector_features),
            ('use_mlp', bool(args.use_mlp))):
        if key in metadata and metadata[key] != expected_value:
            raise ValueError(
                "Test weights '{}' have {}={}, requested {}"
                .format(file_path, key, metadata[key], expected_value)
            )


def _mean_defined_metrics(values):
    defined_values = [value for value in values if np.isfinite(value)]
    return np.mean(defined_values) if defined_values else np.nan


def _raw_label_counts(dataset, split_key, file_path, num_classes):
    categories = ("positive", "negative", "uncertain", "missing")
    label_counts = [dict.fromkeys(categories, 0) for _ in range(num_classes)]
    sample_count = 0

    with open(file_path, "r", newline="") as file_descriptor:
        csv_reader = csv.reader(file_descriptor)
        next(csv_reader, None)
        for line in csv_reader:
            if dataset == "CheXpert" and split_key == "test_list":
                labels = line[1:]
            else:
                labels = line[5:]
            if len(labels) < num_classes:
                raise ValueError(
                    "{} has {} labels; expected {}".format(
                        file_path, len(labels), num_classes
                    )
                )
            sample_count += 1
            for class_index, value in enumerate(labels[:num_classes]):
                value = value.strip()
                if not value:
                    category = "missing"
                elif float(value) == 1:
                    category = "positive"
                elif float(value) == 0:
                    category = "negative"
                elif float(value) == -1:
                    category = "uncertain"
                else:
                    raise ValueError("Unsupported label value {!r} in {}".format(value, file_path))
                label_counts[class_index][category] += 1

    return sample_count, label_counts


def print_label_summary(dataset_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list, log_file=None):
    _print_and_log("Label distribution:", log_file)
    for dataset_index, dataset in enumerate(dataset_list, start=1):
        diseases = datasets_config[dataset]['diseases']
        _print_and_log(
            "Dataset {}/{}: {}".format(dataset_index, len(dataset_list), dataset),
            log_file,
        )
        split_datasets = (
            ("Train", "train_list", dataset_train_list[dataset_index - 1]),
            ("Validation", "val_list", dataset_val_list[dataset_index - 1]),
            ("Test", "test_list", dataset_test_list[dataset_index - 1]),
        )
        uses_test_for_validation = (
            datasets_config[dataset]['val_list'] == datasets_config[dataset]['test_list']
        )
        for split_name, split_key, split_dataset in split_datasets:
            if split_name == "Validation" and uses_test_for_validation:
                split_name = "Validation (test split)"
            sample_count = len(split_dataset)
            _print_and_log(
                "  {}: {} samples".format(split_name, sample_count),
                log_file,
            )
            if sample_count == 0:
                continue

            if dataset in ("CheXpert", "MIMIC"):
                file_path = datasets_config[dataset][split_key]
                label_counts_sample_count, label_counts = _raw_label_counts(
                    dataset,
                    split_key,
                    file_path,
                    len(diseases),
                )
                if label_counts_sample_count != sample_count:
                    raise ValueError(
                        "{} {} has {} parsed labels but dataset has {} samples".format(
                            dataset, split_name, label_counts_sample_count, sample_count
                        )
                    )
                for disease, counts in zip(diseases, label_counts):
                    percentages = {
                        category: 100.0 * counts[category] / sample_count
                        for category in counts
                    }
                    _print_and_log(
                        "    {}: positive {} ({:.1f}%), negative {} ({:.1f}%), "
                        "uncertain {} ({:.1f}%), missing {} ({:.1f}%)".format(
                            disease,
                            counts["positive"],
                            percentages["positive"],
                            counts["negative"],
                            percentages["negative"],
                            counts["uncertain"],
                            percentages["uncertain"],
                            counts["missing"],
                            percentages["missing"],
                        ),
                        log_file,
                    )
            else:
                labels = np.asarray(split_dataset.img_label, dtype=np.float64)
                for class_index, disease in enumerate(diseases):
                    positive_count = int(np.count_nonzero(labels[:, class_index] >= 0.5))
                    negative_count = sample_count - positive_count
                    _print_and_log(
                        "    {}: positive {} ({:.1f}%), negative {} ({:.1f}%)".format(
                            disease,
                            positive_count,
                            100.0 * positive_count / sample_count,
                            negative_count,
                            100.0 * negative_count / sample_count,
                        ),
                        log_file,
                    )


def print_training_configuration(args, model_path, dataset_list,
                                 dataset_train_list, dataset_val_list,
                                 dataset_test_list, num_classes_list,
                                 train_batch_size, accelerator, log_file=None):
    separator = "=" * 80
    _print_and_log(separator, log_file)
    _print_and_log("ARK+ PRETRAINING CONFIGURATION", log_file)
    _print_and_log(separator, log_file)
    _print_and_log("Run Name: {}".format(args.exp_name), log_file)
    _print_and_log("Seed: {}".format(args.seed), log_file)
    _print_and_log("Output Directory: {}".format(os.path.abspath(model_path)), log_file)
    _print_and_log("Backbone: {}".format(args.model_name), log_file)
    _print_and_log("Initialization: {}".format(args.init), log_file)
    _print_and_log("Pretrained Weights: {}".format(bool(args.pretrained_weights)), log_file)
    _print_and_log("Projector Features: {}".format(args.projector_features), log_file)
    _print_and_log("Datasets: {}".format(dataset_list), log_file)
    _print_and_log("Class Counts: {}".format(num_classes_list), log_file)
    _print_and_log("Global Batch Size: {}".format(args.batch_size), log_file)
    _print_and_log("Per-rank Batch Size: {}".format(train_batch_size), log_file)
    _print_and_log("Distributed World Size: {}".format(accelerator.world_size), log_file)
    _print_and_log("Data Loader Workers Per Rank: {}".format(args.workers), log_file)
    _print_and_log("Cycles: {}".format(args.pretrain_epochs), log_file)
    _print_and_log("Optimizer: {}".format(args.opt), log_file)
    _print_and_log("Initial Learning Rate: {}".format(args.warmup_lr), log_file)
    _print_and_log("Peak Learning Rate: {}".format(args.lr), log_file)
    _print_and_log("Final Learning Rate: {}".format(args.min_lr), log_file)
    _print_and_log("Warmup Cycles: {}".format(args.warmup_epochs), log_file)
    _print_and_log("Teacher Momentum Base: {}".format(args.momentum_teacher), log_file)
    _print_and_log("Teacher EMA Mode: {}".format(args.ema_mode), log_file)
    _print_and_log(
        "Validation Evaluation: student and teacher after every dataset",
        log_file,
    )
    _print_and_log("Test Evaluation: after every dataset", log_file)
    _print_and_log("Saved Weights: every cycle", log_file)
    _print_and_log(
        "Resumable Checkpoints: every 10 cycles plus the latest completed cycle",
        log_file,
    )
    _print_and_log("Precision: fp32", log_file)
    _print_and_log("Device: {}".format(accelerator.device), log_file)
    _print_and_log(
        "Execution Mode: {}".format(
            "lazy" if accelerator.lazy_mode else "eager",
        ),
        log_file,
    )
    for dataset, train_dataset, val_dataset, test_dataset, num_classes in zip(
            dataset_list,
            dataset_train_list,
            dataset_val_list,
            dataset_test_list,
            num_classes_list):
        _print_and_log(
            "Dataset {}: train={:,} val={:,} test={:,} labels={}".format(
                dataset,
                len(train_dataset),
                len(val_dataset),
                len(test_dataset),
                num_classes,
            ),
            log_file,
        )
    _print_and_log(separator, log_file)


def omni_engine(args, model_path, output_path, dataset_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list):
    training_start_time = time.time()
    accelerator = Accelerator(args.device)
    accelerator.initialize_distributed()
    device = accelerator.device
    if accelerator.is_cuda:
        cudnn.benchmark = True

    # logs
    exp = 'Ark_Plus'
    for dataset in dataset_list:
        exp += '_' + dataset 
    resume_requested = (
        args.mode == "train"
        and (args.resume or getattr(args, "resume_from", None) is not None)
    )
    resume_checkpoint_path = None
    resume_cycle = None
    test_output_path = None
    if args.mode == "test":
        test_output_path = _test_output_path(output_path, args.pretrained_weights)
        result_status = torch.zeros(1, dtype=torch.int32, device=device)
        if accelerator.is_main_process:
            result_status[0] = int(not os.path.exists(test_output_path))
        accelerator.broadcast(result_status)
        if result_status.item() == 0:
            raise FileExistsError(
                "Test results file already exists: {}".format(test_output_path)
            )
    elif args.mode != "train":
        model_path = os.path.join(model_path, exp)
        model_path = os.path.join(model_path, args.exp_name)
    elif resume_requested and getattr(args, "resume_from", None) is not None:
        source_checkpoint = os.path.realpath(os.path.abspath(args.resume_from))
        target_run_directory = os.path.realpath(os.path.abspath(model_path))
        resume_cycle_tensor = torch.full((1,), -1, dtype=torch.int64, device=device)

        def setup_explicit_resume():
            source_run_directory = os.path.realpath(
                _checkpoint_run_directory(source_checkpoint)
            )
            selected_cycle = _checkpoint_cycle(source_checkpoint)
            if selected_cycle is None or not os.path.isfile(source_checkpoint):
                raise FileNotFoundError(
                    "Cannot resume because checkpoint does not exist or is not a cycle checkpoint: {}"
                    .format(source_checkpoint)
                )
            selected_checkpoint = _load_checkpoint_metadata(source_checkpoint)
            _validate_checkpoint_total_cycles(selected_checkpoint, source_checkpoint, args)
            if selected_cycle >= selected_checkpoint['total_cycles']:
                raise RuntimeError(
                    "Cannot resume from completed checkpoint '{}'".format(source_checkpoint)
                )

            if target_run_directory == source_run_directory:
                if not os.path.isdir(model_path):
                    raise FileNotFoundError(
                        "Cannot resume training because run directory does not exist: {}".format(model_path)
                    )
                _, latest_cycle = _find_latest_valid_checkpoint(
                    os.path.join(model_path, "models", "checkpoints")
                )
                if selected_cycle != latest_cycle:
                    raise ValueError(
                        "Resuming from a checkpoint before the latest requires a new --exp_name"
                    )
                _discard_outputs_after_cycle(model_path, selected_cycle)
            else:
                if os.path.exists(model_path):
                    raise FileExistsError(
                        "Training run directory already exists: {}".format(model_path)
                    )
                _copy_run_through_cycle(
                    source_run_directory,
                    model_path,
                    selected_cycle,
                )
                _discard_outputs_after_cycle(model_path, selected_cycle)

            resume_cycle_tensor[0] = selected_cycle

        _run_main_process_resume_setup(accelerator, device, setup_explicit_resume)
        accelerator.broadcast(resume_cycle_tensor)
        resume_cycle = int(resume_cycle_tensor.item())
        resume_checkpoint_path = _find_checkpoint_for_cycle(
            os.path.join(model_path, "models", "checkpoints"),
            resume_cycle,
        )
        accelerator.barrier()
    elif resume_requested:
        resume_cycle_tensor = torch.full((1,), -1, dtype=torch.int64, device=device)

        def setup_latest_resume():
            if not os.path.isdir(model_path):
                raise FileNotFoundError(
                    "Cannot resume training because run directory does not exist: {}".format(model_path)
                )
            latest_checkpoint, latest_cycle = _find_latest_valid_checkpoint(
                os.path.join(model_path, "models", "checkpoints")
            )
            latest_metadata = _load_checkpoint_metadata(latest_checkpoint)
            _validate_checkpoint_total_cycles(latest_metadata, latest_checkpoint, args)
            if latest_cycle >= latest_metadata['total_cycles']:
                raise RuntimeError(
                    "Cannot resume completed run '{}'".format(model_path)
                )
            _discard_outputs_after_cycle(model_path, latest_cycle)
            resume_cycle_tensor[0] = latest_cycle

        _run_main_process_resume_setup(accelerator, device, setup_latest_resume)
        accelerator.broadcast(resume_cycle_tensor)
        resume_cycle = int(resume_cycle_tensor.item())
        resume_checkpoint_path = _find_checkpoint_for_cycle(
            os.path.join(model_path, "models", "checkpoints"),
            resume_cycle,
        )
        accelerator.barrier()
    else:
        run_status = torch.zeros(1, dtype=torch.int32, device=device)
        if accelerator.is_main_process:
            run_status[0] = int(not os.path.exists(model_path))
        accelerator.broadcast(run_status)
        if run_status.item() == 0:
            raise FileExistsError(
                "Training run directory already exists: {}".format(model_path)
            )

    if accelerator.is_main_process:
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if args.mode == "train":
            os.makedirs(os.path.join(model_path, "models", "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(model_path, "models", "weights"), exist_ok=True)
            for dataset in dataset_list:
                os.makedirs(os.path.join(model_path, "evaluation", dataset), exist_ok=True)
    accelerator.barrier()

    log_file = os.path.join(model_path, "train.log")
    checkpoint_directory = os.path.join(model_path, "models", "checkpoints")
    weights_directory = os.path.join(model_path, "models", "weights")
    evaluation_directory = os.path.join(model_path, "evaluation")
    train_log = None
    loss_file = None
    loss_writer = None
    if accelerator.is_main_process:
        train_log = _configure_logger(log_file if args.mode == "train" else None)
    if args.mode == "train" and accelerator.is_main_process:
        loss_csv_path = os.path.join(output_path, "loss.csv")
        loss_csv_exists = os.path.exists(loss_csv_path) and os.path.getsize(loss_csv_path) > 0
        loss_file = open(loss_csv_path, 'a', newline='')
        loss_writer = csv.writer(loss_file)
        if not loss_csv_exists:
            loss_writer.writerow([
                "cycle",
                "epoch",
                "global_step",
                "batch",
                "dataset",
                "samples",
                "classification_loss",
                "consistency_loss",
                "total_loss",
                "classification_percent",
                "consistency_percent",
                "learning_rate",
                "momentum",
            ])
            loss_file.flush()

    # dataloaders for pretraining
    if accelerator.distributed:
        if args.batch_size % accelerator.world_size != 0:
            raise ValueError("Global batch size must be divisible by world size")
        train_batch_size = args.batch_size // accelerator.world_size
    else:
        train_batch_size = args.batch_size

    data_loader_list_train = []
    train_sampler_list = []
    for d in dataset_train_list:
        train_sampler = DistributedSampler(
            d,
            num_replicas=accelerator.world_size,
            rank=accelerator.rank,
            shuffle=True,
            seed=args.seed,
        ) if accelerator.distributed else None
        train_sampler_list.append(train_sampler)
        data_loader_list_train.append(DataLoader(dataset=d, batch_size=train_batch_size, shuffle=train_sampler is None,
                                        sampler=train_sampler,
                                        num_workers=args.workers, pin_memory=accelerator.pin_memory))
    data_loader_list_val = []
    val_sampler_list = []
    for dv in dataset_val_list:
        val_sampler = DistributedEvaluationSampler(
            dv,
            num_replicas=accelerator.world_size,
            rank=accelerator.rank,
        ) if accelerator.distributed else None
        val_sampler_list.append(val_sampler)
        data_loader_list_val.append(DataLoader(dataset=dv, batch_size=train_batch_size if accelerator.distributed else args.batch_size, shuffle=False,
                                        sampler=val_sampler,
                                        num_workers=args.workers, pin_memory=accelerator.pin_memory))
    data_loader_list_test = []
    for dt in dataset_test_list: 
        test_sampler = DistributedEvaluationSampler(
            dt,
            num_replicas=accelerator.world_size,
            rank=accelerator.rank,
        ) if accelerator.distributed else None
        test_batch_size = max(1, int(train_batch_size / 2)) if accelerator.distributed else int(args.batch_size/2)
        data_loader_list_test.append(DataLoader(dataset=dt, batch_size=test_batch_size, shuffle=False,
                                        sampler=test_sampler,
                                        num_workers=args.workers, pin_memory=accelerator.pin_memory))

    shared_val_test_splits = [False] * len(dataset_list)
    if args.mode == "train":
        shared_val_test_splits = [
            _datasets_have_identical_samples(validation_dataset, test_dataset)
            for validation_dataset, test_dataset in zip(
                dataset_val_list,
                dataset_test_list,
            )
        ]

    num_classes_list = [len(datasets_config[dataset]['diseases']) for dataset in dataset_list]
    if accelerator.is_main_process:
        if args.mode == "train":
            print_training_configuration(
                args,
                model_path,
                dataset_list,
                dataset_train_list,
                dataset_val_list,
                dataset_test_list,
                num_classes_list,
                train_batch_size,
                accelerator,
                train_log,
            )
            print_label_summary(
                dataset_list,
                datasets_config,
                dataset_train_list,
                dataset_val_list,
                dataset_test_list,
                train_log,
            )
            for dataset, splits_are_shared in zip(
                    dataset_list, shared_val_test_splits):
                _print_and_log(
                    "Dataset {}: validation/test samples identical = {}{}"
                    .format(
                        dataset,
                        splits_are_shared,
                        (
                            "; test evaluation will also be recorded as validation"
                            if splits_are_shared else ""
                        ),
                    ),
                    train_log,
                )
        else:
            _print_and_log("Class Counts: {}".format(num_classes_list), train_log)


    # training setups
    model_args = args
    test_state_dicts = None
    if args.mode == "test":
        model_args = copy.copy(args)
        model_args.pretrained_weights = None
        test_state_dicts = _load_test_state_dicts(args.pretrained_weights)
        _validate_test_metadata(
            test_state_dicts[2],
            args.pretrained_weights,
            args,
            dataset_list,
            num_classes_list,
            require_model_metadata=(
                os.path.isdir(args.pretrained_weights)
                or _is_saved_weight_file(args.pretrained_weights)
            ),
        )
    model = build_omni_model(model_args, num_classes_list)
    teacher = build_omni_model(model_args, num_classes_list)
    if test_state_dicts is not None:
        model.load_state_dict(
            accelerator.strip_module_prefix(test_state_dicts[0]),
            strict=True,
        )
        teacher.load_state_dict(
            accelerator.strip_module_prefix(test_state_dicts[1]),
            strict=True,
        )
    model.to(device)
    teacher.to(device)
    model = accelerator.wrap_model(model)
    student_model = accelerator.unwrap_model(model)
    accelerator.synchronize_model(teacher)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    if accelerator.is_main_process:
        _print_and_log(
            "Student and Teacher are built: they are both {} network.".format(args.model_name),
            train_log,
        )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = None
    optimizer = None
    lr_scheduler = None
    if args.mode == "train":
        if args.ema_mode == "epoch":
            momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                                   args.pretrain_epochs, len(dataset_list))
        elif args.ema_mode == "iteration":
            iters_per_epoch = 0
            for d in data_loader_list_train:
                iters_per_epoch += len(d)
            momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                                   args.pretrain_epochs, iters_per_epoch)
        optimizer = create_optimizer(args, model)
        lr_scheduler = WarmupCosineScheduler(
            optimizer,
            total_cycles=args.pretrain_epochs,
            warmup_cycles=args.warmup_epochs,
            start_lr=args.warmup_lr,
            peak_lr=args.lr,
            end_lr=args.min_lr,
        )

    start_epoch = 0
    init_loss = 999999
    best_val_loss = init_loss
    checkpoint_frequency = 10

    if args.mode == "test":
        result_rows = []
        for dataset_index, dataset in enumerate(dataset_list):
            diseases = datasets_config[dataset]['diseases']
            multiclass = datasets_config[dataset]['task_type'] == "multi-class classification"
            y_test, p_test = test_classification(
                student_model,
                dataset_index,
                data_loader_list_test[dataset_index],
                device,
                multiclass,
                len(diseases),
                accelerator=accelerator,
            )
            y_test_teacher, p_test_teacher = test_classification(
                teacher,
                dataset_index,
                data_loader_list_test[dataset_index],
                device,
                multiclass,
                len(diseases),
                accelerator=accelerator,
            )
            y_test = torch.cat(accelerator.gather_tensor(y_test), 0)
            p_test = torch.cat(accelerator.gather_tensor(p_test), 0)
            y_test_teacher = torch.cat(accelerator.gather_tensor(y_test_teacher), 0)
            p_test_teacher = torch.cat(accelerator.gather_tensor(p_test_teacher), 0)

            if dataset == "CheXpert":
                performance_diseases = datasets_config[dataset]['test_diseases_name']
                test_disease_indices = [diseases.index(name) for name in performance_diseases]
                y_test = y_test[:, test_disease_indices]
                p_test = p_test[:, test_disease_indices]
                y_test_teacher = y_test_teacher[:, test_disease_indices]
                p_test_teacher = p_test_teacher[:, test_disease_indices]
            else:
                performance_diseases = diseases

            for model_name, targets, outputs in (
                    ("student", y_test, p_test),
                    ("teacher", y_test_teacher, p_test_teacher)):
                if multiclass:
                    accuracy = accuracy_score(
                        np.argmax(targets.cpu().numpy(), axis=1),
                        np.argmax(outputs.cpu().numpy(), axis=1),
                    )
                    result_rows.append([dataset, model_name, "accuracy", "", accuracy])
                individual_results = metric_AUROC(
                    targets,
                    outputs,
                    len(performance_diseases),
                )
                for disease, auc in zip(performance_diseases, individual_results):
                    result_rows.append([dataset, model_name, "AUC", disease, auc])
                mean_auc = _mean_defined_metrics(individual_results)
                result_rows.append([dataset, model_name, "mAUC", "", mean_auc])

        write_status = torch.zeros(1, dtype=torch.int32, device=device)
        if accelerator.is_main_process:
            try:
                with open(test_output_path, 'x', newline='') as result_file:
                    writer = csv.writer(result_file)
                    writer.writerow(["dataset", "model", "metric", "class", "value"])
                    writer.writerows(result_rows)
            except Exception as error:
                print("Test results could not be written: {}".format(error))
            else:
                write_status[0] = 1
        accelerator.broadcast(write_status)
        if write_status.item() == 0:
            raise RuntimeError(
                "Test results could not be written on rank 0; see rank 0 output for details"
            )
        accelerator.barrier()
    elif args.mode == "train":
        if resume_requested:
            resume = resume_checkpoint_path
            if not os.path.isfile(resume):
                raise FileNotFoundError("Cannot load resume checkpoint: {}".format(resume))
            if accelerator.is_main_process:
                _print_and_log("=> loading checkpoint '{}'".format(resume), train_log)
            checkpoint = torch.load(resume, map_location=device, weights_only=False)
            _validate_checkpoint_total_cycles(checkpoint, resume, args)
            checkpoint_cycle = int(checkpoint.get('cycle', checkpoint['epoch'] + 1))
            if checkpoint_cycle >= args.pretrain_epochs:
                raise RuntimeError("Cannot continue training after cycle {}".format(checkpoint_cycle))
            start_epoch = checkpoint_cycle
            init_loss = checkpoint['lossMIN']
            state_dict = accelerator.strip_module_prefix(checkpoint['state_dict'])
            teacher_state_dict = accelerator.strip_module_prefix(checkpoint['teacher'])
            head_topology_matches = True
            if args.reinit_heads:
                current_head_shapes = {
                    k: tuple(v.shape) for k, v in student_model.state_dict().items()
                    if k.startswith('omni_heads.')
                }
                reinitialized_head_keys = {
                    k for k in current_head_shapes
                }
                checkpoint_head_shapes = {
                    k: tuple(v.shape) for k, v in state_dict.items()
                    if k.startswith('omni_heads.')
                }
                checkpoint_head_keys = {
                    k for k in checkpoint_head_shapes
                }
                checkpoint_head_shapes.update({
                    k: tuple(v.shape) for k, v in teacher_state_dict.items()
                    if k.startswith('omni_heads.')
                })
                checkpoint_head_keys.update(
                    k for k in teacher_state_dict.keys()
                    if k.startswith('omni_heads.')
                )
                head_topology_matches = checkpoint_head_shapes == current_head_shapes
                for k in sorted(checkpoint_head_keys):
                    if accelerator.is_main_process:
                        _print_and_log("Removing key {} from pretrained checkpoint".format(k), train_log)
                    state_dict.pop(k, None)
                    teacher_state_dict.pop(k, None)

                student_load_result = student_model.load_state_dict(state_dict, strict=False)
                teacher_load_result = teacher.load_state_dict(teacher_state_dict, strict=False)
                for model_name, load_result in (
                        ('student', student_load_result),
                        ('teacher', teacher_load_result)):
                    missing_keys = set(load_result.missing_keys)
                    unexpected_keys = set(load_result.unexpected_keys)
                    if missing_keys != reinitialized_head_keys or unexpected_keys:
                        raise RuntimeError(
                            "{} checkpoint load had missing keys {} and unexpected keys {}"
                            .format(model_name, sorted(missing_keys), sorted(unexpected_keys)))
            else:
                student_model.load_state_dict(state_dict, strict=True)
                teacher.load_state_dict(teacher_state_dict, strict=True)

            dataset_configuration_matches = (
                checkpoint.get('dataset_list') == list(dataset_list)
                and checkpoint.get('num_classes_list') == num_classes_list
            )
            restore_optimizer = (
                not args.reinit_heads
                or (head_topology_matches and dataset_configuration_matches)
            )
            if restore_optimizer:
                saved_scheduler_config = checkpoint.get('scheduler_config')
                if saved_scheduler_config != _scheduler_config(args):
                    raise ValueError(
                        "Scheduler configuration does not match checkpoint '{}'"
                        .format(resume)
                    )
                if not dataset_configuration_matches:
                    raise ValueError(
                        "Dataset configuration does not match checkpoint '{}'"
                        .format(resume)
                    )
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if args.reinit_heads:
                    for name, parameter in student_model.named_parameters():
                        if name.startswith('omni_heads.'):
                            optimizer.state.pop(parameter, None)
            elif accelerator.is_main_process:
                _print_and_log(
                    "Skipping optimizer and scheduler state because task-head topology changed",
                    train_log,
                )
            if accelerator.is_main_process:
                _print_and_log(
                    "=> loaded checkpoint '{}' (cycle={:04d}, val_loss={})"
                    .format(resume, checkpoint_cycle, init_loss),
                    train_log,
                )
        
            # wandb.init(
            #     # set the wandb project where this run will be logged
            #     project=exp+'_'+args.exp_name,
            #     resume=True
            # )
        # else:
        #     # start a new wandb run to track this script
        #     wandb.init(
        #         # set the wandb project where this run will be logged
        #         project=exp+'_'+args.exp_name,
                
        #         # track hyperparameters and run metadata
        #         config={
        #         "learning_rate": args.lr,
        #         "architecture": args.model_name,
        #         "dataset": exp,
        #         "epochs": args.pretrain_epochs,
        #         }
        #     )

        it = start_epoch * len(dataset_list)
        global_step = start_epoch * sum(len(data_loader) for data_loader in data_loader_list_train)
        
        for epoch in range(start_epoch, args.pretrain_epochs):
            lr_scheduler.step(epoch)
            if accelerator.is_main_process:
                learning_rates = [
                    "{:.8e}".format(param_group["lr"])
                    for param_group in optimizer.param_groups
                ]
                _print_and_log(
                    "Cycle {:04d}: learning rate = {}".format(
                        epoch + 1, ", ".join(learning_rates)),
                    train_log,
                )
            val_loss_list = None
            for i, data_loader in enumerate(data_loader_list_train): 
                if train_sampler_list[i] is not None:
                    train_sampler_list[i].set_epoch(epoch)
                criterion = torch.nn.CrossEntropyLoss() if datasets_config[dataset_list[i]]['task_type'] == "multi-class classification" else torch.nn.BCEWithLogitsLoss()
                momentum = momentum_schedule[it]
                coff = (momentum - 0.9) * 5
                if accelerator.is_main_process:
                    _print_and_log(
                        "Dataset {} ({}): momentum = {:.6f}, "
                        "classification/consistency = {:.4f}/{:.4f} ({:.1f}%/{:.1f}%)".format(
                            i + 1,
                            dataset_list[i],
                            momentum,
                            1 - coff,
                            coff,
                            100 * (1 - coff),
                            100 * coff,
                        ),
                        train_log,
                    )
                snapshot_directory = os.path.join(
                    model_path,
                    "snapshots",
                    "cycle_{:04d}".format(epoch + 1),
                    dataset_list[i],
                )
                if accelerator.is_main_process:
                    os.makedirs(snapshot_directory, exist_ok=True)
                task_metrics = train_one_epoch(
                    model,
                    i,
                    dataset_list[i],
                    data_loader,
                    device,
                    criterion,
                    optimizer,
                    epoch,
                    args.ema_mode,
                    teacher,
                    momentum_schedule,
                    it,
                    accelerator.is_main_process,
                    accelerator,
                    global_step,
                    momentum,
                    train_log,
                    loss_writer,
                    loss_file,
                    snapshot_directory,
                    args.print_freq,
                    training_start_time,
                )
                if accelerator.is_main_process:
                    _print_and_log(
                        "Finished {} Cycle={}: Loss={:.4f} "
                        "Cls={:.4f} [{:.1f}%] Cons={:.4f} [{:.1f}%] m={:.5f}"
                        .format(
                            dataset_list[i],
                            epoch + 1,
                            task_metrics["total_loss"],
                            task_metrics["classification_loss"],
                            task_metrics["classification_percent"],
                            task_metrics["consistency_loss"],
                            task_metrics["consistency_percent"],
                            momentum,
                        ),
                        train_log,
                    )
                it += 1
                global_step += len(data_loader)
                _evaluate_test_sets(
                    student_model,
                    teacher,
                    dataset_list,
                    datasets_config,
                    data_loader_list_test,
                    device,
                    accelerator,
                    evaluation_directory,
                    epoch + 1,
                    epoch,
                    "after_{}".format(dataset_list[i]),
                    shared_val_test_splits,
                    train_log,
                )
                val_loss_list = _evaluate_validation_sets(
                    student_model,
                    teacher,
                    dataset_list,
                    datasets_config,
                    data_loader_list_val,
                    val_sampler_list,
                    device,
                    accelerator,
                    evaluation_directory,
                    epoch + 1,
                    epoch,
                    "after_{}".format(dataset_list[i]),
                    shared_val_test_splits,
                    train_log,
                )

            accelerator.mark_step()
            cycle = epoch + 1
            if accelerator.is_main_process:
                weight_directory = os.path.join(
                    weights_directory,
                    "epoch_{:04d}".format(cycle),
                )
                os.makedirs(weight_directory, exist_ok=True)
                metadata = _run_metadata(
                    args,
                    dataset_list,
                    num_classes_list,
                    cycle,
                )
                with open(
                        os.path.join(weight_directory, "manifest.json"),
                        'w') as manifest_file:
                    json.dump(metadata, manifest_file, indent=2, sort_keys=True)
                student_state_dict = _copy_to_cpu(student_model.state_dict())
                teacher_state_dict = _copy_to_cpu(teacher.state_dict())
                optimizer_state_dict = _copy_to_cpu(optimizer.state_dict())
                torch.save(
                    student_state_dict,
                    os.path.join(
                        weight_directory,
                        "student_{}.pth".format(_checkpoint_stem(args, cycle)),
                    ),
                )
                torch.save(
                    teacher_state_dict,
                    os.path.join(
                        weight_directory,
                        "teacher_{}.pth".format(_checkpoint_stem(args, cycle)),
                    ),
                )
                checkpoint = metadata.copy()
                checkpoint.update({
                    'epoch': epoch,
                    'lossMIN': val_loss_list,
                    'state_dict': student_state_dict,
                    'teacher': teacher_state_dict,
                    'optimizer': optimizer_state_dict,
                    'scheduler': _copy_to_cpu(lr_scheduler.state_dict()),
                    'total_cycles': args.pretrain_epochs,
                    'scheduler_config': _scheduler_config(args),
                })
                save_checkpoint(
                    checkpoint,
                    filename=os.path.join(
                        checkpoint_directory,
                        _checkpoint_stem(args, cycle),
                    ),
                )
                _remove_stale_latest_checkpoints(
                    checkpoint_directory,
                    cycle,
                    checkpoint_frequency,
                )
            accelerator.barrier()

        accelerator.barrier()
    if loss_file is not None:
        loss_file.close()
    _close_logger(train_log)
    accelerator.destroy_distributed()

    
        
