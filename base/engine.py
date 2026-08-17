
import os
import sys
import shutil
import time
import csv
import glob
import re
import tempfile
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
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

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from functools import partial
import torch.nn as nn

# import wandb

sys.setrecursionlimit(40000)


def _print_and_log(message, log_file=None):
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def _append_evaluation_rows(file_path, rows):
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    with open(file_path, 'a', newline='') as file_descriptor:
        writer = csv.writer(file_descriptor)
        if not file_exists:
            writer.writerow(["cycle", "epoch", "metric", "student", "teacher"])
        writer.writerows(rows)


def _checkpoint_cycle(file_path):
    match = re.fullmatch(r"cycle_(\d+)\.pth\.tar", os.path.basename(file_path))
    return int(match.group(1)) if match else None


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
        'pretrain_epochs': args.pretrain_epochs,
        'sched': args.sched,
        'lr': args.lr,
        'lr_noise': args.lr_noise,
        'lr_noise_pct': args.lr_noise_pct,
        'lr_noise_std': args.lr_noise_std,
        'warmup_lr': args.warmup_lr,
        'min_lr': args.min_lr,
        'decay_epochs': args.decay_epochs,
        'warmup_epochs': args.warmup_epochs,
        'cooldown_epochs': args.cooldown_epochs,
        'decay_rate': args.decay_rate,
        'patience_epochs': args.patience_epochs,
        'ema_mode': args.ema_mode,
        'momentum_teacher': args.momentum_teacher,
        'batch_size': args.batch_size,
    }


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
        match = re.match(r"^Cycle\s+(\d+)(?::|\s+\|)", line)
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

    _discard_outputs_after_cycle(target_directory, max_cycle)


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


def omni_engine(args, model_path, output_path, dataset_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list):
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
    if args.mode != "train":
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

            resume_cycle_tensor[0] = selected_cycle

        _run_main_process_resume_setup(accelerator, device, setup_explicit_resume)
        accelerator.broadcast(resume_cycle_tensor)
        resume_cycle = int(resume_cycle_tensor.item())
        resume_checkpoint_path = os.path.join(
            model_path,
            "models",
            "checkpoints",
            "cycle_{:04d}.pth.tar".format(resume_cycle),
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
        resume_checkpoint_path = os.path.join(
            model_path,
            "models",
            "checkpoints",
            "cycle_{:04d}.pth.tar".format(resume_cycle),
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
    if args.mode == "train" and accelerator.is_main_process:
        train_log = open(log_file, 'a', buffering=1)
        train_log.write(str(args) + "\n")
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

    num_classes_list = [len(datasets_config[dataset]['diseases']) for dataset in dataset_list]
    if accelerator.is_main_process:
        _print_and_log("num_classes_list: {}".format(num_classes_list), train_log)
        if args.mode == "train":
            print_label_summary(
                dataset_list,
                datasets_config,
                dataset_train_list,
                dataset_val_list,
                dataset_test_list,
                train_log,
            )


    # training setups
    model = build_omni_model(args, num_classes_list)
    teacher = build_omni_model(args, num_classes_list)     
    model.to(device)
    teacher.to(device)
    model = accelerator.wrap_model(model)
    student_model = accelerator.unwrap_model(model)
    accelerator.synchronize_model(teacher)
    for p in teacher.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        _print_and_log(
            "Student and Teacher are built: they are both {} network.".format(args.model_name),
            train_log,
        )

    # momentum parameter is increased to 1. during training with a cosine schedule
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
    lr_scheduler, _ = create_scheduler(args, optimizer)

    start_epoch = 0
    init_loss = 999999
    best_val_loss = init_loss
    checkpoint_frequency = 10

    if args.mode == "train":
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

        test_results,test_results_teacher = [],[]
        it = start_epoch * len(dataset_list)
        global_step = start_epoch * sum(len(data_loader) for data_loader in data_loader_list_train)
        
        for epoch in range(start_epoch, args.pretrain_epochs):
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
                train_one_epoch(
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
                )
                it += 1
                global_step += len(data_loader)

            accelerator.barrier()
            val_loss_list = []
            for i, dv in enumerate(data_loader_list_val):
                dataset = dataset_list[i]
                diseases = datasets_config[dataset]['diseases']
                multiclass = datasets_config[dataset]['task_type'] == "multi-class classification"
                criterion = torch.nn.CrossEntropyLoss() if multiclass else torch.nn.BCEWithLogitsLoss()
                val_loss, y_val, p_val = evaluate(
                    student_model,
                    i,
                    dv,
                    device,
                    criterion,
                    dataset,
                    return_outputs=True,
                    multiclass=multiclass,
                    num_classes=len(diseases),
                )
                if accelerator.distributed:
                    sample_count = len(val_sampler_list[i])
                    val_loss_stats = torch.tensor(
                        [val_loss * sample_count, sample_count],
                        dtype=torch.float32,
                        device=device,
                    )
                    accelerator.all_reduce(val_loss_stats)
                    val_loss = (val_loss_stats[0] / val_loss_stats[1]).item()
                y_val = torch.cat(accelerator.gather_tensor(y_val), 0)
                p_val = torch.cat(accelerator.gather_tensor(p_val), 0)
                if accelerator.is_main_process:
                    val_auc_values = metric_AUROC(y_val, p_val, len(diseases))
                    val_rows = [[epoch + 1, epoch, "validation_loss", val_loss, ""]]
                    for disease, auc in zip(diseases, val_auc_values):
                        val_rows.append([epoch + 1, epoch, "AUC_{}".format(disease), auc, ""])
                    if val_auc_values:
                        val_rows.append([epoch + 1, epoch, "mAUC", np.mean(val_auc_values), ""])
                    _append_evaluation_rows(
                        os.path.join(evaluation_directory, dataset, "val_performance.csv"),
                        val_rows,
                    )
                val_loss_list.append(val_loss)
            
            avg_val_loss = np.average(val_loss_list)
            if args.val_loss_metric == "average":
                val_loss_metric = avg_val_loss
            else:
                val_loss_metric = val_loss_list[dataset_list.index(args.val_loss_metric)]
            lr_scheduler.step(val_loss_metric)
            
            # wandb.log({"avg_val_loss": avg_val_loss})
            
            if accelerator.is_main_process:
                _print_and_log(
                    "Cycle {:04d}: avg_val_loss {:.5f}".format(
                        epoch + 1,
                        avg_val_loss,
                    ),
                    train_log,
                )

                if train_log is not None:
                    train_log.write("     Datasets  : " + str(dataset_list) + "\n")
                    train_log.write("     Val Losses: " + str(val_loss_list) + "\n")
                    train_log.flush()
  
            if epoch % args.test_epoch == 0 or epoch+1 == args.pretrain_epochs:
                t_res, t_res_teacher = [],[]
                for i, dataset in enumerate(dataset_list):
                    diseases = datasets_config[dataset]['diseases']

                    multiclass =  datasets_config[dataset]['task_type'] == "multi-class classification"
                    y_test, p_test = test_classification(student_model, i, data_loader_list_test[i], device, multiclass, len(diseases))
                    y_test_teacher, p_test_teacher = test_classification(teacher, i, data_loader_list_test[i], device, multiclass, len(diseases))
                    y_test = torch.cat(accelerator.gather_tensor(y_test), 0)
                    p_test = torch.cat(accelerator.gather_tensor(p_test), 0)
                    y_test_teacher = torch.cat(accelerator.gather_tensor(y_test_teacher), 0)
                    p_test_teacher = torch.cat(accelerator.gather_tensor(p_test_teacher), 0)
                    if not accelerator.is_main_process:
                        continue
                    if multiclass:
                        acc = accuracy_score(np.argmax(y_test.cpu().numpy(),axis=1),np.argmax(p_test.cpu().numpy(),axis=1))
                        acc_teacher = accuracy_score(np.argmax(y_test_teacher.cpu().numpy(),axis=1),np.argmax(p_test_teacher.cpu().numpy(),axis=1))
                        _print_and_log(
                            ">>{}:Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(
                                dataset,
                                acc,
                                acc_teacher,
                            ),
                            train_log,
                        )
                        _append_evaluation_rows(
                            os.path.join(evaluation_directory, dataset, "test_performance.csv"),
                            [[epoch + 1, epoch, "accuracy", acc, acc_teacher]],
                        )
                        t_res.append(acc)
                        t_res_teacher.append(acc_teacher)

                    if dataset == "CheXpert":
                        test_diseases_name = datasets_config['CheXpert']['test_diseases_name']
                        test_diseases = [diseases.index(c) for c in test_diseases_name]
                        performance_diseases = test_diseases_name
                        y_test = copy.deepcopy(y_test[:,test_diseases])
                        p_test = copy.deepcopy(p_test[:, test_diseases])
                        individual_results = metric_AUROC(y_test, p_test, len(test_diseases))
                        y_test_teacher = copy.deepcopy(y_test_teacher[:,test_diseases])
                        p_test_teacher = copy.deepcopy(p_test_teacher[:, test_diseases])
                        individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(test_diseases))
                    else:
                        performance_diseases = diseases
                        individual_results = metric_AUROC(y_test, p_test, len(diseases))
                        individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(diseases))
                    _print_and_log(
                        ">>{}:Student AUC = {}, \nTeacher AUC = {}\n".format(
                            dataset,
                            np.array2string(np.array(individual_results), precision=4, separator='\t'),
                            np.array2string(np.array(individual_results_teacher), precision=4, separator='\t'),
                        ),
                        train_log,
                    )
                    test_rows = [
                        [
                            epoch + 1,
                            epoch,
                            "AUC_{}".format(disease),
                            student_auc,
                            teacher_auc,
                        ]
                        for disease, student_auc, teacher_auc in zip(
                            performance_diseases,
                            individual_results,
                            individual_results_teacher,
                        )
                    ]
                    mean_over_all_classes = np.mean(individual_results) if individual_results else np.nan
                    mean_over_all_classes_teacher = np.mean(individual_results_teacher) if individual_results_teacher else np.nan
                    _print_and_log(
                        ">>{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}".format(
                            dataset,
                            mean_over_all_classes,
                            mean_over_all_classes_teacher,
                        ),
                        train_log,
                    )
                    test_rows.append([
                        epoch + 1,
                        epoch,
                        "mAUC",
                        mean_over_all_classes,
                        mean_over_all_classes_teacher,
                    ])
                    _append_evaluation_rows(
                        os.path.join(evaluation_directory, dataset, "test_performance.csv"),
                        test_rows,
                    )
                    t_res.append(mean_over_all_classes)
                    t_res_teacher.append(mean_over_all_classes_teacher)

                if accelerator.is_main_process:
                    test_results.append(t_res)
                    test_results_teacher.append(t_res_teacher)
        
                    _print_and_log(
                        "Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(
                            test_results,
                            test_results_teacher,
                        ),
                        train_log,
                    )

            cycle = epoch + 1
            if accelerator.is_main_process:
                weight_directory = os.path.join(
                    weights_directory,
                    "epoch_{:04d}".format(cycle),
                )
                os.makedirs(weight_directory, exist_ok=True)
                torch.save(
                    student_model.state_dict(),
                    os.path.join(weight_directory, "student.pth"),
                )
                torch.save(
                    teacher.state_dict(),
                    os.path.join(weight_directory, "teacher.pth"),
                )
                if cycle % checkpoint_frequency == 0 or cycle == args.pretrain_epochs:
                    save_checkpoint(
                        {
                            'epoch': epoch,
                            'cycle': cycle,
                            'total_cycles': args.pretrain_epochs,
                            'scheduler_config': _scheduler_config(args),
                            'dataset_list': list(dataset_list),
                            'num_classes_list': num_classes_list,
                            'lossMIN': val_loss_list,
                            'state_dict': student_model.state_dict(),
                            'teacher': teacher.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': lr_scheduler.state_dict(),
                        },
                        filename=os.path.join(
                            checkpoint_directory,
                            "cycle_{:04d}".format(cycle),
                        ),
                    )
            accelerator.barrier()

        accelerator.barrier()
    if loss_file is not None:
        loss_file.close()
    if train_log is not None:
        train_log.close()
    accelerator.destroy_distributed()

    
        
