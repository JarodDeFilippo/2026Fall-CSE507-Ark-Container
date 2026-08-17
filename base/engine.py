
import os
import sys
import shutil
import time
import csv
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
    if args.mode != "train":
        model_path = os.path.join(model_path, exp)
        model_path = os.path.join(model_path, args.exp_name)
    elif args.resume:
        run_status = torch.zeros(1, dtype=torch.int32, device=device)
        if accelerator.is_main_process:
            run_status[0] = int(os.path.isdir(model_path))
        accelerator.broadcast(run_status)
        if run_status.item() == 0:
            raise FileNotFoundError(
                "Cannot resume training because run directory does not exist: {}".format(model_path)
            )
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
    save_model_path = os.path.join(model_path, exp)
    checkpoint_frequency = 10

    if args.mode == "train":
        if args.resume:
            resume = save_model_path + '.pth.tar'
            if os.path.isfile(resume):
                if accelerator.is_main_process:
                    _print_and_log("=> loading checkpoint '{}'".format(resume), train_log)
                checkpoint = torch.load(resume, map_location=device, weights_only=False)
                start_epoch = checkpoint['epoch']
                init_loss = checkpoint['lossMIN']
                state_dict = accelerator.strip_module_prefix(checkpoint['state_dict'])
                teacher_state_dict = accelerator.strip_module_prefix(checkpoint['teacher'])
                
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
                if not args.reinit_heads or head_topology_matches:
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
                        "=> loaded checkpoint '{}' (epoch={:04d}, val_loss={})"
                        .format(resume, start_epoch, init_loss),
                        train_log,
                    )
                start_epoch += 1
            else:
                if accelerator.is_main_process:
                    _print_and_log("=> no checkpoint found at '{}'".format(args.resume), train_log)
        
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

    
        
