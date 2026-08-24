import datetime
import os
import time

import torch

from utils import MetricLogger, save_image


def train_one_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, ema_mode, teacher, momentum_schedule, it, is_main_process=True, accelerator=None, global_step=0, momentum=None, train_log=None, loss_writer=None, loss_file=None, snapshot_directory=None, print_freq=50, training_start_time=None):
    raw_losses_cls = MetricLogger('Raw loss_'+dataset+' cls', ':.4e')
    raw_losses_mse = MetricLogger('Raw loss_'+dataset+' mse', ':.4e')
    losses_cls = MetricLogger('Loss_'+dataset+' cls', ':.4e')
    losses_mse = MetricLogger('Loss_'+dataset+' mse', ':.4e')
    losses_total = MetricLogger('Loss_'+dataset+' total', ':.4e')
    batch_time = MetricLogger('Batch time', ':6.3f')
    data_time = MetricLogger('Data time', ':6.3f')
    if training_start_time is None:
        training_start_time = time.time()

    model.train()
    teacher.eval()
    MSE = torch.nn.MSELoss()
    coff = (momentum_schedule[it] - 0.9) * 5
    if momentum is None:
        momentum = momentum_schedule[it]
    cls_percent = 0
    const_percent = 0
    #print(momentum_schedule[it],it, coff)
    end = time.time()
    for i, (samples1, samples2, targets) in enumerate(data_loader_train):
        data_time.update(time.time() - end)
        samples1, samples2, targets = samples1.float().to(device), samples2.float().to(device), targets.float().to(device)
        
        with torch.no_grad():
            feat_t, pred_t = teacher(samples2, use_head_n)
        feat_s, pred_s = model(samples1, use_head_n)
        loss_cls = criterion(pred_s, targets)
        loss_const = MSE(feat_s, feat_t)

        # outputs_t = teacher(samples2)
        # outputs_s = model(samples1)
        # loss_cls = criterion(outputs_s[use_head_n], targets)
        # loss_const = 0
        # for i in range(len(outputs_t)):
        #     loss_const += MSE(outputs_t[i], outputs_s[i])
        
        loss = (1-coff) * loss_cls + coff * loss_const

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_cls_value = (1 - coff) * loss_cls.item()
        loss_const_value = coff * loss_const.item()
        batch_size = samples1.size(0)
        loss_stats = torch.tensor(
            [
                loss_cls.item() * batch_size,
                loss_const.item() * batch_size,
                loss_cls_value * batch_size,
                loss_const_value * batch_size,
                loss.item() * batch_size,
                batch_size,
            ],
            dtype=torch.float32,
            device=device,
        )
        if accelerator is not None and accelerator.distributed:
            accelerator.all_reduce(loss_stats)
        global_batch_size = int(loss_stats[5].item())
        global_raw_cls_value = (loss_stats[0] / loss_stats[5]).item()
        global_raw_const_value = (loss_stats[1] / loss_stats[5]).item()
        global_cls_value = (loss_stats[2] / loss_stats[5]).item()
        global_const_value = (loss_stats[3] / loss_stats[5]).item()
        global_total_value = (loss_stats[4] / loss_stats[5]).item()
        raw_losses_cls.update(global_raw_cls_value, global_batch_size)
        raw_losses_mse.update(global_raw_const_value, global_batch_size)
        losses_cls.update(global_cls_value, global_batch_size)
        losses_mse.update(global_const_value, global_batch_size)
        losses_total.update(global_total_value, global_batch_size)

        total_loss = losses_total.avg
        if total_loss != 0:
            cls_percent = 100 * losses_cls.avg / total_loss
            const_percent = 100 * losses_mse.avg / total_loss
        else:
            cls_percent = 0
            const_percent = 0
        current_total = global_total_value
        if current_total != 0:
            current_cls_percent = 100 * global_cls_value / current_total
            current_const_percent = 100 * global_const_value / current_total
        else:
            current_cls_percent = 0
            current_const_percent = 0

        if is_main_process and loss_writer is not None:
            loss_writer.writerow([
                epoch + 1,
                epoch,
                global_step + i,
                i + 1,
                dataset,
                global_batch_size,
                global_cls_value,
                global_const_value,
                global_total_value,
                current_cls_percent,
                current_const_percent,
                optimizer.param_groups[0]["lr"],
                momentum,
            ])
            if loss_file is not None:
                loss_file.flush()

        if is_main_process and snapshot_directory is not None and i == 0:
            save_image(
                samples1[0].cpu().numpy().transpose(1, 2, 0),
                os.path.join(snapshot_directory, "student"),
            )
            save_image(
                samples2[0].cpu().numpy().transpose(1, 2, 0),
                os.path.join(snapshot_directory, "teacher"),
            )

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0 or i + 1 == len(data_loader_train):
            if is_main_process:
                elapsed = str(
                    datetime.timedelta(
                        seconds=int(time.time() - training_start_time),
                    )
                )
                message = (
                    "Ark+ {} head={} Cycle={} g_step={} [B {}/{}] "
                    "BT={:.2f}({:.2f}) DT={:.2f}({:.2f}) "
                    "LR={:.2e} m={:.5f} w_cons={:.3f} "
                    "Loss={:.4f}({:.4f}) "
                    "Cls={:.4f}({:.4f}) [{:.1f}%] "
                    "Cons={:.4f}({:.4f}) [{:.1f}%] Elapsed={}"
                ).format(
                    dataset,
                    use_head_n,
                    epoch + 1,
                    global_step + i + 1,
                    i + 1,
                    len(data_loader_train),
                    batch_time.val,
                    batch_time.avg,
                    data_time.val,
                    data_time.avg,
                    optimizer.param_groups[0]["lr"],
                    momentum,
                    coff,
                    global_total_value,
                    total_loss,
                    global_raw_cls_value,
                    raw_losses_cls.avg,
                    cls_percent,
                    global_raw_const_value,
                    raw_losses_mse.avg,
                    const_percent,
                    elapsed,
                )
                if train_log is not None:
                    train_log.info(message)
                else:
                    print(message)

        if ema_mode == "iteration":
            ema_update_teacher(model, teacher, momentum_schedule, it)
            it += 1

    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1

    return {
        "total_loss": losses_total.avg,
        "classification_loss": raw_losses_cls.avg,
        "consistency_loss": raw_losses_mse.avg,
        "classification_percent": cls_percent,
        "consistency_percent": const_percent,
    }


def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


def evaluate(model, use_head_n, data_loader_val, device, criterion, dataset, return_outputs=False, multiclass=False, num_classes=None):
    model.eval()

    with torch.no_grad():
        losses = MetricLogger('Loss', ':.4e')
        targets_list = []
        outputs_list = []

        for i, (samples, _, targets) in enumerate(data_loader_val):
            samples, targets = samples.float().to(device), targets.float().to(device)

            _, outputs = model(samples, use_head_n)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), samples.size(0))
            if return_outputs:
                targets_list.append(targets)
                outputs_list.append(torch.softmax(outputs, dim=1) if multiclass else torch.sigmoid(outputs))

    if not return_outputs:
        return losses.avg

    if targets_list:
        return losses.avg, torch.cat(targets_list, 0), torch.cat(outputs_list, 0)

    empty_outputs = torch.empty((0, num_classes), device=device)
    return losses.avg, empty_outputs, empty_outputs


def test_classification(model, use_head_n, data_loader_test, device, multiclass = False, num_classes = None):
       
    model.eval()

    y_test = torch.empty((0, num_classes), device=device)
    p_test = torch.empty((0, num_classes), device=device)

    with torch.no_grad():
        for i, (samples, _, targets) in enumerate(data_loader_test):
            targets = targets.to(device)
            y_test = torch.cat((y_test, targets), 0)

            if len(samples.size()) == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif len(samples.size()) == 5:
                bs, n_crops, c, h, w = samples.size()

            varInput = torch.autograd.Variable(samples.view(-1, c, h, w).to(device))

            _, out = model(varInput, use_head_n)
            if multiclass:
                out = torch.softmax(out,dim = 1)
            else:
                out = torch.sigmoid(out)
            outMean = out.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test, outMean.data), 0)

    return y_test, p_test
    
