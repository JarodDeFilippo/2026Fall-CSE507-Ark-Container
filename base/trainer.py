from utils import MetricLogger, ProgressLogger, save_image, save_snapshot
import time
import torch
from tqdm import tqdm

def train_one_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, ema_mode, teacher, momentum_schedule, it, is_main_process=True, accelerator=None):
    batch_time = MetricLogger('Time', ':6.3f')
    losses_cls = MetricLogger('Loss_'+dataset+' cls', ':.4e')
    losses_mse = MetricLogger('Loss_'+dataset+' mse', ':.4e')
    progress = ProgressLogger(
        len(data_loader_train),
        [batch_time, losses_cls, losses_mse],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    MSE = torch.nn.MSELoss()
    coff = (momentum_schedule[it] - 0.9) * 5
    #print(momentum_schedule[it],it, coff)
    end = time.time()
    for i, (samples1, samples2, targets) in enumerate(data_loader_train):
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

        losses_cls.update(loss_cls.item(), samples1.size(0))
        losses_mse.update(loss_const.item(), samples1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            if accelerator is not None and accelerator.distributed:
                loss_stats = torch.tensor(
                    [
                        losses_cls.sum,
                        losses_cls.count,
                        loss_cls.item() * samples1.size(0),
                        samples1.size(0),
                        losses_mse.sum,
                        losses_mse.count,
                        loss_const.item() * samples1.size(0),
                        samples1.size(0),
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                accelerator.all_reduce(loss_stats)
                losses_cls.val = (loss_stats[2] / loss_stats[3]).item()
                losses_cls.avg = (loss_stats[0] / loss_stats[1]).item()
                losses_mse.val = (loss_stats[6] / loss_stats[7]).item()
                losses_mse.avg = (loss_stats[4] / loss_stats[5]).item()

            if is_main_process:
                progress.display(i)
                save_image(samples1[0].cpu().numpy().transpose(1, 2, 0), "Models/student"+str(i))
                save_image(samples2[0].cpu().numpy().transpose(1, 2, 0),"Models/teacher"+str(i))

        if ema_mode == "iteration":
            ema_update_teacher(model, teacher, momentum_schedule, it)
            it += 1

    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1
    

def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


def evaluate(model, use_head_n, data_loader_val, device, criterion, dataset):
    model.eval()

    with torch.no_grad():
        batch_time = MetricLogger('Time', ':6.3f')
        losses = MetricLogger('Loss', ':.4e')
        progress = ProgressLogger(
        len(data_loader_val),
        [batch_time, losses], prefix='Val_'+dataset+': ')

        end = time.time()
        for i, (samples, _, targets) in enumerate(data_loader_val):
            samples, targets = samples.float().to(device), targets.float().to(device)

            _, outputs = model(samples, use_head_n)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), samples.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

    return losses.avg


def test_classification(model, use_head_n, data_loader_test, device, multiclass = False): 
       
    model.eval()

    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)

    with torch.no_grad():
        for i, (samples, _, targets) in enumerate(tqdm(data_loader_test)):
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
    
