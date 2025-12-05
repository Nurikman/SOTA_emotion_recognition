
'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy


def val_epoch_multimodal(epoch, data_loader, model, criterion, opt, modality='both', dist=None):
    print('validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_preds = []
    all_targets = []

    end_time = time.time()
    for i, (inputs_audio, inputs_visual, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        # ----- reshape -----
        inputs_visual = inputs_visual.permute(0,2,1,3,4)
        inputs_visual = inputs_visual.reshape(
            inputs_visual.shape[0]*inputs_visual.shape[1],
            inputs_visual.shape[2],
            inputs_visual.shape[3],
            inputs_visual.shape[4]
        )
        targets = targets.to(opt.device)
        inputs_audio = inputs_audio.to(opt.device)
        inputs_visual = inputs_visual.to(opt.device)

        with torch.no_grad():
            outputs = model(inputs_audio, inputs_visual)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            loss = criterion(outputs, targets)
            prec1, _ = calculate_accuracy(outputs.data, targets.data, topk=(1,5))

        top1.update(prec1, inputs_audio.size(0))
        losses.update(loss.data, inputs_audio.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print(
            'Epoch: [{0}][{1}/{2}] '
            'Loss {loss.val:.4f} ({loss.avg:.4f}) '
            'Prec@1 {top1.val:.5f} ({top1.avg:.5f})'.format(
                epoch, i+1, len(data_loader), loss=losses, top1=top1
            )
        )
    return losses.avg.item(), top1.avg.item(), all_targets, all_preds


def val_epoch(epoch, data_loader, model, criterion, opt, modality='both', dist=None):
    return val_epoch_multimodal(epoch, data_loader, model, criterion, opt, modality, dist)
