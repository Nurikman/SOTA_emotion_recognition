
 
'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy


def train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt):
    print('train at epoch {}'.format(epoch))
    
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
        
    end_time = time.time()
    for i, (audio_inputs, visual_inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets = targets.to(opt.device)
            
        # ----- multimodal mask handling -----
        if opt.mask is not None:
            with torch.no_grad():
                if opt.mask == 'noise':
                    audio_inputs = torch.cat(
                        (audio_inputs, torch.randn(audio_inputs.size()), audio_inputs), dim=0)
                    visual_inputs = torch.cat(
                        (visual_inputs, visual_inputs, torch.randn(visual_inputs.size())), dim=0)
                    targets = torch.cat((targets, targets, targets), dim=0)
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
                    
                elif opt.mask == 'softhard':
                    coefficients = torch.randint(
                        low=0, high=100, size=(audio_inputs.size(0),1,1)) / 100
                    vision_coefficients = 1 - coefficients

                    coefficients = coefficients.repeat(
                        1, audio_inputs.size(1), audio_inputs.size(2))
                    vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(
                        1, visual_inputs.size(1),
                        visual_inputs.size(2),
                        visual_inputs.size(3),
                        visual_inputs.size(4)
                    )

                    audio_inputs = torch.cat(
                        (audio_inputs, audio_inputs*coefficients,
                         torch.zeros(audio_inputs.size()), audio_inputs), dim=0)
                    visual_inputs = torch.cat(
                        (visual_inputs, visual_inputs*vision_coefficients,
                         visual_inputs, torch.zeros(visual_inputs.size())), dim=0)
                    
                    targets = torch.cat((targets, targets, targets, targets), dim=0)
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]

        # ----- reshape video -----
        visual_inputs = visual_inputs.permute(0,2,1,3,4)
        visual_inputs = visual_inputs.reshape(
            visual_inputs.shape[0]*visual_inputs.shape[1],
            visual_inputs.shape[2],
            visual_inputs.shape[3],
            visual_inputs.shape[4],
        )

        audio_inputs = audio_inputs.to(opt.device)
        visual_inputs = visual_inputs.to(opt.device)
        audio_inputs = Variable(audio_inputs)
        visual_inputs = Variable(visual_inputs)
        targets = Variable(targets)

        outputs = model(audio_inputs, visual_inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.data, audio_inputs.size(0))
        prec1, _ = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, audio_inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % 10 == 0:
            print(
                'Epoch: [{0}][{1}/{2}] '
                'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                'Prec@1 {top1.val:.5f} ({top1.avg:.5f})'.format(
                    epoch, i, len(data_loader), loss=losses, top1=top1
                )
            )

    return losses.avg.item(), top1.avg.item()


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt):
    return train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt)
