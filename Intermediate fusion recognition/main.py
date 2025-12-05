
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import pickle

import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from train import train_epoch
from validation import val_epoch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from inference import TrainingAlignedInference
if __name__ == '__main__':
    opt = parse_opts()
    n_folds = 5

    # metric storage

    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    device = opt.device
    if opt.media_path != "":
        infer = TrainingAlignedInference()

        infer.device = torch.device(device)
        infer.model.to(infer.device)

        print("\nRunning sliding-window inference")
        results = infer.inference_sliding_window(
            media_path=opt.media_path,
            step_sec=1.0,
            plot_results=not opt.no_plot
        )

        if opt.save_plot is not None:
            print(f"Saving plot to {opt.save_plot}")
            infer.plot_emotions(
                results["times"],
                results["probabilities"],
                save_path=opt.save_plot
            )
    if not opt.only_inference:
        pretrained = opt.pretrain_path != 'None'    
        
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
            
        opt.arch = '{}'.format(opt.model)  
        opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])
                
        for fold in range(n_folds):
            train_loss_history = {}
            train_acc_history = {}
            val_loss_history = {}
            val_acc_history = {}
            # if fold != 0:
            if opt.dataset == 'RAVDESS':
                opt.annotation_path = (
                    'D:/Yeskendir_files/multimodal-emotion-recognition/ravdess_preprocessing/'
                    'annotations_croppad_fold' + str(fold+1) + '.txt'
                )

            print(opt)
            with open(os.path.join(opt.result_path, 'opts'+str(time.time())+str(fold)+'.json'), 'w') as opt_file:
                json.dump(vars(opt), opt_file)
                
            torch.manual_seed(opt.manual_seed)
            model, parameters = generate_model(opt)

            criterion = nn.CrossEntropyLoss().to(opt.device)
            
            if not opt.no_train:

                video_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotate(),
                    transforms.ToTensor(opt.video_norm_value)])

                training_data = get_training_set(opt, spatial_transform=video_transform) 
            
                train_loader = torch.utils.data.DataLoader(
                    training_data,
                    batch_size=opt.batch_size,
                    shuffle=True,
                    num_workers=opt.n_threads,
                    pin_memory=True)

                optimizer = optim.SGD(
                    parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=opt.dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=False)

                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer, 'min', patience=opt.lr_patience)
                
            if not opt.no_val:

                video_transform = transforms.Compose([
                    transforms.ToTensor(opt.video_norm_value)])     

                validation_data = get_validation_set(opt, spatial_transform=video_transform)
                
                val_loader = torch.utils.data.DataLoader(
                    validation_data,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=opt.n_threads,
                    pin_memory=True)

            best_prec1 = 0
            best_loss = 1e10

            if opt.resume_path:
                print('loading checkpoint {}'.format(opt.resume_path))
                checkpoint = torch.load(opt.resume_path, weights_only=False)
                assert opt.arch == checkpoint['arch']
                best_prec1 = checkpoint['best_prec1']
                opt.begin_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])

            for i in range(opt.begin_epoch, opt.n_epochs + 1):

                # -------- TRAIN --------
                if not opt.no_train:
                    adjust_learning_rate(optimizer, i, opt)
                    train_loss, train_acc = train_epoch(i, train_loader, model, criterion, optimizer, opt)

                    train_loss_history[i] = float(train_loss)
                    train_acc_history[i] = float(train_acc)

                    scheduler.step(train_loss)

                    state = {
                        'epoch': i,
                        'arch': opt.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_prec1': best_prec1
                    }
                    train_loss_path = "train_loss.pkl" + str(fold+1)
                    with open(os.path.join(opt.result_path, train_loss_path), "wb") as f:
                        pickle.dump(train_loss_history, f)
                    train_acc_path = "train_acc.pkl" + str(fold+1)
                    with open(os.path.join(opt.result_path, train_acc_path), "wb") as f:
                        pickle.dump(train_acc_history, f)
                    save_checkpoint(state, False, opt, fold)
                
                # -------- VALIDATION --------
                if not opt.no_val:
                    val_loss, prec1, y_true, y_pred = val_epoch(i, val_loader, model, criterion, opt)

                    val_loss_history[i] = float(val_loss)
                    val_acc_history[i] = float(prec1)

                    is_best = prec1 > best_prec1
                    best_prec1 = max(prec1, best_prec1)

                    state = {
                        'epoch': i,
                        'arch': opt.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer,
                        'best_prec1': best_prec1
                    }
                
                    save_checkpoint(state, is_best, opt, fold)
                    val_loss_path = "val_loss.pkl" + str(fold+1)
                    with open(os.path.join(opt.result_path, val_loss_path), "wb") as f:
                        pickle.dump(val_loss_history, f)
                    val_acc_path = "val_acc.pkl" + str(fold+1)
                    with open(os.path.join(opt.result_path, val_acc_path), "wb") as f:
                        pickle.dump(val_acc_history, f)

            
            # -------- TESTING --------
            if opt.test:
                base_dir = "D:/Yeskendir_files/results2"

                test_accuracies = []

                video_transform = transforms.Compose([
                    transforms.ToTensor(opt.video_norm_value)
                ])

                test_data = get_test_set(opt, spatial_transform=video_transform)

                # Create folder for this fold
                fold_dir = os.path.join(base_dir, f"fold {str(fold+1)}")
                os.makedirs(fold_dir, exist_ok=True)

                # Load best model for this fold
                best_state = torch.load(
                    f"{fold_dir}/{opt.store_name}_best{fold}.pth",
                    weights_only=False
                )
                model.load_state_dict(best_state["state_dict"])

                test_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=opt.n_threads,
                    pin_memory=True
                )

                test_loss, test_prec1, y_true, y_pred = val_epoch(
                    10000, test_loader, model, criterion, opt
                )

                # Emotion labels
                emotion_labels = [
                    "neutral", "calm", "happy", "sad",
                    "angry", "fearful", "disgust", "surprise"
                ]

                # ----- SAVE REPORT CSV -----
                # report_dict = report(y_true, y_pred, emotion_labels)
                save_csv_report(y_true,y_pred, fold_dir, fold, emotion_labels)
                save_confusion_matrix(y_true,y_pred,fold_dir, fold, emotion_labels)
                with open(os.path.join(fold_dir, f"test_summary_fold_{fold}.txt"), "w") as f:
                    f.write(f"Precision@1: {test_prec1}\n")
                    f.write(f"Loss: {test_loss}\n")
                test_accuracies.append(test_prec1)