# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
# import wandb
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
from tabulate import tabulate


def train(mmt_model, train_loader, test_loader, args, tokenizer_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    best_epoch, best_f1, best_acc = 0, 0, 0
    global_step, epoch = 0, 0
    stop_patience = 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)
    
    mmt_model = mmt_model.to(device)

    if not isinstance(mmt_model, nn.DataParallel):
        mmt_model = nn.DataParallel(mmt_model)


    # Set up the optimizer
    trainables = [p for p in mmt_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in mmt_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    large_update_trainables = [p for n, p in mmt_model.named_parameters() if p.requires_grad and "bert" not in n]
    small_update_trainables = [p for n, p in mmt_model.named_parameters() if p.requires_grad and "bert" in n]

    optimizer = torch.optim.Adam([
        {'params': small_update_trainables, 'lr': args.lr / args.text_lr_factor},
        {'params': large_update_trainables, 'lr': args.lr, 'weight_decay': 5e-7, 'betas':(0.95, 0.999) }
    ])

    # optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=, betas=(0.95, 0.999))


    # dataset specific settings
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)

    if args.dataset == 'mosei':
        print('scheduler for mosei is used')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
        main_metrics = 'acc'
        loss_fn = nn.BCEWithLogitsLoss()
        warmup = False
    elif args.dataset == 'iemocap':
        print('scheduler for iemocap is used')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
        main_metrics = 'acc'
        loss_fn = nn.BCEWithLogitsLoss()
        warmup = False
    else:
        raise ValueError('unknown dataset, dataset should be in [audioset, speechcommands, esc50]')
    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    args.loss_fn = loss_fn

    epoch += 1
    # for amp
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 14])
    mmt_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        mmt_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        # loss_show = 0
        for i, (audio_input, video_input, text_input, labels, ids) in enumerate(train_loader):

            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            video_input = video_input.to(device, non_blocking=True)

            text_input = tokenizer_model(text_input, return_tensors='pt', max_length=args.text_max_len, padding='max_length', truncation=True)
            text_input = text_input.to(device)



            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            with autocast():
                # tav_output, v_overlap_one, a_overlap_two, v_overlap_two = mmt_model(audio_input, video_input, text_input)
                tav_output = mmt_model(audio_input, video_input, text_input)
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    loss = loss_fn(tav_output, torch.argmax(labels.long(), axis=1))
                else:
                    loss = loss_fn(tav_output, labels)
                # if i%100 == 0 and i!=0:
                #     # wandb.log({"temp":loss_show})
                #     wandb.log({"iemocap_loss":loss_show})
                #     loss_show = 0
                # else:
                #     loss_show+=loss
                # reg_loss_fn = nn.L1Loss()
                # eye_tensor = torch.eye(256).to(device)

                # reg_loss = (reg_loss_fn(a_overlap_two, eye_tensor) \
                #                     + reg_loss_fn(v_overlap_one, eye_tensor) + reg_loss_fn(v_overlap_two, eye_tensor))

                # loss += args.scale * reg_loss
                

            # optimization if amp is not used
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # optimiztion if amp is used
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        stats, valid_loss = validate(mmt_model, test_loader, args, epoch, tokenizer_model)
        accs, recalls, precisions, f1s, aucs, best_thresholds = stats

        if args.dataset == 'mosei':
            annotations = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']
        elif args.dataset == 'iemocap':
            annotations = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
        
        headers = [
                ['phase (acc)', *annotations, 'average'],
                ['phase (recall)', *annotations, 'average'],
                ['phase (precision)', *annotations, 'average'],
                ['phase (f1)', *annotations, 'average'],
                ['phase (auc)', *annotations, 'average']
            ]
        for i in range(len(headers)):
            content_str = ["{:.4f}".format(content) for content in stats[i]]
            print(tabulate([
                            ['Valid', *content_str],
            ], headers=headers[i]))

        print("train_loss: {:.4f}".format(loss_meter.avg))
        print("valid_loss: {:.4f}".format(valid_loss))

        tmp = [accs[-1], recalls[-1], precisions[-1], f1s[-1], aucs[-1], loss_meter.avg, valid_loss, optimizer.param_groups[0]['lr']]
        tmp.extend(best_thresholds)
        result[epoch-1, :] = tmp
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if main_metrics == 'f1':
            if f1s[-1] > best_f1:
                best_f1 = f1s[-1]
                best_epoch = epoch
            else:
                stop_patience += 1

        if main_metrics == 'acc':
            if accs[-1] > best_acc:
                best_acc = accs[-1]
                best_epoch = epoch
            else:
                stop_patience += 1

        if stop_patience >= args.early_stop:
            print("Early Stop !!!")
            break

        if best_epoch == epoch:
            stop_patience = 0
            torch.save(mmt_model.state_dict(), "%s/models/best_mmt_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        torch.save(mmt_model.state_dict(), "%s/models/mmt_model.%d.pth" % (exp_dir, epoch))
        if len(train_loader.dataset) > 2e5:
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

        scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()


def validate(mmt_model, val_loader, args, epoch, tokenizer_model, thresholds=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(mmt_model, nn.DataParallel):
        mmt_model = nn.DataParallel(mmt_model)
    mmt_model = mmt_model.to(device)
    # switch to evaluate mode
    mmt_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    ids = []
    with torch.no_grad():
        for i, (audio_input, video_input, text_input, labels, curr_id) in enumerate(val_loader):
            ids.extend(curr_id)
            audio_input = audio_input.to(device)
            video_input = video_input.to(device, non_blocking=True)
            text_input = tokenizer_model(text_input, return_tensors='pt', max_length=args.text_max_len, padding='max_length', truncation=True)
            text_input = text_input.to(device)
            # compute output
            tav_output = mmt_model(audio_input, video_input, text_input)
            # do not use Sigmoid here, use it in the calculate_stats Function
            # predictions = torch.sigmoid(tav_output)
            predictions = tav_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(tav_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(tav_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        tav_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        if args.dataset == 'mosei':
            stats = calculate_stats(tav_output, target, thresholds, True, ids=ids, valid_test=epoch, args=args)
        elif args.dataset == 'iemocap':
            stats = calculate_stats(tav_output, target, thresholds, False, ids=ids, valid_test=epoch, args=args)

        # save the prediction here
        exp_dir = args.exp_dir
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', tav_output, delimiter=',')

    return stats, loss

def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
    if epoch == 1:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        cum_predictions = cum_predictions + predictions
        # remove the prediction file to save storage space
        os.remove(exp_dir+'/predictions/predictions_' + str(epoch-1) + '.csv')

    cum_predictions = cum_predictions / epoch
    np.savetxt(exp_dir+'/predictions/cum_predictions.csv', cum_predictions, delimiter=',')

    stats = calculate_stats(cum_predictions, target)
    return stats

