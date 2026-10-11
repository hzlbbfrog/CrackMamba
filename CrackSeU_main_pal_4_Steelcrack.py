"""Distributed training entry point for CrackMamba segmentation."""


import os
import torch.distributed as dist

import numpy as np
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
import torch.nn.functional as F
from datetime import datetime
import time
import copy
import cv2
from ptflops import get_model_complexity_info

from Dataset import DeepcrackDatasetrandom, Get_Steel_Cracks_with_Edge
from utils.loss import dice_loss_function
from utils.metrics import get_IoU,get_Dice


def getArgs():
    """Parse the command-line arguments for distributed training."""
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', type=int, default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train")
    parse.add_argument('--model', '-a', metavar='Model name', default='CrackSeU_B_LN_VT_VSSC2_noFFN',
                       help='CrackSeU_B_LN_VT_VIM/CrackSeU_B_LN_VT_SS2D/CrackSeU_L_LN_VT/CrackSeU_B_LN_VT/UNet/')
    parse.add_argument('--save_path', default='./Checkpoints/')
    parse.add_argument("--batch_size", type=int, default=9)
    parse.add_argument('--dataset', default='Steelcrack', help='Concrete_crack/Deepcrack/Steelcrack')
    parse.add_argument("--learning_rates",type=float,default=6e-3)
    parse.add_argument("--log_dir", default='Result_log/log', help="log dir")
    parse.add_argument("--epoch", type=int, default=70)
    parse.add_argument('--times', type=str, default='1')
    args = parse.parse_args()
    return args


def getLog(args):
    """Configure file logging for the selected experiment."""
    dirname = os.path.join(args.log_dir,args.model,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level = logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

# Model construction.
def getModel(args):
    """Construct the training model on the configured device."""
    if 'CrackSeU_B_LN_VT_VIM' == args.model:
        from Nets.CrackSeU_B_LN_VT_VIM import CrackSeU
    elif 'CrackSeU_B_LN_VT_VanillaVSS' == args.model:
        from Nets.CrackSeU_B_LN_VT_VanillaVSS import CrackSeU
    elif 'CrackSeU_B_LN_VT_VSSBlock' == args.model:
        from Nets.CrackSeU_B_LN_VT_VSSBlock import CrackSeU

    elif 'CrackSeU_B_LN_VT_CrackMamba' == args.model:
        from Nets.CrackSeU_B_LN_VT_CrackMamba import CrackSeU

    model = CrackSeU(3, 1).to(device)
    return model

def getModel1(args):
    """Construct the inference model used for complexity profiling."""
    if 'CrackSeU_B_LN_VT_VIM' == args.model:
        from Nets.CrackSeU_B_LN_VT_VIM import CrackSeU_Inference
    elif 'CrackSeU_B_LN_VT_VanillaVSS' == args.model:
        from Nets.CrackSeU_B_LN_VT_VanillaVSS import CrackSeU_Inference
    elif 'CrackSeU_B_LN_VT_VSSBlock' == args.model:
        from Nets.CrackSeU_B_LN_VT_VSSBlock import CrackSeU_Inference

    elif 'CrackSeU_B_LN_VT_CrackMamba' == args.model:
        from Nets.CrackSeU_B_LN_VT_CrackMamba import CrackSeU_Inference

    model = CrackSeU_Inference(3, 1).to(device)
    return model


def getDataset(args):

    """Build the dataset loaders and distributed training sampler."""
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    Current_dir = os.getcwd().replace('\\','/')

    # Select the dataset class.
    if args.dataset =='Concrete_crack':
        Dataset = ConcreteCrackDataset
    elif args.dataset =='Deepcrack':
        Dataset = DeepcrackDatasetrandom
    elif args.dataset == 'Steelcrack':
        Dataset = Get_Steel_Cracks_with_Edge

    train_dataset = Dataset(Current_dir,'Train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=num_gpus,rank=args.local_rank)  # Partition training samples across worker processes.
    train_dataloaders = DataLoader(dataset=train_dataset, batch_size=args.batch_size//num_gpus, shuffle=False, sampler=train_sampler)  # Interpret batch_size as the total batch size across all GPUs.

    val_dataset = Dataset(Current_dir,'Validation')
    val_dataloaders = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)  # Each process currently evaluates the complete validation split.

    test_dataset = Dataset(Current_dir,'Test')
    test_dataloaders = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    return train_dataloaders, train_sampler, val_dataloaders, test_dataloaders


def Validation(args, model, best_epoch, epoch, best_dice, val_dataloader):

    """Evaluate soft Dice and save a checkpoint when it improves."""
    with torch.no_grad():
        model= model.eval()

        i=0
        IoU_total = 0
        Dice_total = 0
        num = len(val_dataloader)  # Number of validation batches in this process.
        print(num)

        if best_epoch == 0:
            model1 = getModel1(args)
            total = sum([param.nelement() for param in model1.parameters()])
            print("Number of parameter: %.2fM" % (total/1e6))
            # Profile MACs and parameters with ptflops: https://github.com/sovrasov/flops-counter.pytorch
            macs, params = get_model_complexity_info(model1, (3, 512, 512), as_strings = False, print_per_layer_stat = False, verbose = False)
            print("# ------------- ptflops -------------- #")
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            print(f"macs = {macs/1e9}G")
            print(f"params = {params/1e6}M")
            logging.info("# ------------- ptflops -------------- #")
            logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
            logging.info(f"macs = {macs/1e9}G")
            logging.info(f"params = {params/1e6}M")

        for image, gt, name in val_dataloader:

            image = image.to(device)
            predict,_ = model(image)
            predict = predict.squeeze()

            # Accumulate soft Dice across validation images.
            gt = gt.squeeze().to(device)
            Dice = get_Dice(gt, predict)
            Dice_total += Dice

            if i < num:i+=1

        aver_iou = IoU_total / num
        aver_dice = Dice_total / num

        print('aver_iou = %f, aver_dice = %f' % (aver_iou, aver_dice))
        logging.info('aver_iou = %f, aver_dice = %f' % (aver_iou, aver_dice))
        if aver_dice > best_dice:

            print(f"aver_iou={aver_dice:f} > best_dice={best_dice:f}\n")
            logging.info(f"aver_iou={aver_dice:f} > best_dice={best_dice:f}")

            best_dice = aver_dice
            best_epoch = epoch

            print('===========>save best model!')
            logging.info('===========>save best model!')
            torch.save(model.state_dict(), save_path + str(args.model) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(epoch) + '.pth')

        print(f"best_epoch = {best_epoch}, Best dice = {best_dice:f}\n")
        logging.info(f"best_epoch = {best_epoch}, Best dice = {best_dice:f}\n")

        return best_epoch, best_dice, aver_dice


def Train(args, model, criterion, criterion2, optimizer, train_dataloader,train_sampler, val_dataloader):

    # Initialize training statistics.
    """Train the main and auxiliary segmentation outputs."""
    num_epochs = args.epoch
    best_epoch, best_dice, aver_dice = 0., 0., 0.
    loss_list = []
    dice_list = []
    loss_end, dice_loss_end, loss_2_BCE, loss_2_Dice= 0., 0., 0., 0.

    # Training loop.
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Use a different distributed sampling order each epoch.
        model = model.train()
        Current_epoch = epoch+1
        print('Epoch {}/{}\n'.format(Current_epoch, num_epochs))
        logging.info('Epoch {}/{}\n'.format(Current_epoch, num_epochs))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)  # Total number of training images.
        print('train set size = %d\n' % dt_size)
        total_step = len(train_dataloader)  # Number of training batches in this process.
        print(f'total_step = {total_step}')
        epoch_loss = 0
        step = 0

        for inputs, labels, gt_edge, name in train_dataloader:
            step += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_2 = F.interpolate(labels, scale_factor = 0.5, mode='bicubic', align_corners=True).to(device)

            optimizer.zero_grad()
            out, out_2 = model(inputs)
            loss_end = criterion(out, labels)
            dice_loss_end = dice_loss_function(out, labels)
            loss_2_BCE = criterion2(out_2, labels_2)

            loss = loss_end + dice_loss_end + 0.1 * loss_2_BCE

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if step % 100 == 0 or step == total_step:
                print("%s STEP [%d/%d], train_loss: %f, loss_end: %f, dice_loss_end: %f, loss_2_BCE: %f, loss_2_Dice: %f," % (datetime.now(), step, (dt_size) // args.batch_size, loss.item(), loss_end, dice_loss_end, loss_2_BCE, loss_2_Dice))
                logging.info("%s STEP [%d/%d], train_loss: %f, loss_end: %f, dice_loss_end: %f, loss_2_BCE: %f, loss_2_Dice: %f," % (datetime.now(), step, (dt_size) // args.batch_size, loss.item(), loss_end, dice_loss_end, loss_2_BCE, loss_2_Dice))

        print("epoch %d loss:%0.3f\n" % (Current_epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f\n" % (Current_epoch, epoch_loss))
        loss_list.append(epoch_loss)

        # Evaluate during the final 30 percent of training epochs.
        if Current_epoch >= num_epochs * 0.7:
            best_epoch, best_dice,aver_dice = Validation(args, model, best_epoch, Current_epoch, best_dice, val_dataloader)
            dice_list.append(aver_dice)

    print('------------- Congratulations! -------------')

    return model


if __name__ =="__main__":


    args = getArgs()
    logging = getLog(args)

    print('*' * 30)
    print(f'Dataset:{args.dataset},\nmodels:{args.model},\nEpoch={args.epoch},\nBatch size={args.batch_size},\nLearning rate={args.learning_rates}\n')
    logging.info(f'Dataset:{args.dataset},\nmodels:{args.model},\nEpoch={args.epoch},\nBatch size={args.batch_size},\nLearning rate={args.learning_rates}\n')
    print('*' * 30)

    save_path = args.save_path + args.dataset + args.model + '/' + args.times +'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)

    local_rank = args.local_rank
    num_gpus = torch.cuda.device_count()

    # Select the CUDA device for this worker.
    device = torch.device('cuda:%d' % local_rank)
    print(device)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',world_size=num_gpus, rank=local_rank)

    # Construct the model on the selected device.
    model = getModel(args)
    # Synchronize batch normalization across workers.
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    # Wrap the model for distributed training.
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank, ],output_device=local_rank)


    train_dataloader,train_sampler, val_dataloader,test_dataloader = getDataset(args)

    # Configure the loss functions and optimizer.
    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rates)

    Train(args, model, criterion, criterion2, optimizer, train_dataloader, train_sampler, val_dataloader)
