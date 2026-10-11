"""Checkpoint evaluation and prediction export for CrackMamba."""

# Original script reference: https://github.com/Andy-zhujunwen/UNET-ZOO

import os
import torch.distributed as dist


import os
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
import copy
import cv2
from ptflops import get_model_complexity_info

from Nets.CrackSeU_B_LN_VT_CrackMamba import CrackSeU, CrackSeU_Inference

from Dataset import DeepcrackDatasetrandom, Get_Steel_Cracks_with_Edge
from utils.loss import dice_loss_function
from utils.metrics import get_IoU,get_Dice,Get_F1_score


def getArgs():
    """Parse the command-line arguments for checkpoint evaluation."""
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', type=int, default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="test")
    parse.add_argument('--arch', '-a', metavar='ARCH', default='CrackSeU_B_LN_VT_CrackMamba_2026_branch_two_1' ,
                       help='CrackSeU_B_LN_VT_VIM/CrackSeU_L_LN_VT/CrackSeU_B_LN_VT/CrackSeU_B_LN_VT_VSSBlock/')
    parse.add_argument('--save_path', default='./Checkpoints/')
    parse.add_argument("--batch_size", type=int, default=9)
    parse.add_argument('--dataset', default='Steelcrack', help='Concrete_crack/Deepcrack/Steelcrack')
    parse.add_argument("--learning_rates",type=float,default=6e-3)
    parse.add_argument("--log_dir", default='Result_log/log', help="log dir")
    parse.add_argument("--epoch", type=int, default=70)
    parse.add_argument('--times', type=str, default='1_0')
    parse.add_argument("--test_epoch", type=int, default=67)
    args = parse.parse_args()
    return args


def getLog(args):
    """Configure file logging for the selected experiment."""
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level = logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging


def getModel(args):
    """Construct the training architecture used to load checkpoints."""
    if 'CrackSeU' in args.arch :
        model = CrackSeU(3, 1)
        model = model.to(device)
    return model


def getModel1(args):
    """Construct the inference architecture for complexity profiling."""
    if 'CrackSeU' in args.arch :
        model = CrackSeU_Inference(3, 1)
        model = model.to(device)
    return model


def getDataset(args):

    """Build validation and test dataset loaders."""
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    Current_dir = os.getcwd().replace('\\','/')

    # Select the dataset class.
    if args.dataset =='Concrete_crack':
        Dataset = ConcreteCrackDataset
    elif args.dataset =='Deepcrack':
        Dataset = DeepcrackDatasetrandom
    elif args.dataset == 'Steelcrack':
        Dataset = Get_Steel_Cracks_with_Edge

    val_dataset = Dataset(Current_dir,'Validation')
    val_dataloaders = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)  # Evaluate the complete validation split in one process.

    test_dataset = Dataset(Current_dir,'Test')
    test_dataloaders = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    return val_dataloaders, test_dataloaders


def Test(test_dataloader, save_predict=False):

    # Record the evaluation start time with millisecond precision.
    """Load a checkpoint and evaluate mean soft Dice and binary IoU."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"Test started at {current_time}.")
    logging.info(f"Test started at {current_time}.")

    if save_predict ==True:
        dir = os.path.join(r'./Predictions',str(args.arch),str(args.batch_size),str(args.epoch),str(args.dataset))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('Dir already exist!')

    torch.cuda.synchronize()
    load_name = save_path + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(args.test_epoch) + '.pth'
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(load_name, map_location=device).items()}, strict=True)
    model.eval()

    print(load_name)

    with torch.no_grad():

        i=0
        IoU_total = 0
        confusionMatrixe = np.zeros((2, 2))
        Dice_total = 0
        F1_score_total = 0
        num = len(test_dataloader)
        print(f"Num of Test images = {num}")

        for image, gt, name in test_dataloader:

            image = image.to(device)
            predict,_ = model(image)
            predict = predict.squeeze()

            gt = gt.squeeze().to(device)
            gt_IoU = copy.deepcopy(gt)
            predict_IoU = copy.deepcopy(predict)

            # Export prediction probabilities as grayscale images.
            predict_output = predict.cpu().numpy()
            cv2.imwrite(os.path.join(dir, name[0]+'.png'), predict_output*255.0)

            # Accumulate soft Dice.
            Dice = get_Dice(gt, predict)
            Dice_total += Dice

            # Accumulate binary IoU at a threshold of 0.5.
            IoU = get_IoU(gt_IoU, predict_IoU)
            IoU_total += IoU

            if i < num:
                i += 1

        torch.cuda.synchronize()

        # Report mean metrics across test images.
        print('mi IoU=%f' % (IoU_total/ num))
        print('mi Dice=%f' % (Dice_total / num))

        # Count parameters in the inference model.
        model1 = getModel1(args)
        total = sum([param.nelement() for param in model1.parameters()])
        print("Number of parameter: %.2fM" % (total/1e6))

        # Profile MACs and parameters with ptflops.
        # Profiler reference: https://github.com/sovrasov/flops-counter.pytorch
        macs, params = get_model_complexity_info(model1, (3, 512, 512), as_strings = False, print_per_layer_stat = False, verbose = False)
        print("# ------------- ptflops -------------- #")
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print(f"params = {params/1e6}M")
        print(f"macs = {macs/1e9}G")


if __name__ =="__main__":

    # Configure evaluation.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = getArgs()
    logging = getLog(args)
    print('*' * 30)
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    print('*' * 30)

    save_path = args.save_path + args.dataset + args.arch + '/' + args.times +'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)

    model = getModel(args).to(device)

    val_dataloaders, test_dataloaders = getDataset(args)

    Test(test_dataloaders, save_predict=True)
