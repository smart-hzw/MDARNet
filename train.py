import random
import numpy as np
import torch
import os
import time
from tqdm import tqdm
import argparse
from torch import nn
from SSIM import SSIM
from MDARNet import MDARNet
# from MDARNet_original import  Net_test1
import torch.optim as optim
from torch.autograd import Variable
from utils import findLastCheckpoint,batch_PSNR
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from warmup_scheduler import GradualWarmupScheduler
from MWT_complex import Fourier_slice,get_filters,MWT_ana
from data_RGB import get_training_data, get_validation_data

def get_args():
    #学习率递减策略: 余弦退火算法递减
    parser = argparse.ArgumentParser(description="MDARNet_train")
    parser.add_argument("--batchSize", type=int,required=False, default=8,help="Training batch size")
    parser.add_argument("--pachSize", type=int,required=False, default=128,help="Training batch size")
    parser.add_argument("--epochs", type=int, required=False , default=120, help="Number of training epochs")
    parser.add_argument("--milestone", type=int, default=[30,60,90,120], help="When to decay learning rate; should be less than epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--save_weights", type=str, required=False,default="./models/mixTrain", help='path of log files')
    parser.add_argument("--train_data", type=str, required=False, default='./data/train/', help='path to training data')
    parser.add_argument("--val_data", type=str, required=False, default='./data/test/Rain100H/', help='path to training data')
    parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
    return parser.parse_args()

if __name__ == '__main__':
    opt = get_args()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # loading datasets	
    train_dataset = get_training_data(opt.train_data, {'patch_size': opt.pachSize})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=4,drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(opt.val_data, {'patch_size': opt.pachSize})
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=4, drop_last=False,
                            pin_memory=True)

    # loading model
    H = opt.pachSize
    W = opt.pachSize
    model = MDARNet(H=H,W=W,batch_size=opt.batchSize)

    ######### Loss ###########
    criterion = SSIM()
    criterion1 = nn.L1Loss()

    if opt.use_GPU:
        model = model.cuda()
        criterion.cuda()
        criterion1.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,betas=(0.9, 0.999),eps=1e-8)
    # scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.1)  # learning rates
    warmup_epochs = 4
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs - warmup_epochs,
                                                            eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    ft_dims, cos_ang, sin_ang, paths, row_sub, col_sub, row, col = Fourier_slice([H, W], 0)
    filters1 = get_filters(ft_dims, 1, 0.5, opt.batchSize, cos_ang, sin_ang)
    filters2 = get_filters(ft_dims, 2, 0.5, opt.batchSize, cos_ang, sin_ang)
    initial_epoch = findLastCheckpoint(save_dir=opt.save_weights)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_weights, 'net_epoch%d.pth' % initial_epoch)))
    best_psnr = 0
    for epoch in range(initial_epoch, opt.epochs):
        print("======================",epoch,'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
        epoch_start_time = time.time()
        epoch_loss = 0
        model.train()
        for i, (data) in enumerate(tqdm(train_loader), 0):
        #for i, (input_, target) in enumerate(tqdm(train_loader), 0):
            optimizer.zero_grad()
            target = data [0]
            input_ = data [1]
            if opt.use_GPU:
                input_train, target_train = Variable(input_.cuda()), Variable(target.cuda())

            # 单基因训练模块
            subband1 = MWT_ana(target, ft_dims=ft_dims, paths=paths, row_sub=row_sub, col_sub=col_sub, row=row,           col=col,filters=filters1)
            subband2 = MWT_ana(subband1 [:, 0:3, :, :], ft_dims=ft_dims, paths=paths, row_sub=row_sub, col_sub=col_sub, row=row, col=col,filters=filters2)

            lo1,lo2,out,times = model(input_train)
            pixel_loss = criterion(subband1, lo1)+0.25*criterion(subband2, lo2)
            loss2 = criterion1(subband1, lo1)+0.25*criterion1(subband2, lo2)

            loss = 1 - pixel_loss + loss2

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        model.eval()
        psnr_val_rgb = 0
       
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val [0]
            input_ = data_val [1]
            if opt.use_GPU:
                target = target.cuda()
                input_ = input_.cuda()

            with torch.no_grad():
                lo1,lo1,restored,times = model(input_)

            restored = torch.clamp(restored, 0., 1.)
            psnr_train = batch_PSNR(restored, target, 1.)
            psnr_val_rgb+=psnr_train
        epoch_psnr = psnr_val_rgb/len(val_loader)
        if best_psnr<epoch_psnr:
            best_psnr = epoch_psnr
        print("=========epoch:%d"%epoch,"========epoch PSNR:%f"%epoch_psnr,"========best PSNR:%f"%best_psnr)
        torch.save(model.state_dict(), os.path.join(opt.save_weights, 'net_epoch%d.pth' % (epoch+1)))
        torch.save(model.state_dict(), os.path.join(opt.save_weights, 'net_last.pth'))
        f = open('./MixTrain.txt', mode='a')
        f.write('epoch:' + '%2.4f' % (epoch + 1) + '    ')
        f.write('lr={:.6f}'.format(scheduler.get_lr()[0]) + '    ')
        f.write('epoch_loss:' + '%2.4f' % (epoch_loss / len(train_loader)) + '    ')
        f.write('epoch_psnr:' + '%2.4f' % epoch_psnr + '\n')
        f.close()




