import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import time
import torch
import utils
import lpips
import cv2
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from MDARNet import MDARNet
from skimage import img_as_ubyte
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio,structural_similarity

def imgToTensor(img):
    b, g, r = cv2.split(img)
    y = cv2.merge([r, g, b])
    y = utils.normalize(np.float32(y))
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    y = Variable(torch.Tensor(y),requires_grad=False)
    return y

def TensorToimg(input):
    save_out = np.uint8(255 * input.data.cpu().numpy().squeeze())
    save_out = save_out.transpose(1, 2, 0)
    b, g, r = cv2.split(save_out)
    save_out = cv2.merge([r, g, b])
    return save_out

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint,strict=False)
    except:
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def compute_lpips(norain_path,rain_path):
    target_file = os.listdir(norain_path)
    output_file = os.listdir(rain_path)
    data_len = len(target_file)
    psnr = []
    ssim = []
    lpips_val = []
    lpips_loss = lpips.LPIPS(net='alex')
    lpips_loss.eval()
    for i in range(data_len):
        target_path = os.path.join(norain_path,target_file[i])
        output_path = os.path.join(rain_path,output_file[i])
        target = cv2.imread(target_path)
        output = cv2.imread(output_path)
        h,w,c = output.shape
        target = target[0:h,0:w,:]
        target_copy = target.copy()
        output_copy = output.copy()
        target_copy = np.expand_dims(target_copy.transpose(2, 0, 1), 0)
        output_copy = np.expand_dims(output_copy.transpose(2, 0, 1), 0)
        target_copy = torch.Tensor(target_copy)
        output_copy = torch.Tensor(output_copy)

        psnr.append(peak_signal_noise_ratio(output, target, data_range=255.))
        ssim.append(structural_similarity(output,target,multichannel=True))
        lpips_val.append(lpips_loss(output_copy, target_copy))
        print("image%s:" % output_file[i] + '    ' + 'PSNR:%2.4f' % psnr[i] + '    ' + 'SSIM:%2.4f' % ssim[i]+'    '+'LPIPS:%2.4f'%lpips_val[i])
    avg_psnr = sum(psnr)/data_len
    avg_ssim = sum(ssim)/data_len
    avg_lpips = sum(lpips_val)/data_len
    print("PSNR: %.4f " % (avg_psnr))
    print("SSIM: %.4f" % (avg_ssim))
    print('LPIPS: %.4f' % (avg_lpips))
    return avg_psnr

def test():
    parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')

    parser.add_argument('--input_dir', default='./data/test/', type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='./data/test/', type=str, help='Directory for results')
    parser.add_argument('--weights', default='./models/mixTrain/net_epoch114.pth', type=str,
                        help='Path to weights')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    os.environ ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # datasets = ['Rain100H', 'Rain100L','Test100','Test1200','Test2800']
    datasets = ['Test100']
    for dataset in datasets:
        rgb_dir_test = os.path.join(args.input_dir, dataset, 'input')
        print(rgb_dir_test)

        result_dir = os.path.join(args.result_dir,dataset,'derain')
        input_files = os.listdir(rgb_dir_test)
        data_len = len(input_files)
        utils.mkdir(result_dir)

        list_h = []
        list_w = []
        for i in range(data_len):
            print("=========img%d is processing"%i)
            input_path = os.path.join(rgb_dir_test, input_files[i])
            input_ = cv2.imread(input_path)
            # input_ = data_test [0]
            input_ = imgToTensor(input_)
            b, c, h, w = input_.shape
            flag = True
            for i in range(len(list_h)):
                if h == list_h[i] and w == list_w[i]:
                    flag = False
            if flag == True:
                list_h.append(h)
                list_w.append(w)
        model = []
        for i in range(len(list_h)):
            model_restoration = MDARNet(H=int(list_h[i]), W=int(list_w[i]), batch_size=1)
            utils.print_network(model_restoration)
            # load_checkpoint(model_restoration, args.weights)
            load_checkpoint(model_restoration, args.weights)
            model.append(model_restoration)
            print("===>Testing using weights: %d"%i,'/%d'%len(list_h),"H=%d"%list_h[i],"W=%d"%list_w[i], args.weights)
            model[i].cuda()
            model[i] = nn.DataParallel(model[i])
            model[i].eval()
        with torch.no_grad():
            sum_time = 0
            for i in range(data_len):

                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                input_path = os.path.join(rgb_dir_test, input_files [i])
                input_ = cv2.imread(input_path)
                input_ = imgToTensor(input_)

                B,C,H, W = input_.shape
                for j in range(len(list_h)):
                    if H==list_h[j] and W== list_w[j]:
                        lo1,lo2,restored,run_time = model[j](input_)
                        sum_time += run_time
                        print('=======the', input_files [i], 'is processing...','run time:%f'%run_time)
                restored = torch.clamp(restored, 0, 1)
                restored = TensorToimg(restored)
                cv2.imwrite(os.path.join(result_dir, input_files[i]),restored)
            print("avg time:%f" % (sum_time / data_len))


if __name__=="__main__":
    test()
    rain_path = './data/test/Test100/derain'
    norain_path = './data/test/Test100/target'
    compute_lpips(norain_path=norain_path,rain_path=rain_path)
