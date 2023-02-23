import torch
import math
import numpy as np
from torch import tensor,where,zeros,mul,permute
from torch.fft import fft2,fftshift,ifft2,ifftshift
from math import *
from utils import rizer_filters
import cv2
import operator
from functools import reduce
def idx_num(x,y,size):
    result = x + size[0]*(y-1)
    return result
def find(arr):
    #寻找满足数组中等于数组最小元素值的第一个元素
    proc = np.where(arr == min(arr))
    result = proc[0][0]
    return result+1
def get_num(slice_idx,paths,i):
    #返回的结果当前离散线中每个点距所有离散线中哪条线最近
    result =[]
    arr = paths[0,i]
    arr = arr.astype(int)
    for i in range(0,len(arr)):
        result.append(slice_idx[arr[i]-1])
    result = np.array(result)
    return result
def atan2_array(arr1,arr2):
    arr_len = len(arr1)
    arr =[]
    for i in range(0,arr_len):
        result = atan2(arr1[i],arr2[i])
        arr.append(result)
    arr = np.array(arr)
    return arr
def atan2D_array(arr1,arr2):
    H,W = arr1.shape
    arr = np.zeros((H,W))
    for i in range(0,H):
        for j in range(0,W):
            result = atan2(arr1[i,j],arr2[i,j])
            arr[i,j] = result
    return arr
def dim1to2(num,H,W):
    #将一维数组按列排列转换为二维数组
    col = (num-1) / H
    row = (num-1) % H
    col = col.astype(int)
    row = row.astype(int)
    col = col.tolist()
    row = row.tolist()
    return row,col
def Fourier_slice(size,line_type):
    #ft_dims, ang, sub_path_idx, paths = Fourier_slice(size,line_type)
    #size:[H,W,C]
    H = size[0]
    W = size[1]

    Ho = 2 * math.floor(H / 2) + 1
    Wo = 2 * math.floor(W / 2) + 1

    Hc = math.ceil(Ho / 2)
    Wc = math.ceil(Wo / 2)

    #y1
    half1_y = np.arange(1,Wc+1)
    half1_y = half1_y[::-1]
    half2_y = np.arange(Wc+1,W+1)
    half2_y = half2_y[::-1]
    y1 = np.concatenate((half1_y, half2_y), axis=0)

    #x1
    x1 = Hc*(np.ones((1,len(y1))))

    #x2
    proc1_x2 = np.arange(Hc, H+1)
    proc2_x2 = np.arange(1, Hc)
    x2 = np.concatenate((proc1_x2, proc2_x2), axis=0)

    #y2
    y2 = Wc * (np.ones((1, len(x2))))

    #paths
    paths = np.empty((1,Ho+Wo-1),dtype=object)
    paths[0, 1] = idx_num(x1[0, :], y1, [H, W])
    paths[0,Hc+Wc-1] = idx_num(x2,y2[0,:],[H,W])

    #x0
    proc1_x0 = np.arange(Hc + 1, Ho+1)
    proc2_x0 = Ho*np.ones((1,Wc-2))
    x0 = np.concatenate((proc1_x0, proc2_x0[0, :]), axis=0)

    #y0
    proc1_y0 = np.ones((1,Ho-Hc))
    proc2_y0 = np.arange(2, Wc)
    y0 = np.concatenate((proc1_y0[0, :], proc2_y0), axis=0)

    for i in range(1, Hc + Wc - 2):
        X = []
        Y = []
        x = x0[i - 1]
        y = y0[i - 1]

        a = x - Hc
        b = y - Wc
        if line_type == 0:
            d = max(abs(a), abs(b)) / 2
        if line_type == 1:
            d = sqrt(a ** 2 + b ** 2) / 2
        if line_type == 2:
            d = (1 + abs(a) + abs(b)) / 2

        while x != Hc or y != Wc:
            X.append(x)
            Y.append(y)
            temp_ab = a * (y - Wc) - b * (x - Hc)
            if abs(temp_ab + b) <= d:
                x = x - 1
            elif abs(temp_ab + a) <= d:
                y = y + 1
            else:
                x = x - 1
                y = y + 1
        X = np.array(X)
        Y = np.array(Y)

        X1_len = len(X)
        X1 = np.insert(X[::-1], [X1_len], -X + Ho + 1, axis=0)
        X1 = np.insert(X1, [0], Hc, axis=0)

        Y1_len = len(Y)
        Y1 = np.insert(Y[::-1], [Y1_len], -Y + Wo + 1, axis=0)
        Y1 = np.insert(Y1, [0], Wc, axis=0)

        X2 = X1
        Y2 = Wo - Y1 + 1
        #f
        idx1 = (X1 <= H) & (Y1 <= W)
        idx2 = (X2 <= H) & (Y2 <= W)
        #
        paths[0, i + 1] = idx_num(X1[idx1], Y1[idx1], [H, W]);
        paths[0, Ho + Wo - i - 1] = idx_num(X2[idx2], Y2[idx2], [H, W])
    # ang
    proc1_ang1 = np.arange(0, Hc - 1)
    proc2_ang1 = (Hc - 1) * np.ones((1, Wo))

    proc3_ang1 = np.arange(1, Hc - 1)
    proc3_ang1 = proc3_ang1[::-1]
    ang1 = np.concatenate((proc1_ang1, proc2_ang1[0, :], proc3_ang1), axis=0)

    proc1_ang2 = (1 - Wc) * np.ones((1, Hc - 1))
    proc2_ang2 = np.arange(1 - Wc, Wc)
    proc3_ang2 = (Wc - 1) * np.ones((1, Hc - 2))
    ang2 = np.concatenate((proc1_ang2[0, :], proc2_ang2, proc3_ang2[0, :]), axis=0)
    ang = atan2_array(ang2, ang1)

    # ang2D
    oneH = np.arange(-math.floor(H / 2), math.ceil(H / 2))
    oneW = np.arange(-math.floor(W / 2), math.ceil(W / 2))
    oneH = np.array([oneH])
    H1 = oneH.T * np.ones((1, W))
    H2 = oneW * np.ones((H, 1))
    ang2D = atan2D_array(H2, H1)

    idx = (ang2D >= pi / 2)
    ang2D[idx] = ang2D[idx] - pi
    idx = (ang2D < -pi / 2)
    ang2D[idx] = ang2D[idx] + pi

    slice_idx = np.zeros((H, W))
    for i in range(0, H):
        for j in range(0, W):
            ang_dif = abs(ang - ang2D[i, j])
            slice_idx[i, j] = find(ang_dif)

    slice_idx = slice_idx.flatten(order="F")
    sub_path_idx = np.empty((1, Ho + Wo - 2), dtype=object)
    ft_dims = np.zeros((1, Ho + Wo - 2))
    sub_dims = np.zeros((1, Ho + Wo - 2))
    for i in range(1, Ho + Wo - 1):
        num = get_num(slice_idx, paths, i)
        sub = np.where(i == num)
        sub_path_idx[0, i - 1] = sub[0] + 1
        ft_dims[0, i - 1] = len(paths[0, i])
        sub_dims[0, i - 1] = len(sub_path_idx[0, i - 1])
    ft_dims = ft_dims[0,:].astype(int)
    sub_dims = sub_dims[0, :].astype(int)
    paths1 = np.zeros((len(ft_dims), np.max(ft_dims)))
    sub_path_idx1 = np.zeros((len(sub_dims), np.max(sub_dims)), dtype=int)
    row = []
    col = []
    row_sub = []
    col_sub = []
    cos_ang = np.zeros((len(ft_dims),1),dtype=np.float32)
    sin_ang = np.zeros((len(ft_dims),1),dtype=np.float32)
    for i in range(len(ft_dims)):
        paths1[i,0:ft_dims[i]]=paths[0,i+1].astype(int)
        sub_path_idx1[i,:sub_dims[i]]=sub_path_idx[0,i].astype(int)
        sign3 = paths[0, i + 1]
        sign4 = sub_path_idx1[i, :sub_dims[i]]
        row1, col1 = dim1to2(sign3[sign4 - 1], H, W)
        row.append(row1)
        col.append(col1)
        save_path = (sign4-1).copy()
        save_path = save_path.tolist()
        for j in range(len(sign4)):
            row_sub.append(i)
        col_sub.append(save_path)
        cos_ang[i,0] = cos(ang[i])
        sin_ang[i,0] = sin(ang[i])
    row = np.asarray(reduce(operator.add, row))
    col = np.asarray(reduce(operator.add, col))
    row_sub = np.asarray(row_sub)
    col_sub = np.asarray(reduce(operator.add, col_sub))
    cos_ang = torch.from_numpy(cos_ang)
    cos_ang = cos_ang.type(torch.float32).cuda()
    cos_ang.requires_grad = False
    sin_ang = torch.from_numpy(sin_ang)
    sin_ang = sin_ang.type(torch.float32).cuda()
    sin_ang.requires_grad = False
    return ft_dims,cos_ang,sin_ang,paths1,row_sub,col_sub,row,col
def get_filters(ft_dims,scal,factor,batch_size,cos_ang,sin_ang):
    f_filters = zeros((batch_size,3,len(ft_dims),max(ft_dims)),dtype=torch.float32)
    g_filters = zeros((batch_size,3,len(ft_dims),max(ft_dims)),dtype=torch.float32)
    gh_filters = zeros((batch_size,3,len(ft_dims),max(ft_dims)),dtype=torch.complex64)
    for i in range(batch_size):
        for j in range(3):
            for k in range(len(ft_dims)):
                f,g,gh = rizer_filters(ft_dims[k],scal,factor)
                f_filters[i,j,k,0:ft_dims[k]]=f
                g_filters[i,j,k,0:ft_dims[k]]=g
                gh_filters[i,j,k,0:ft_dims[k]]=gh
    f_filters=f_filters.cuda()
    g_filters=g_filters.cuda()
    gh_filters=gh_filters.cuda()
    cos_gh = mul(cos_ang,gh_filters)
    sin_gh = mul(sin_ang,gh_filters)
    filters = torch.cat((f_filters,g_filters,cos_gh,sin_gh),1)
    filters = filters.cuda()
    filters.requires_grad=False
    return filters
#hilbert =(a+bi)(c+dj)=ac+bci+adj-bdk
def get_original(input,ft_dims,paths):
    B,C,H,W = input.shape
    #传入的input的shape为 [B,C,H,W]
    ft_img = fftshift(fft2(input)).permute(0, 1, 3, 2).flatten(start_dim=2, end_dim=3)
    original = zeros((B, C, len(ft_dims), max(ft_dims)), dtype=torch.complex64).cuda()
    original[:, :, :, :] = ft_img[:, :, paths[:, :] - 1]
    return original
def MWT_ana(img, ft_dims, paths,row_sub,col_sub,row,col, filters):
    B,C,H,W=img.shape
    original = get_original(img,ft_dims,paths)
    original = torch.cat((original,original,original,original),1)
    out = zeros((B, C*4, H, W), dtype=torch.complex64).cuda()
    rizer_mwt = mul(filters,original)
    out[:,:,row,col]=rizer_mwt[:,:,row_sub,col_sub]
    out1 = ifft2(ifftshift(out)).real
    return out1
def MWT_sys(img, ft_dims, paths,row_sub,col_sub,row,col, filters):
    B, C, H, W = img.shape
    out = zeros((B, 3, H, W), dtype=torch.complex64).cuda()
    rizer_sys = get_original(img,ft_dims,paths)
    back_mwt = mul(rizer_sys[:, 0:3, :, :], filters[:,0:3,:,:]) + (mul(rizer_sys[:, 3:6, :, :], filters[:, 3:6, :, :])
                                                                   - (mul(rizer_sys[:, 6:9, :, :], filters[:, 6:9, :, :]) + mul(rizer_sys[:, 9:12, :, :], filters[:, 9:12, :, :]))) / 2
    out[:, :, row, col] = back_mwt[:, :, row_sub, col_sub]
    out1 = ifft2(ifftshift(out)).real
    return out1
import time
import os
if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    img1 = cv2.imread('./data/real/GT_rain_acc22.jpg')
    # img2 = cv2.imread('./data/rain100H/norain/norain-003.png')
    img1 = img1[0:301, 0:301, :]
    # img2 = img2[0:101, 0:101, :]
    img1 = torch.tensor(img1).cuda()
    # img2 = torch.tensor(img2)
    img = torch.zeros((2, 301, 301, 3))
    for i in range(2):
        img[i, :, :, :] = img1
    img = img.permute(0, 3, 1, 2).cuda()
    ft_dims,cos_ang,sin_ang, paths,row_sub,col_sub,row,col = Fourier_slice([301, 301], 0)
    filters = get_filters(ft_dims, 1, 0.5,2,cos_ang,sin_ang)
    start = time.time()
    r = MWT_ana(img, ft_dims, paths,row_sub,col_sub,row,col,filters)
    back = MWT_sys(r, ft_dims, paths,row_sub,col_sub,row,col,filters)
    save_out = back.data.cpu().numpy().squeeze()  # back to cpu

    print(time.time()-start)
    print(torch.max(back-img))
    print(torch.min(back-img))



