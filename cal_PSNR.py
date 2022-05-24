from math import log10, sqrt
import numpy as np
from argparse import ArgumentParser
import cv2
import os


gt_path = './myDataset/my_dataset_final/test/gt/'
target_path = './Learnbale_Bandpass_Filter/testing_result/'

def validate_psnr(path1, path2):

    gt_list = []
    ns_list = []
    
    tmp = os.listdir(path1)
    tmp.sort()
    for k in tmp:
        gt_list.append(os.path.join(path1,k))

    tmp = os.listdir(path2)
    tmp.sort()
    for k in tmp:
        ns_list.append(os.path.join(path2,k))

    psnr = 0
    count = 0
    print(len(gt_list))
    print(len(ns_list))
    for i in range(len(gt_list)):	
        count += 1
        gt = np.float64(cv2.imread(gt_list[i]))
        ns = np.float64(cv2.imread(ns_list[i],1))	
        mse = np.mean((ns-gt)**2)
        _psnr = 0
        if mse==0:
            _psnr=100
        else:
            _psnr = 20*log10(255/sqrt(mse))
        print(_psnr)
        psnr += _psnr

    print(count)
    print (psnr/count)
    return psnr/count


validate_psnr(gt_path, target_path)
