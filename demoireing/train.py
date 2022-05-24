import os, cv2, random, keras
from skimage.measure import compare_ssim
import numpy as np
from model import *
from keras import optimizers
from keras.utils import multi_gpu_model
from math import log10
from utils import *
from datetime import datetime
import time
from PIL import Image

#weight file path
weight_path = './model/MBCNN_weights_1.h5'

#train data path
train_gt_path = '../dataset/train/gt/'
train_ns_path = '../dataset/train/moire/'
tmp_model = './tmp_model/'

multi_output = False
model = MBCNN(64,multi_output)

#Generating training data
def generate_training(train_list):
    train_gt_list = []
    train_ns_list = []
    name_list = []
    _width = 128
    count = 0
    for f in train_list:
        count+=1
        gt = cv2.imread(train_gt_path+f)
        print(train_gt_path+f)
        ns = cv2.imread(train_ns_path+f)
        gt = gt.astype(np.float32)/255.0
        ns = ns.astype(np.float32)/255.0
        name_list.append(f)
        #训练时因显存原因需将图片分割为更小的分辨率    
        for i in range(0,1024,_width):
            for j in range(0,1024,_width):
                _gt = gt[i:i+_width,j:j+_width]
                _ns = ns[i:i+_width,j:j+_width]
                train_gt_list.append(_gt)
                train_ns_list.append(_ns)
    return train_gt_list, train_ns_list

keras.backend.tensorflow_backend.set_session(get_session())
train_list = os.listdir(train_gt_path)
train_list = list_filter(train_list, '.jpg')
train_gt_list, train_ns_list = generate_training(train_list)

train_ns_list = np.array(train_ns_list)
train_gt_list = np.array(train_gt_list)

print(train_ns_list.shape)
print(train_gt_list.shape)

#model.load_weights(weight_path, by_name = True)

adam = optimizers.Adam(lr=1e-6)
model.compile(loss='mean_absolute_error',optimizer=adam)

model.fit(train_ns_list,train_gt_list,epochs=40,batch_size=4)

model.save_weights(tmp_model+'tmp_weights.h5')

exit(0)
