import tensorflow as tf
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
import math 

test_path = '../dataset/test/moire/'
weight_path = './model/MBCNN_weights_3.h5'

multi_output = True
model = MBCNN(64,multi_output)

def test(model):
    print("==============================================")
    file_list = os.listdir(test_path)
    file_list = list_filter(file_list,'.JPG')

    _width = 1024
    for f in file_list:
        ns = cv2.imread(test_path+f)
        h, w = ns.shape[0], ns.shape[1]
        m, n = math.ceil(h/_width), math.ceil(w/_width)
        tmp_h = m*_width
        tmp_w = n*_width
        # 将原图长宽填补至_width的整数倍
        tmp = np.zeros([tmp_h,tmp_w,3])
        tmp.fill(255)
        result = np.zeros([tmp_h,tmp_w,3])
        tmp[:h,:w,:] = ns
        print(tmp.shape)
        #切割后输入神经网络得到相应区域的输出
        for i in range(0,tmp_h,_width):
            for j in range(0,tmp_w,_width):
                _input = tmp[i:i+_width,j:j+_width]
                _input = _input.astype(np.float32)/255.0
                _input = _input.reshape((1,)+_input.shape)
                start = time.clock()
                _output = model.predict(_input)
                end = time.clock()
                print(f, end-start)
                #处理网络输出结果并填入结果图
                _output = _output[-1]
                _output = _output[0]
                _output[_output>1] = 1
                _output[_output<0] = 0
                _output = _output*255.0
                _output = np.round(_output).astype(np.uint8)
                result[i:i+_width,j:j+_width] = _output
        #截取出原图分辨率大小的结果图并输出
        restore_image1 = result[:h,:w,:]
        cv2.imwrite('results/'+f,restore_image1)


os.environ["CUDA_VISIBLE_DEVICES"]="0"
keras.backend.tensorflow_backend.set_session(get_session())

model.summary()
model.load_weights(weight_path, by_name = True)
test(model)

exit(0)
