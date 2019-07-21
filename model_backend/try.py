#-*- coding:utf-8 _*-  
""" 
@author:limuyu
@file: try.py 
@time: 2019/07/20
@contact: limuyu0110@pku.edu.cn

"""

import cv2
from model import create_model, SIZE
import numpy as np
import tensorflow as tf

graph = tf.get_default_graph()

image = cv2.imread('tmp.jpg')
image = cv2.resize(image, (SIZE, SIZE))
model = create_model(
    input_shape=(SIZE, SIZE, 3),
    n_out=5
)
print(image.shape, image.dtype)
with graph.as_default():
    model.load_weights('./model_checkpoints/resnet_300.h5')
    score_predict = model.predict((image[np.newaxis]) / 255)
    label_predict = np.argmax(score_predict)


print(label_predict)