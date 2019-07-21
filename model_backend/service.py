#-*- coding:utf-8 _*-  
""" 
@author:limuyu
@file: service.py 
@time: 2019/07/20
@contact: limuyu0110@pku.edu.cn

"""

from flask import Flask, request
from model import create_model, SIZE
import numpy as np
import cv2
import base64
import tensorflow as tf

graph = tf.get_default_graph()

app = Flask('demo')

model = create_model(
    input_shape=(SIZE, SIZE, 3),
    n_out=5
)
model.load_weights('./model_checkpoints/resnet_300.h5')

@app.route('/', methods=['GET', 'POST'])
def f():
    if request.method == 'POST':
        global graph
        with graph.as_default():
            img = request.form.get('data')
            img_b64decode = base64.b64decode(img)
            img_array = np.fromstring(img_b64decode, np.uint8)
            image = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
            # image = cv2.imread('tmp.png')
            image = cv2.resize(image, (SIZE, SIZE))
            print(image.shape, image.dtype)
            score_predict = model.predict((image[np.newaxis]) / 255)
            label_predict = np.argmax(score_predict)

            return str(label_predict)
    else:
        return "Error: Please use POST method!!"


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=80,
        debug=True
    )
