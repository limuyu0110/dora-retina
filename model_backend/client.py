#-*- coding:utf-8 _*-  
""" 
@author:limuyu
@file: client.py 
@time: 2019/07/20
@contact: limuyu0110@pku.edu.cn

"""

import requests

s = requests

data = {'str': "hahaha"}
r = s.post('http://23.96.49.78:80/', data)

print(r.status_code)
print(r.headers['content-type'])
print(r.encoding)
print(r.text)