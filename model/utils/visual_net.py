# -*- coding: utf-8 -*-
# @Time    : 18-12-11 下午4:12
# @Author  : unicoe
# @Email   : unicoe@163.com
# @File    : visual_net.py
# @Software: PyCharm


from torch.autograd import Variable
from utils.visualize import  make_dot
from ssd_resnet_512 import build_ssd
#from ssd_640_480_512base_seg_merge_1_280_add_attention import build_ssd
from ssd_640_480_512base_seg_merge_1_170_orginal import build_ssd
# from ssd_640_480_512base_seg_merge_3_6 import vgg
#from ssd_640_480_512base_seg_merge_1_170_vis import vgg
import torch

"""
计算图可视化
"""

# vgg ssd
# ssd_net = build_ssd('train', 300, 2)
# net = ssd_net
#
# x = Variable(torch.randn(1,3,300,300))
#
# y = net(x)
# print(net)
#
# dot = make_dot(y, params=dict(net.named_parameters()))
# print(dot)
# dot.view()


# resnet
#
# from layers.resnet import *
#
# resnet = resnet50()
# net = resnet
#
# x = Variable(torch.randn(1,3,300,300))
#
# y = net(x)
# print(net)
#
# dot = make_dot(y, params=dict(net.named_parameters()))
# dot.view()

# resnet ssd 512

# base = {
#     '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
#             512, 512, 512],
#     '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
#             512, 512, 512],
# }
# net = vgg(base['512'], 3)
#
# print(net)
#
# x = (torch.randn(1,3,480,640))
# for k in net.state_dict():
#
#     print(k)

# y = net(x)
# # print(net)
# # cnt = 1



# print(net)

# dot = make_dot(y, params=dict(net.named_parameters()))
# dot.view()

ssd_net = build_ssd('train', [640,480], 2)
net = ssd_net

x = Variable(torch.randn(1,3,480,640))

y = net(x)
print(net)

dot = make_dot(y, params=dict(net.named_parameters()))
dot.view()