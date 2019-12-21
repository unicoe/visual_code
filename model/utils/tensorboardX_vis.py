# -*- coding: utf-8 -*-
# @Time    : 19-2-28 下午7:59
# @Author  : unicoe
# @Email   : unicoe@163.com
# @File    : tensorboardX_vis.py
# @Software: PyCharm
from torch.autograd import Variable
from utils.visualize import  make_dot
from ssd_640_480 import build_ssd
#from ssd_resnet_512 import build_ssd
from ssd_640_480_512base_seg_merge_1_170_vis import build_ssd
import torch
from tensorboardX import SummaryWriter

import torch.onnx

"""
计算图可视化
"""

#vgg ssd
# ssd_net = build_ssd('train', 300, 2)
# net = ssd_net
#
# x = Variable(torch.randn(1,3,300,300))
# #
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

ssd_net = build_ssd('train', [640,480], 2)
net = ssd_net

x = torch.randn(1,3,480,640)

y = net(x)
print(net)

dot = make_dot(y, params=dict(net.named_parameters()))
dot.view()

print(ssd_net)

with SummaryWriter(comment='ssd_net') as w:
    w.add_graph(ssd_net,x)

# torch.onnx.export(net, x, "ssd_net.proto", verbose=True)