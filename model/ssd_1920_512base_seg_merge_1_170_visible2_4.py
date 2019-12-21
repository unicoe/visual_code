# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers.functions.prior_box_640_480 import PriorBox
from layers.modules.l2norm import L2Norm
from layers.functions.detection_seg_1920_1080_visible import DetectSeg
from data import v as cfg
import os
from IPython import embed
"""
添加新任务,预测行人的可见部分
2019年05月31日16:51:20
"""

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(cfg["1920_1080_base512"])
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)

        self.L2Norm1 = L2Norm(512, 20)
        self.L2Norm2 = L2Norm(512, 20)
        self.L2Norm3 = L2Norm(1024, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if self.phase == 'test':
            self.softmax = nn.Softmax()
            self.detect_seg = DetectSeg(num_classes, self.size, 0, 600, 0.0001, 0.55)

        # add seg
        self.score_fr = nn.Conv2d(1024, 512, 1)
        self.score_pool3 = nn.Conv2d(256, 512, 1)
        self.score_pool4 = nn.Conv2d(512, 512, 1)

        self.upscore8 = nn.ConvTranspose2d(
            512, self.num_classes,
            16, stride=8, padding=4, bias=False)

        self.upscore_pool4 = nn.ConvTranspose2d(
            512, 512,
            (3,4), stride=2, padding=(0, 1), bias=False)

        # add seg_vis
        self.score_fr_vis = nn.Conv2d(1024, 512, 1)
        self.score_pool3_vis = nn.Conv2d(256, 512, 1)
        self.score_pool4_vis = nn.Conv2d(512, 512, 1)

        self.upscore8_vis = nn.ConvTranspose2d(
            512, self.num_classes,
            16, stride=8, padding=4, bias=False)

        self.upscore_pool4_vis = nn.ConvTranspose2d(
            512, 512,
            (3,4), stride=2, padding=(0,1), bias=False)

        self.upscore16 = nn.ConvTranspose2d(
            1024, self.num_classes,
            32, stride=16, padding=(4,8), bias=False)

        self.upscore16_vis = nn.ConvTranspose2d(
            1024, self.num_classes,
            32, stride=16, padding=(4,8), bias=False)

        # merge feature
        self.resume_channel = nn.Conv2d(512, 512, 1)

        # down scale
        self.downscale_8_16 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=(0,1))

        # merge feature
        self.resume_channel_vis = nn.Conv2d(512, 512, 1)

        # down scale
        self.downscale_8_16_vis = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=(0,1))

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300]. or [batch,3,512,512]

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch,num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors,4]
                    3: priorbox layers, Shape: [num_priors,4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
            if k == 16:
                seg_pool3 = x
                seg_pool3_vis = x
        for_l2norm_feature = x

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
            if k == 23:
                seg_pool4 = x
                seg_pool4_vis = x
        fcn_x_vis = x
        fcn_x = x
        # vgg_fc7_feature = x

        # 添加可见部分分支，对行人的可见部分进行预测，先有个简单的结果，
        # fcn_visible = x
        # print(fcn_visible.shape)
        #fcn_visible_output = self.upscore8_visible(fcn_visible)

        fcn_x_vis = self.score_fr_vis(fcn_x_vis)
        fcn_x = self.score_fr(fcn_x)
        #print(fcn_x.shape)
        seg_upscore2_vis = fcn_x_vis
        seg_upscore2 = fcn_x

        fcn_x_vis = self.score_pool4_vis(seg_pool4_vis*0.01)
        fcn_x = self.score_pool4(seg_pool4*0.01)
        #print(fcn_x.shape)
        seg_score_pool4c_vis = fcn_x_vis
        seg_score_pool4c = fcn_x

        fcn_x_vis = seg_upscore2_vis + seg_score_pool4c_vis
        fcn_x = seg_upscore2 + seg_score_pool4c
        #print(fcn_x.shape)
        fcn_x_vis = self.upscore_pool4(fcn_x_vis)
        fcn_x = self.upscore_pool4(fcn_x)
        #print(fcn_x.shape)
        seg_upscore_pool4_vis = fcn_x_vis
        seg_upscore_pool4=fcn_x
        
        fcn_x_vis = self.score_pool3_vis(seg_pool3_vis*0.001)
        fcn_x = self.score_pool3(seg_pool3*0.001)
        #print(fcn_x.shape)
        seg_score_pool3c_vis = fcn_x_vis
        seg_score_pool3c = fcn_x

        fcn_x_vis = seg_upscore_pool4_vis + seg_score_pool3c_vis
        fcn_x = seg_upscore_pool4 + seg_score_pool3c
        #print(fcn_x.shape)
        seg_merger_feature = fcn_x
        vis_merger_feature = fcn_x_vis

        # fcn_output = self.upscore8(fcn_x)
        # fcn_visible_output  = self.upscore8_vis(fcn_x_vis)
        # seg result fcn_output
        # print("before merge feature: ")
        # print(for_l2norm_feature.shape)
        # print(seg_merger_feature.shape)

        #pre_merge_feature = seg_merger_feature * vis_merger_feature

        merger_feature = for_l2norm_feature*self.resume_channel(seg_merger_feature)
        merger_feature = self.L2Norm1(merger_feature)

        merger_feature = merger_feature * self.resume_channel_vis(vis_merger_feature)
        s = self.L2Norm2(merger_feature)
        sources.append(s)

        second_merge_feature = self.downscale_8_16(seg_merger_feature)
        second_vis_feature = self.downscale_8_16_vis(vis_merger_feature)

        fcn_output = self.upscore16(second_merge_feature)
        fcn_visible_output = self.upscore16_vis(second_vis_feature)

        x = x*second_merge_feature
        x = x*second_vis_feature
        x = self.L2Norm3(x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # print(l(x).permute(0, 2, 3, 1).contiguous().shape)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            # print(c(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect_seg(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data)),                 # default boxes
                fcn_output,
                fcn_visible_output,
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors,
                fcn_output, # seg output
                fcn_visible_output,
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def add_extras(cfg, size, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    # SSD512 need add one more Conv layer(Conv12_2)
    if size == 512:
        # layers += [nn.Conv2d(in_channels, 256, kernel_size=4, padding=1)]
        layers += [nn.Conv2d(in_channels, 256, kernel_size=3)]
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    #'512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128],
    '512': [512, 'S', 512, 256, 'S', 512, 256, 512, 256, 512, 256],
}
mbox = {
    '300': [6, 6, 10, 10, 10, 10],  # number of boxes per feature map location
    '512': [6, 6, 10, 10, 20, 20, 20],
}


def build_ssd(phase, size=[1920,1080], num_classes=2):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size[0] != 1920 and size[1] != 1080:
        print("Error: Sorry only SSD1920_10800 is supported currently!")
        return


    # TODO 修改300-> 640 之类
    return SSD(phase, size, *multibox(vgg(base['512'], 3),
                                add_extras(extras['512'], 512, 1024),
                                mbox['512'], num_classes), num_classes=2)