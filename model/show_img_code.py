# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable

import sys
sys.path.append('/home/user/Disk1.8T/unicoe/pytorch-ssd-2-4')

from data import VOCroot, VOC_CLASSES_ADD_SEG as labelmap
from PIL import Image
from data import AnnotationTransformAddSeg, VOCBboxSeg, BaseTransform, BaseTransformCaltech, VOC_CLASSES_ADD_SEG
import torch.utils.data as data
from ssd_1920_512base_seg_merge_1_170_visible2_4 import build_ssd
import time
import numpy as np
import pdb
# from utils.fcn_visual_utils import visualize_segmentation,get_tile_image
# from utils.mkdir import mkdir
import cv2

print(VOCroot)
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='/home/user/Disk1.8T/unicoe/pytorch-ssd-2-4/weights/output/7_291_seg_merge_base512/ssd1920_0712_100000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='/home/user/Disk1.8T/unicoe/pytorch-ssd-2-4/experments/7_291_seg_merge_base512/res/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to test model')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh,trained_model):
    # dump predictions and assoc. ground truth to text file for now
    # pdb.set_trace()
    time_tup = time.localtime(time.time())
    format_time = '%Y_%m_%d_%a_%H_%M_%S'
    cur_time = time.strftime(format_time, time_tup)

    filename = save_folder + cur_time + '.txt'
    num_images = len(testset)

    visualizations = []
    label_preds = []

    for i in range(num_images):
        img = testset.pull_image(i)
        img_id = testset.pull_id(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0), volatile=True)
        if cuda:
            x = x.cuda()

        y,fcn_output, fcn_visible_output = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image   要将检测结果转换回原图上
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])

        print(" scale each detection back up to the image ")
        print(scale)
        pred_num = 0
        ls = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] > 0:
                print("detections: ")
                print(detections[0, i, j, 0])
                score = detections[0, i, j, 0]
                label_name = labelmap[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(img_id + ' ' + str(round(score, 4)))

                    for c in coords:
                        f.write(' ')
                        f.write(str(round(c, 2)))
                    f.write('\n')
                    ls.append([round(score, 2), round(coords[0], 2), round(coords[1], 2), round(coords[2], 2),
                               round(coords[3], 2)])
                    pass
                j += 1
            print(j)

        _imgpath = "/home/user/Disk1.8T/unicoe/pytorch-ssd-2-4/data/VOCdevkit/VOC0712/JPEGImages/" + img_id + ".jpg"
        print(_imgpath)
        im = cv2.imread(_imgpath, cv2.IMREAD_COLOR)

        for idx_bbox in ls:

            x1 = int(float(idx_bbox[1]))
            y1 = int(float(idx_bbox[2]))
            x2 = int(float(idx_bbox[3]))
            y2 = int(float(idx_bbox[4]))

            if float(idx_bbox[0]) >= 0.5:
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(im, (x1, y1), (x1 + 50, y1 - 20), (0, 255, 0), -1)
                cv2.putText(im, str(float(idx_bbox[0]))[:4], (x1, y1), 0, 0.6, (0, 0, 0), 1)
        im = cv2.resize(im,(960,540))
        # cv2.imshow('imshow', im)
        # cv2.waitKey(500)
        return im
        #print(type(fcn_output))
        # handle fcn_output
    #     lbl_pred = fcn_visible_output.data.max(1)[1].cpu().numpy()[:,:,:]
    #     """
    #     这里的max(1)就是取i编号1维度的最大值,返回最大值对应的index
    #     """
    #
    #     # 转换成功
    #     # TODO 引入图片seg_gt,进行结果可视化 19/1/6 21:48
    #     img = img
    #
    #     # print(img.shape)
    #     # print("img shape: before visual")
    #
    #
    #     label_preds.append(lbl_pred)
    #
    #     #print(img.shape)
    #
    #     viz = visualize_segmentation(
    #         lbl_pred=lbl_pred,
    #         img=img,
    #         n_class=num_classes,
    #         label_names= np.array(['background','person',])
    #     )
    #     visualizations.append(viz)
    #
    #
    # # viz = get_tile_image(visualizations)
    # # skimage.io.imsave('/home/user/Disk1.8T/unicoe/pytorch-ssd-2-4-1/experments/7_291_seg_merge_base512/visual/viz_evaluate.png', viz)
    # # 替换保存方式,将所有语义分割的测试结果都保存,然后看效果,
    # get_model_id = trained_model.split("/")[-1][12:-4]
    # mkdir("/home/user/Disk1.8T/unicoe/pytorch-ssd-2-4-1/experments/7_291_seg_merge_base512/visual" + get_model_id)
    #
    # for i_vis in range(len(visualizations)):
    #     skimage.io.imsave(
    #         "/home/user/Disk1.8T/unicoe/pytorch-ssd-2-4-1/experments/7_291_seg_merge_base512/visual" + get_model_id + "/" + str(
    #         i_vis) + ".png", visualizations[i_vis])ssd_1920_512base_seg_merge_1_170_visible2_4.py


def get_img():

    # load net
    num_classes = 2 # +1 background
    net = build_ssd('test', [1920,1080], num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCBboxSeg(args.voc_root, [('0712', 'test')], None, AnnotationTransformAddSeg())
    if args.cuda:
        net = net.cuda()

    im = test_net(args.save_folder, net, args.cuda, testset,
             BaseTransformCaltech(net.size, (118.3, 118.1, 112.5)),
             thresh=args.visual_threshold, trained_model=args.trained_model)
    return im