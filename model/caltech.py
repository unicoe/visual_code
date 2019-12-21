# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
sys.path.append('/home/user/Disk1.8T/unicoe/pytorch-ssd-2')
from data import VOCroot, VOC_CLASSES as labelmap
from PIL import Image
from data import AnnotationTransform,AnnotationTransform_caltech, VOCDetection, BaseTransform, BaseTransformCaltech, VOC_CLASSES
import torch.utils.data as data
from ssd_640_480_512base_again import build_ssd
from log import log
import time
import pdb
import cv2

# load net
num_classes = 2 # +1 background
net = build_ssd('test', [640,480], num_classes) # initialize SSD
trained_model = '/home/user/Disk1.8T/unicoe/pytorch-ssd-2/weights/output/again_1_141_640_480_512baseline/ssd640_0712_100000.pth'
net.load_state_dict(torch.load(trained_model))
net.eval()
log.l.info('Finished loading model!')
# load data
testset = VOCDetection(VOCroot, [('0712', 'test')], None, AnnotationTransform_caltech())
net = net.cuda()

# evaluation
print("net size: ")
print(net.size)

time_tup = time.localtime(time.time())
format_time = '%Y_%m_%d_%a_%H_%M_%S'
cur_time = time.strftime(format_time, time_tup)
save_folder = '/home/user/Disk1.8T/unicoe/pytorch-ssd-2/experments/again_1_141_640_480_512baseline/res/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
filename = save_folder + cur_time + '.txt'
num_images = len(testset)
for i in range(num_images):
    log.l.info('Testing image {:d}/{:d}....'.format(i+1, num_images))
    img = testset.pull_image(i)
    img1 = img
    img_id = testset.pull_id(i)
    transform = BaseTransformCaltech(net.size, (106.6, 110.3, 107.7))
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0), volatile=True)

    # with open(filename, mode='a') as f:
    #     f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
    #     for box in annotation:
    #         f.write('label: '+' || '.join(str(b) for b in box)+'\n')

    x = x.cuda()

    y = net(x)      # forward pass
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
        while detections[0, i, j, 0]  > 0:
            print("detections: ")
            print(detections[0,i,j,0])
            # if pred_num == 0:
            #     with open(filename, mode='a') as f:
            #         f.write('PREDICTIONS: '+'\n')
            score = detections[0, i, j, 0]
            label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])
            pred_num += 1
            with open(filename, mode='a') as f:
                # f.write(str(pred_num)+' label: '+label_name+' score: ' +
                #         str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')

                f.write(img_id + ' ' + str(round(score,4)))

                for c in coords:
                    f.write(' ')
                    f.write(str(round(c,2)))
                f.write('\n')
                ls.append([round(score,2), round(coords[0],2), round(coords[1],2), round(coords[2],2), round(coords[3],2)])
                #print(ls)
                # pdb.set_trace()
                pass
            j += 1
        print(j)
    print(type(img1))
    print(img_id)
    _imgpath = "/home/user/Disk1.8T/unicoe/pytorch-ssd-2/data/VOCdevkit/VOC0712/JPEGImages/"+ img_id+".jpg"
    print(_imgpath)
    im = cv2.imread(_imgpath, cv2.IMREAD_COLOR)

    for idx_bbox in ls:

        x1 = int(float(idx_bbox[1]))
        y1 = int(float(idx_bbox[2]))
        x2 = int(float(idx_bbox[3]))
        y2 = int(float(idx_bbox[4]))

        if float(idx_bbox[0]) >= 0.3:

            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(im, (x1, y1), (x1 + 20, y1 - 10), (0, 255, 0), -1)
            cv2.putText(im, str(float(idx_bbox[0]))[:4], (x1, y1), 0, 0.3, (0, 0, 0), 1)


