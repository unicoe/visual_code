import torch
from torchvision import transforms
import cv2
import PIL.Image
import numpy as np
import types
from numpy import random
import scipy.misc as m

import pdb


def mkdir(path):
    import os

    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + 'ok')
        return True
    else:
        print(path + 'failed!')
        return False

# TODO 2019-01-06 11:28:59,需要对程序的正确性进行验证

# 设置随机数种子,保证实验的可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


set_seed(47)

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, seglbl=None):
        for t in self.transforms:
            img, boxes, labels, seglbl = t(img, boxes, labels, seglbl)

        return img, boxes, labels, seglbl


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None, seglbl=None):
        return self.lambd(img, boxes, labels, seglbl)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None, seglbl=None):
        # 增加 seglbl标注,返回np.int32
        # return image.astype(np.float32), boxes, labels, seglbl.astype(np.int32)

        seglbl = np.array(seglbl, dtype=np.int32)

        return image.astype(np.float32), boxes, labels, seglbl


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    # 只对图像进行了操作, 其他的按原本返回
    def __call__(self, image, boxes=None, labels=None, seglbl=None):
        image = image.astype(np.float32)
        image -= self.mean

        seglbl = np.array(seglbl, dtype=np.int32)
        return image.astype(np.float32), boxes, labels, seglbl


class ToAbsoluteCoords(object):
    # 计算boxes,对应于原图的数值坐标
    def __call__(self, image, boxes=None, labels=None, seglbl=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels, seglbl


class ToPercentCoords(object):
    # 计算boxes,对应于原图的百分比坐标
    def __call__(self, image, boxes=None, labels=None, seglbl=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels, seglbl


class Resize(object):
    def __init__(self, size=[640,480]):
        #self.size = size if isinstance(size,tuple) else (size,size)
        self.size = size
        # print("before resize: ")

        # print(self.size)

    def __call__(self, image, boxes=None, labels=None, seglbl=None):
        # print("before resize: ")
        # print(image.shape)
        # print(seglbl.shape)

        image = cv2.resize(image, (self.size[0], self.size[1]))
        # seglbl = cv2.resize(seglbl, (self.size[0], self.size[1]))

        # 要保证 image resize之后, seglbl也能够进行同样的resize操作
        # 明确一下,这个是否能保证??
        # print(type(seglbl))

        # seglbl = PIL.Image.fromarray(np.int32(seglbl))

        # 对语义分割结果的图片处理得到
        #PIL Image resize 格式:  img = img.resize((width, height))
        # print(seglbl.shape)
        # pdb.set_trace()
        # seglbl = seglbl.resize((self.size[0], self.size[1]))

        # classes = np.unique(seglbl)
        seglbl = seglbl.astype(float)
        seglbl = m.imresize(seglbl, (self.size[1], self.size[0]), "nearest", mode="F")
        seglbl = seglbl.astype(int)
        # print(seglbl.shape)
        # print(seglbl)
        # if not np.all(classes == np.unique(seglbl)):
        #     print("WARN: resizing labels yielded fewer classes")
        #
        # if not np.all(np.unique(seglbl[seglbl != self.ignore_index]) < self.n_classes):
        #     print("after det", classes, np.unique(seglbl))
        #     raise ValueError("Segmentation map contained invalid class values")


        # print("after resize: ")
        # print(image.shape)
        # print(seglbl.size)

        # mkdir("/home/user/Disk1.8T/draw_result/augmentations/Resize")
        # mkdir("/home/user/Disk1.8T/draw_result/augmentations/Resize_seglbl")
        #
        # im_name = random.randint(1, 1000000)
        #
        # cv2.imwrite(
        #     "/home/user/Disk1.8T/draw_result/augmentations/Resize/" + str(im_name) + ".png",
        #     image)
        # cv2.imwrite(
        #     "/home/user/Disk1.8T/draw_result/augmentations/Resize_seglbl/" + str(im_name) + ".png",
        #     seglbl)

        return image, boxes, labels, seglbl

# 随机饱和度
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None, seglbl=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels, seglbl

# 随机色调
class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, seglbl=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels, seglbl

# 随机光噪声
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None, seglbl=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels, seglbl


# 颜色空间转换
class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None, seglbl=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels, seglbl

# 随机对比度
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None, seglbl=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels, seglbl

# 随机亮度
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, seglbl=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels, seglbl


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None, seglbl=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels, seglbl


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None, seglbl=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels, seglbl

# 随机裁剪
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox

        *seglbl (?? Image: P): 也将对应的语义分割标注对应过来,这里可能不好调,做好心理准备
            seglbl需要保证格式不变,是P模式,之后转换成tensor

        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
            seglbl (Tensor): the seglbl for image
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None, seglbl=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels, seglbl

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)  ?? 如何理解?
            for _ in range(50):
                current_image = image
                current_seglbl = seglbl

                # print("before: ")
                # print(current_seglbl.shape)
                # print(current_image.shape)


                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2   这里对aspect ratio进行了限制,具体意义呢?
                # 是对随机裁剪的图像进行限制
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],:]

                # print("current_image type: ")
                # print(type(current_image))

                # 使用seglbl,要确保seglbl的格式不会改变
                # print(seglbl.shape)
                current_seglbl = current_seglbl[rect[1]:rect[3], rect[0]:rect[2]]


                # print("current_seglbl shape:  after")
                # print(current_seglbl.shape)
                # print(current_image.shape)


                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                # print(type(current_seglbl)) # "numpy"
                # cv2.imwrite(
                #     "/home/user/Disk1.8T/draw_result/augmentations/RandomSampleCrop/" + str(_) + ".png",
                #     current_image)
                # cv2.imwrite(
                #     "/home/user/Disk1.8T/draw_result/augmentations/RandomSampleCrop_seglbl/" + str(_) + ".png",
                #     current_seglbl)
                return current_image, current_boxes, current_labels, current_seglbl


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels, seglbl):
        if random.randint(2):
            return image, boxes, labels, seglbl

        height, width, depth = image.shape
        ratio = random.uniform(1, 1.5)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image


        # 对seglbl进行对应处理
        s_h, s_w = seglbl.shape
        expand_seglbl = np.zeros((int(s_h*ratio), int(s_w*ratio)), dtype=seglbl.dtype)
        #expand_seglbl[:, :] = [0,0] # 扩展的其他位置都是黑色,
        expand_seglbl[int(top):int(top + height),
                     int(left):int(left + width)] = seglbl
        seglbl = expand_seglbl

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        # mkdir("/home/user/Disk1.8T/draw_result/augmentations/Expand")
        # mkdir("/home/user/Disk1.8T/draw_result/augmentations/Expand_seglbl")
        #
        # im_name = random.randint(1, 1000000)
        #
        # cv2.imwrite(
        #     "/home/user/Disk1.8T/draw_result/augmentations/Expand/" + str(im_name) + ".png",
        #     image)
        # cv2.imwrite(
        #     "/home/user/Disk1.8T/draw_result/augmentations/Expand_seglbl/" + str(im_name) + ".png",
        #     seglbl)

        return image, boxes, labels, seglbl


class RandomMirror(object):
    def __call__(self, image, boxes, labels, seglbl):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
            seglbl = seglbl[:, ::-1]  # 这就是镜像操作吗?确认一下

            # mkdir("/home/user/Disk1.8T/draw_result/augmentations/RandomMirror")
            # mkdir("/home/user/Disk1.8T/draw_result/augmentations/RandomMirror_seglbl")
            #
            # im_name = random.randint(1, 1000000)
            #
            # cv2.imwrite(
            #     "/home/user/Disk1.8T/draw_result/augmentations/RandomMirror/" + str(im_name) + ".png",
            #     image)
            # cv2.imwrite(
            #     "/home/user/Disk1.8T/draw_result/augmentations/RandomMirror_seglbl/" + str(im_name) + ".png",
            #     seglbl)

        return image, boxes, labels, seglbl


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

# 图像矩阵扭曲
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels, seglbl):
        im = image.copy()
        im, boxes, labels, seglbl = self.rand_brightness(im, boxes, labels, seglbl)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels, seglbl = distort(im, boxes, labels, seglbl)

        # mkdir("/home/user/Disk1.8T/draw_result/augmentations/PhotometricDistort")
        # mkdir("/home/user/Disk1.8T/draw_result/augmentations/PhotometricDistort_seglbl")
        #
        # im_name = random.randint(1, 1000000)
        #
        # cv2.imwrite(
        #     "/home/user/Disk1.8T/draw_result/augmentations/PhotometricDistort/" + str(im_name) + ".png",
        #     image)
        # cv2.imwrite(
        #     "/home/user/Disk1.8T/draw_result/augmentations/PhotometricDistort_seglbl/" + str(im_name) + ".png",
        #     seglbl)

        return self.rand_light_noise(im, boxes, labels, seglbl)


class SSDAugmentation(object):
    def __init__(self, size=[640,480], mean=(106.6, 110.3, 107.7)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])


    def __call__(self, img, boxes, labels, seglbl):

        return self.augment(img, boxes, labels, seglbl)
