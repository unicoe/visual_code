# -*- coding: utf-8 -*-
import torch
from math import sqrt as sqrt
from itertools import product as product
import pdb

# 针对 640*480 这样宽高不一致的图片，进行代码编写

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # self.type = cfg.name
        # print(cfg)
        # pdb.set_trace()
        self.image_size = cfg['min_dim']
        # print("im size: ") # w640 h480
        # print(self.image_size[0], self.image_size[1])

        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        # version is v2_512 or v2_300
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # TODO merge these
        # pdb.set_trace()
        for k, f in enumerate(self.feature_maps):
            # print("prior box 640 480: ")
            # print(k, f)
            # print("image_size : ")
            # print(self.image_size[0])
            # print(self.image_size[1])

            # 修改成 i j 分开的

            for i in range(f[0]):   # h
                for j in range(f[1]): # w
                    # 对应于各种不同的feature maps
                    # print("feature maps size: ")
                    # print(f[0], f[1])
                    f_k_w = self.image_size[0] / self.steps[k][1]   # 用到了steps ，但是没有做修正
                    f_k_h = self.image_size[1] / self.steps[k][0]

                    # print("self steps k 1,0: ")
                    # print(self.steps[k][1], self.steps[k][0])
                    # unit center x,y  如果是两个，就是两个的计算方式
                    cx = (j + 0.5) / f_k_w
                    cy = (i + 0.5) / f_k_h

                    # aspect_ratio: 1
                    # rel size: min_size
                    # 分别对应宽高
                    # s_k_w = self.min_sizes[k] / self.image_size[0]
                    # s_k_h = self.min_sizes[k] / self.image_size[1]
                    # mean += [cx, cy, s_k_w, s_k_h]

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    # s_k_prime_w = sqrt(s_k_w * (self.max_sizes[k] / self.image_size[0]))
                    # s_k_prime_h = sqrt(s_k_h * (self.max_sizes[k] / self.image_size[1]))
                    # mean += [cx, cy, s_k_prime_w, s_k_prime_h]

                    # rest of aspect ratios
                    for tmp_size in range(self.min_sizes[k], self.max_sizes[k], 30):
                        """
                        :type tmp_size from min to max size
                        """
                        s_k_tmp_w = tmp_size / self.image_size[0]
                        s_k_tmp_h = tmp_size / self.image_size[1]
                        for ar in self.aspect_ratios[k]:
                            mean += [cx, cy, s_k_tmp_w* sqrt(ar), s_k_tmp_h / sqrt(ar)]
                        # aspect ratios 代码中只用到了单一的，在这里变成分之一
                        # mean += [cx, cy, s_k_w / sqrt(ar), s_k_h * sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        print("output shape")
        print(output.shape)
        return output
