# -*- coding: utf-8 -*-
import torch
from math import sqrt as sqrt
from itertools import product as product
import pdb

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
        self.image_size = cfg['min_dim']
        # print("im size: ")
        # print(self.image_size)
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
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y  如果是两个，就是两个的计算方式
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k] / self.image_size
                # aspect_ratio: 1
                # rel size: min_size
                # s_k = self.min_sizes[k]/self.image_size
                # mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                # s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                # mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]

                    # aspect ratios 代码中只用到了单一的，在这里变成分之一
                    # mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                    # print("cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)")
                    # print(cx, cy, s_k/sqrt(ar), s_k*sqrt(ar))

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        # print("output shape")
        # print(output.shape)
        return output
