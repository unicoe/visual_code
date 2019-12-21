import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from data import v


class DetectSegAddLoss(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        cfg = v[str(size)]
        self.variance = cfg['variance']
        self.output = torch.zeros(1, self.num_classes, self.top_k, 5)

    def forward(self, loc_data, conf_data, prior_data, fcn_output, fcn_output_ass1, fcn_output_ass2):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]


            fcn_output: seg result
        """
        # print('''batch_size {.1d}'''.format(loc_data.size(0)))
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        self.output.zero_()
        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.t().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes).transpose(2, 1)
            self.output.expand_(num, self.num_classes, self.top_k, 5)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            num_det = 0
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                self.output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = self.output.view(-1, 5)
        _, idx = flt[:, 0].sort(0)
        _, rank = idx.sort(0)
        flt[(rank >= self.top_k).unsqueeze(1).expand_as(flt)].fill_(0)

        #imgs     = data.data.cpu()
        # label_preds = []
        # lbl_pred = fcn_output.data.max(1)[1].cpu().numpy()[:, :, :]
        #lbl_true = target.data.cpu()
        # for img, lt, lp in zip(lbl_pred):
            #img, lt = self.val_loader.dataset.untransform(img, lt)
            #label_trues.append(lt)
            #label_preds.append(lp)
            # if len(visualizations) < 9:
            #     viz = fcn.utils.visualize_segmentation(
            #         lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
            #     visualizations.append(viz)

        # 最简单的输出
        return self.output, fcn_output, fcn_output_ass1, fcn_output_ass2
