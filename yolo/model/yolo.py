import math
import torch
import torchvision.transforms.functional as F
from torch import nn

from .head import Head
from .backbone import darknet_pan_backbone
from .transform import Transformer
from .backbone import mobilevit_backbone
from .model import MobileViT_S

class YOLOv5(nn.Module):
    def __init__(self, num_classes, model_size=(0.33, 0.5),
                 match_thresh=4, giou_ratio=1, img_sizes=(320, 416),
                 score_thresh=0.1, nms_thresh=0.6, detections=100):
        super().__init__()
        # original
        anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ]
        # [320, 416]
        # anchors = [
        #     [[6.1, 8.1], [20.6, 12.6], [11.2, 23.7]],
        #     [[36.2, 26.8], [25.9, 57.2], [57.8, 47.9]],
        #     [[122.1, 78.3], [73.7, 143.8], [236.1, 213.1]],
        # ]
        loss_weights = {"loss_box": 0.05, "loss_obj": 1.0, "loss_cls": 0.5}
        
        # self.backbone = darknet_pan_backbone(
        #     depth_multiple=model_size[0], width_multiple=model_size[1]) # 7.5M parameters
        # # wherever you assemble the model
        # def load_mobilevit_weights(model_path):
        #     # Create an instance of the MobileViT model
        #     net = MobileViT_S()

        #     # Load the PyTorch state_dict
        #     state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']

        #     # Since there is a problem in the names of layers, we will change the keys to meet the MobileViT model architecture
        #     for key in list(state_dict.keys()):
        #         state_dict[key.replace('module.', '')] = state_dict.pop(key)

        #     # Once the keys are fixed, we can modify the parameters of MobileViT
        #     net.load_state_dict(state_dict)

        #     return net
        # net = load_mobilevit_weights(r"ckpts\model_best.pth.tar")  # returns MobileViT_S instance
        if isinstance(img_sizes, int):
            img_sizes = (img_sizes, img_sizes)
        self.backbone = mobilevit_backbone(img_size=img_sizes[0])
        
        in_channels_list = self.backbone.body.out_channels_list
        strides = (8, 16, 32)
        num_anchors = [len(s) for s in anchors]
        predictor = Predictor(in_channels_list, num_anchors, num_classes, strides)
        
        self.head = Head(
            predictor, anchors, strides, 
            match_thresh, giou_ratio, loss_weights, 
            score_thresh, nms_thresh, detections)
        
        self.transformer = Transformer(
            min_size=img_sizes[0], max_size=img_sizes[1], stride=max(strides))
    
    def forward(self, images, targets=None):
        def preprocess_tensor(img: torch.Tensor):
            # img shape: [C,H,W], dtype float or uint8
            img = F.center_crop(img, [640, 640])   # crop to square
            img = F.resize(img, [640, 640])        # resize to 256x256
            if img.dtype != torch.float32:
                img = img.float() / 255.0          # normalize to [0,1] if needed
            return img
        
        images = [preprocess_tensor(img) for img in images]
        images = torch.stack(images, dim=0)   # [B,C,256,256]
        
        images, targets, scale_factors, image_shapes = self.transformer(images, targets)
        features = self.backbone(images)
        
        if self.training:
            losses = self.head(features, targets)
            return losses
        else:
            max_size = max(images.shape[2:])
            results, losses = self.head(features, targets, image_shapes, scale_factors, max_size)
            return results, losses
        
    def fuse(self):
        # fusing conv and bn layers
        for m in self.modules():
            if hasattr(m, "fused"):
                m.fuse()


class Predictor(nn.Module):
    def __init__(self, in_channels_list, num_anchors, num_classes, strides):
        super().__init__()
        self.num_outputs = num_classes + 5
        self.mlp = nn.ModuleList()
        
        for in_channels, n in zip(in_channels_list, num_anchors):
            out_channels = n * self.num_outputs
            self.mlp.append(nn.Conv2d(in_channels, out_channels, 1))
            
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        for m, n, s in zip(self.mlp, num_anchors, strides):
            b = m.bias.detach().view(n, -1)
            b[:, 4] += math.log(8 / (416 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (num_classes - 0.99))
            m.bias = nn.Parameter(b.view(-1))
            
    def forward(self, x):
        N = x[0].shape[0]
        L = self.num_outputs
        preds = []
        for i in range(len(x)):
            h, w = x[i].shape[-2:]
            pred = self.mlp[i](x[i])
            pred = pred.permute(0, 2, 3, 1).reshape(N, h, w, -1, L)
            preds.append(pred)
        return preds
    
    