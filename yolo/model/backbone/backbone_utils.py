from collections import OrderedDict
from yolo.model.model import MobileViT_S
from .path_aggregation_network import PathAggregationNetwork
from .utils import IntermediateLayerGetter
from .darknet import CSPDarknet

import torch
from torch import nn

class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, fpn):
        super().__init__()
        self.body = backbone
        self.fpn = fpn
        
    def forward(self, x):
        x = self.body(x)
        # x = self.fpn(x)
        return x
    
class MyAwesomeLayers(nn.ModuleDict):
    def __init__(self, model: nn.Module):
        self.out_channels_list = [96, 128, 160]
        layers = OrderedDict()
        for name, module in model.named_children():
            if name == 'avgpool':
                break
            layers[name] = module
            

        super().__init__(layers)
    def forward(self, x):
        outputs = []
        for name, module in self.items():
            if name in ['stem', 'stage1']:
                x = module(x)
            if name in ['stage2', 'stage3']:
                x = module(x)
                outputs.append(x)
            if name in ['stage4']:
                for (i, stg4module) in enumerate(module.children()):
                    if i < 1:
                        x = stg4module(x)
                    else:
                        # i=1, we want this. i=2 is Conv2d (we don't want)
                        x = stg4module(x)
                        outputs.append(x)
                        break
                    
        return outputs

    
def darknet_pan_backbone(depth_multiple, width_multiple): # 33 47 61 75
    out_channels_list = [round(width_multiple * x) for x in [64, 128, 256, 512, 1024]]
    layers = [max(round(depth_multiple * x), 1) for x in [3, 9, 9]]
    model = CSPDarknet(out_channels_list, layers) 

    return_layers = {'layer2', 'layer3', 'layer4'}

    backbone = IntermediateLayerGetter(model, return_layers)
    backbone.out_channels_list = out_channels_list[2:]
    
    depth = max(round(3 * depth_multiple), 1)
    fpn = PathAggregationNetwork(out_channels_list[2:], depth)
    return BackboneWithFPN(backbone, fpn)

def mobilevit_backbone(img_size: int):
    weights = torch.load(r'ckpts\model_best.pth.tar', weights_only=False, map_location=torch.device('cpu'))
    # print(weights['state_dict'])
    state_dict:dict = weights['state_dict']
    model = MobileViT_S(img_size=img_size, num_classes=1000)
    dir(model)
    for key in list(state_dict.keys()):
        state_dict[key.replace('module.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict)

    my_god=MyAwesomeLayers(model)
    return BackboneWithFPN(my_god, None)