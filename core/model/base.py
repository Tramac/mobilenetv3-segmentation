"""Base Model for Semantic Segmentation"""
import torch.nn as nn

from .base_model import mobilenet_v3_large_1_0, mobilenet_v3_small_1_0

__all__ = ['SegBaseModel']


class SegBaseModel(nn.Module):
    def __init__(self, nclass, aux=False, backbone='mobilenetv3_small', pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.mode = backbone.split('_')[-1]
        assert self.mode in ['large', 'small']
        if backbone == 'mobilenetv3_large':
            self.pretrained = mobilenet_v3_large_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'mobilenetv3_small':
            self.pretrained = mobilenet_v3_small_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)

        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        c4 = self.pretrained.conv5(c4)

        return c1, c2, c3, c4


if __name__ == '__main__':
    model = SegBaseModel(20, pretrained_base=False)
