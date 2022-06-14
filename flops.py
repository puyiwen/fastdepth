from enum import Flag
from ptflops import get_model_complexity_info
#from backbones import get_model
import argparse
import models
from imagenet.mobilenetv2 import MobileNetV2


if __name__ == '__main__':
    
    # net = models.MobileNetV2SkipAdd(output_size=(3,224,224))
    net = models.MobileNetV2SkipConcat(output_size=(3,224,224))
    # net = MobileNetV2()
    # net = ShuffleFaceNet()
    macs, params = get_model_complexity_info(
        net, (3, 224,224), as_strings=False,
        print_per_layer_stat=True, verbose=True)
    gmacs = macs / (1000**3)
    print("%.3f GFLOPs"%gmacs)
    print("%.3f Mparams"%(params/(1000**2)))

    if hasattr(net, "extra_gflops"):
        print("%.3f Extra-GFLOPs"%net.extra_gflops)
        print("%.3f Total-GFLOPs"%(gmacs+net.extra_gflops))