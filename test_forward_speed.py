import torch
import os

from utils.parameter_setting import *
from PSFLocModel import *
from utils.visual_utils import *
from utils.local_tifffile import *


torch.backends.cudnn.benchmark = True


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"


if __name__ == '__main__':

    win_size = 64 
    batch_size = 10
    model2 = LocalizationCNN_Unet_downsample_128_Unet().cuda()
    dummy_input = torch.randn(batch_size, 1,win_size,win_size).cuda()
    model2.get_parameter_number(dummy_input)




