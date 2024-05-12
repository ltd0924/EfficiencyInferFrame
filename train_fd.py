from utils.parameter_setting import *
from PSFLocModel import *
from utils.visual_utils import *
from utils.local_tifffile import *
import sys
import datetime
import torch

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.allow_tf32 = True
# data = torch.randn([10, 64, 128, 128], dtype=torch.float, device='cuda', requires_grad=True)
# net = torch.nn.Conv2d(64, 4, kernel_size=[1, 1], padding=[0, 0], stride=[1, 1], dilation=[1, 1], groups=1)
# net = net.cuda().float()
# out = net(data)
# out.backward(torch.randn_like(out))
# torch.cuda.synchronize()

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':


    psf_type = 'Astigmatism'

    settings = parameters_set2()

    log_path = settings['data_params']['valid_data_path']
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + datetime.datetime.now().strftime('%Y-%m-%d') + "_" + psf_type + "_" + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)
    setup_seed(15)

    psfmodel = PSFLocModel(settings, type=psf_type, fd= False)


    ShowSamplePSF(psf_pars=settings['psf_params'], type=psf_type)
    ShowTrainImg(image_num=4, data_params=settings['data_params'], camera_params=settings['camera_params'],
                 psf_params=settings['psf_params'], type=psf_type)

    psfmodel.train(True)