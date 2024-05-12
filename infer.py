import pickle
import numpy as np
import torch

from tqdm import trange
from torch.cuda.amp import autocast as autocast
import traceback
from multiprocessing import Process,Value,shared_memory


from utils.local_tifffile import *
from PSFLocModel import *
from utils.parameter_setting import *
from network.eval_utils import *
from utils.record_utils import *
from network.infer_utils import *
from utils.visual_utils import *


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
# os.environ['CUDA_LAUNCH_BLOCKING']='1'


def predict(device, shm, setup_params,tif_file, r_index,r_pos,w_index,w_pos):
    scale_factor =[setup_params['psf_params']['pixel_size_x'], setup_params['psf_params']['pixel_size_y'], 1, 1]  # pixel x, pixel y, z scale, ph scale
    scale_factor[3] = setup_params['psf_params']['ph_scale']
    scale_factor[2] = setup_params['psf_params']['z_scale']

    psfinfer = PsfInfer(setup_params['infer_params'], shm,tif_file, device=device,net_params=setup_params['net_params'], fd=False)

    print('start')
    time_sum = time.time()

    psfinfer.inferdata(r_index,r_pos,w_index,w_pos,scale_factor,setup_params['eval_params']['threhold'])
    time_sum = (time.time() - time_sum) * 1000
    print("time for inference is " + str(time_sum) + "ms. \n")



if __name__ == '__main__':


    setup_params = parameters_setfs_astig()

    q_stack,tif_file = get_img_info(setup_params['infer_params'],setup_params['eval_params']['batch_size'])
    
    process_list=[]
    data_list=[]
    shm_data=[]
    rindex_l = []
    rpos_l = []
    windex_l = []
    wpos_l = []
    device_count = torch.cuda.device_count()
    try:
        for i in range(device_count):
            shm_data.append(shared_memory.SharedMemory(name='share{0}'.format(i), create=True, size=int(4e9/device_count)))
            rindex_l.append(Value('i', 0))
            rpos_l.append(Value('i', 0))
            windex_l.append(Value('i', 0))
            wpos_l.append(Value('i', 0))
            process_list.append(Process(target=predict, args=("cuda:{0}".format(i),  shm_data[i],setup_params,tif_file,rindex_l[i],rpos_l[i],windex_l[i],wpos_l[i])))
            data_list.append(Process(target=sample_consumer, args=(
                q_stack, tif_file, shm_data[i], rpos_l[i], rindex_l[i], wpos_l[i], windex_l[i],)))
            data_list[i].start()

        
        for x in process_list:
            x.start()

    except Exception as e:
        print('\n', '>>>' * 20)
        print(traceback.print_exc())
    finally:
        tt = time.time()
        for x in data_list:
            x.join()
        for x in process_list:
            x.join()
        for x in shm_data:
            x.close()
            x.unlink()

    print("whole time:{0}".format(time.time()-tt))
    print("Done!")

