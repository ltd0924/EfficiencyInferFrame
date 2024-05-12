import numpy as np
from tqdm import tqdm

from utils.dataGenerator import *
from utils.parameter_setting import *
import pandas as pd
import tifffile as tiff
import math
from utils.help_utils import *

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def gene_random_data(settings, name, type):
    DataGen = DataGenerator(settings['data_params'], settings['camera_params'], settings['psf_params'], type=type)
    validation_generator, validNum = DataGen.genValidData_local()
    eval_imgs = []
    # sr = 0
    for j, (xemit, yemit, z, S, Nphotons, s_mask, gt) in tqdm(enumerate(validation_generator)):
        imgs_sim = DataGen.simulatedImg(S, xemit, yemit, z, Nphotons)
        for x in imgs_sim:
            eval_imgs.append(cpu(x))

    tiff.imwrite("../random_data/" + name+'.tif', eval_imgs)
    del eval_imgs
    # print("\n"+name+":"+str(sr))

    i = 1
    with open("../random_data/" + name+".csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['frame', 'xnano', 'ynano', 'znano', 'intensity'])
        for j, (xemit, yemit, z, S, Nphotons, s_mask, gt) in enumerate(validation_generator):
            gtt = cpu(torch.squeeze(gt))
            for pred_list in gtt:
                for p in pred_list:
                    if p[0] == 0:
                        continue
                    csvwriter.writerow([str(i), str(p[0] * settings['psf_params']['pixel_size_x']),
                                        str(p[1] * settings['psf_params']['pixel_size_y']),
                                        str(p[2] * settings['psf_params']['z_scale']),
                                        str(p[3] * settings['psf_params']['ph_scale'])])
                i = i + 1

def read_csv(name, x_size, y_size, p_x, p_y, ph_scale):
    preds = pd.read_csv(name, header=0).values
    maxframe = int(preds[-1][0])
    index = 1
    X_os = []
    Y_os = []
    Z = []
    I = []
    S = np.zeros((maxframe, x_size, y_size))
    x_tmp = []
    y_tmp = []
    z_tmp = []
    i_tmp = []
    for k in preds:
        xx = int(math.ceil(k[1]/p_x))
        yy = int(math.ceil(k[2]/p_y))
        k_int = int(k[0])
        S[k_int-1, xx-1, yy-1] = 1
        if k[0] != index:
            X_os.append(np.array(x_tmp))
            Y_os.append(np.array(y_tmp))
            Z.append(np.array(z_tmp))
            I.append(np.array(i_tmp))
            index = k[0]
            x_tmp = []
            y_tmp = []
            z_tmp = []
            i_tmp = []
        x_tmp.append((k[1] - xx*p_x+50) / p_x)
        y_tmp.append((k[2] - yy*p_x+50) / p_x)
        z_tmp.append(k[3])
        i_tmp.append(k[4]/ph_scale)
    X_os.append(np.array(x_tmp))
    Y_os.append(np.array(y_tmp))
    Z.append(np.array(z_tmp))
    I.append(np.array(i_tmp))
    return X_os, Y_os, Z, I, S


if __name__ == '__main__':
    psf_type = 'Astigmatism'
    setting = parameters_set1()
    gene_random_data(setting, psf_type + "_1w_local", psf_type)

    # backg = 127.45
    # factor = 174.55
    # offset = 133.39
    # setup_params = parameters_set('DilatedLoc', backg=backg, factor=factor, offset=offset)
    # X_os, Y_os, Z, I, S = read_csv("S:/Users/Fei_Yue/DilatedLoc-main_v5/simulate_data_20230406/demo3_para/demo3_activations_yx.csv",
    #                                x_size=64, y_size=64, p_x=110, p_y=110, ph_scale=setup_params['psf_params']['ph_scale'])
    # Datagene = DataGenerator(setup_params['data_params'], setup_params['camera_params'], setup_params['psf_params'], 'DMO')
    # imgs = []
    # index = 0
    # for i in range(S.shape[0]):
    #     if (np.sum(S[i]) == 0):  # 如果当前帧没有单分子点
    #         img = torch.zeros((64, 64)).cuda()
    #         imgs.append(cpu(Datagene.sim_noise(img)))
    #     else:
    #         S_tensor = gpu(S[i])
    #         X_os_tensor = gpu(X_os[index])
    #         Y_os_tensor = gpu(Y_os[index])
    #         Z_tensor = gpu(Z[index])
    #         I_tensor = gpu(I[index])
    #         imgs.append(cpu(torch.squeeze(Datagene.simulatedImg(S_tensor, X_os_tensor, Y_os_tensor, Z_tensor, I_tensor))))
    #
    #         index = index + 1
    # # X_os = torch.tensor(X_os)
    # # Y_os = torch.tensor(Y_os)
    # # Z = torch.tensor(Z)
    # # I = torch.tensor(I)
    # # S = torch.tensor(S)
    # # for x, y, z, i, s in zip(X_os, Y_os, Z, I, S):
    # #     img.append(Datagene.simulatedImg(s, x, y, z, i))
    # tiff.imwrite('S:/Users/Fei_Yue/DilatedLoc-main_v5/simulate_data_20230406/demo4_para/demo4_para_yx.tif', imgs)
    #



