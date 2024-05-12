import time

import numpy as np
import copy
import torch
from numba import jit
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast as autocast


from utils.help_utils import *
class Eval:
    def __init__(self,eval_matrix,  que = None,  scale =[110, 110, 1500, 20000]):
        self.threshold=eval_matrix['threshold']
        self.min_int = eval_matrix['min_int']
        self.limited_x = eval_matrix['limited_x']
        self.limited_x[0] += eval_matrix['padding']
        self.limited_x[1] -= eval_matrix['padding']
        self.limited_y = eval_matrix['limited_y']
        self.limited_y[0] += eval_matrix['padding']
        self.limited_y[1] -= eval_matrix['padding']
        self.tolerance = eval_matrix['tolerance']
        self.tolerance_ax = eval_matrix['tolerance_ax']
        self.batch_size = eval_matrix['batch_size']
        self.candi_thre =eval_matrix['candi_thre']
        self.post_que = que
        self.scale = scale


    def predlist(self,P, xyzi_est, start, start_coord=[0, 0]):
        xyzi_est = cpu(xyzi_est)
        xo = xyzi_est[:,0,:,:].reshape([-1,xyzi_est.shape[-2],xyzi_est.shape[-1]])
        yo = xyzi_est[:,1, :, :].reshape([-1,xyzi_est.shape[-2],xyzi_est.shape[-1]])
        zo = xyzi_est[:,2,:,:].reshape([-1,xyzi_est.shape[-2],xyzi_est.shape[-1]])
        ints = xyzi_est[:,3,:,:].reshape([-1,xyzi_est.shape[-2],xyzi_est.shape[-1]])


        p_nms =cpu(P[:,0])


        sample = np.where(p_nms > self.threshold, 1, 0)

        pred_list = {}
        num = 0

        for i in range(len(xo)):
            pos = np.nonzero(sample[i])  # get the deterministic pixel position
            for j in range(len(pos[0])):
                x_tmp = (0.5 + pos[1][j] + float(start_coord[0]) + xo[i,pos[0][j], pos[1][j]]) * self.scale[0]
                y_tmp = (0.5 + pos[0][j] + float(start_coord[1]) + yo[i,pos[0][j], pos[1][j]]) * self.scale[1]
                if x_tmp < self.limited_x[0] or x_tmp > self.limited_x[1] :
                    continue
                if y_tmp < self.limited_y[0]  or y_tmp > self.limited_y[1]:
                    continue
                k =i+start+1
                if k not in pred_list:
                    pred_list[k] = []
                num += 1
                pred_list[k].append([k,x_tmp,
                                  y_tmp,
                                  zo[i,pos[0][j], pos[1][j]]  * self.scale[2],
                                  ints[i,pos[0][j], pos[1][j]] * self.scale[3],
                                  float(p_nms[i,pos[0][j], pos[1][j]])])



        return pred_list, num




    def ShowRecovery3D(self,match):
        # define a figure for 3D scatter plot
        ax = plt.axes(projection='3d')

        # plot boolean recoveries in 3D
        ax.scatter(match[:, 0], match[:, 1], match[:, 2], c='b', marker='o', label='GT', depthshade=False)
        ax.scatter(match[:, 4], match[:, 5], match[:, 6], c='r', marker='^', label='Rec', depthshade=False)

        # add labels and and legend
        ax.set_xlabel('X [nm]')
        ax.set_ylabel('Y [nm]')
        ax.set_zlabel('Z [nm]')
        plt.legend()

    


    def inferlist_eval(self, pred_res):
        pred_list = {}
        num = 0

        for i in range(len(pred_res['Prob'])):
            tmp, t_num= self.predlist(P=pred_res["Prob"][i], xyzi_est=pred_res["preds"][i],
                                       start=pred_res["index"][i], start_coord=pred_res["coord"][i])
            pred_list = {**pred_list, **tmp}
            num += t_num


        print('\nevaluation on {0} images, predict: {1}'.format(len(pred_list), num))


        return pred_list

