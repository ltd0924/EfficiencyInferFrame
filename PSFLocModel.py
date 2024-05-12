import collections
import time
import os
import pickle

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.optim import NAdam # change
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from pytorchtools import EarlyStopping

from network.loss_utils import *
from utils.dataGenerator import DataGenerator
from PSF_vector_gpu.PsfSimulation import *
from network.DilatedLoc import *
from network.eval_utils import *
from utils.record_utils import *
# from torchsummary import summary

class PSFLocModel:
    def __init__(self,setup_params,type, fd = False):

        torch.backends.cudnn.benchmark = True

        self.DataGen = DataGenerator(setup_params['data_params'],setup_params['camera_params'],setup_params['psf_params'],
                                     type=type)



        if fd:
            self.model = FdDeeploc(setup_params['net_params']).to(torch.device('cuda'))
        else:
            self.model = LocalizationCNN_Unet_downsample_128_Unet()

        self.local = setup_params['net_params']['local_flag']

        self.EvalM = Eval(setup_params['eval_params'], scale=[setup_params['psf_params']['pixel_size_x'] ,
                                                              setup_params['psf_params']['pixel_size_x'],
                                                              setup_params['psf_params']['z_scale'] ,
                                                              setup_params['psf_params']['ph_scale']])

        self.net_weight = list(self.model.parameters())[:-1]
        #
        self.optimizer = NAdam(self.net_weight, lr=setup_params['train_params']['lr'], betas=(0.8, 0.8888), eps=1e-8) # change
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=setup_params['train_params']['lr_decay'])

        # loss function

        self.criterion = LossFuncs(batch_size=setup_params['data_params']['batch_size'],
                                   train_size=setup_params['data_params']['size_x'],
                                   fd = fd)

        if setup_params['data_params']['dataresume']:
            self.validation_generator, self.validNum = self.DataGen.readValidFile()
        else:
            self.validation_generator, self.validNum = self.DataGen.genValidData()

        self.start_epoch = 0
        self.endEpoch = setup_params['train_params']['max_iters']
        self.pathSave = setup_params['train_params']['results_path']
        self.interval = setup_params['train_params']['interval']
        self.batch_size = setup_params['data_params']['batch_size']
        if setup_params['train_params']['netresume']:
            self.loadConfig()

        self.recorder = {}
        self.init_recorder()


    def init_recorder(self):

        self.recorder['cost_hist'] = collections.OrderedDict([])
        self.recorder['recall'] = collections.OrderedDict([])
        self.recorder['precision'] = collections.OrderedDict([])
        self.recorder['jaccard'] = collections.OrderedDict([])
        self.recorder['rmse_lat'] = collections.OrderedDict([])
        self.recorder['rmse_ax'] = collections.OrderedDict([])
        self.recorder['rmse_vol'] = collections.OrderedDict([])
        self.recorder['jor'] = collections.OrderedDict([])
        self.recorder['eff_lat'] = collections.OrderedDict([])
        self.recorder['eff_ax'] = collections.OrderedDict([])
        self.recorder['eff_3d'] = collections.OrderedDict([])
        self.recorder['update_time'] = collections.OrderedDict([])

    def train(self, printmodel=False):

        if printmodel:
            print('LocalizationCNN architecture')
            print('=' * 50)
            print(self.model)
            print("number of parameters: ", sum(param.numel() for param in self.model.parameters()))

        print('start training!')
        # cur = self.DataGen.num_particles

        while self.start_epoch < self.endEpoch:
            tot_cost = []
            tt = time.time()
            for _ in range(0, self.interval):
                S, X, Y, Z, I, s_mask, xyzi_gt, locs = self.DataGen.generateBatch(self.batch_size, val=False, local_context=self.local)
                imgs_sim = self.DataGen.simulatedImg(S, X, Y, Z, I)
                if self.local:
                    imgs_sim = imgs_sim.reshape([self.batch_size, 3, imgs_sim.shape[-2], imgs_sim.shape[-1]])

                p, xyzi_est, xyzi_sig = self.model(imgs_sim)

                loss_total = self.criterion.final_loss(p, xyzi_est, xyzi_sig, xyzi_gt, s_mask, locs)

                self.optimizer.zero_grad()

                loss_total.backward()

                # update the network and the optimizer state
                self.optimizer.step()
                self.scheduler.step()

                tot_cost.append(cpu(loss_total))

            self.start_epoch += 1
            # self.DataGen.num_particles = min(int(0.5*self.start_epoch + cur),60)
            print(f"Epoch{self.start_epoch}/{self.endEpoch}")
            self.recorder['cost_hist'][self.start_epoch] = np.mean(tot_cost)
            self.recorder['update_time'][self.start_epoch] = (time.time() - tt) * 1000 / self.interval
            self.evaluation()
            self.saveStatus()
            self.printResult()
        print('training finished!')

    def evaluation(self):
        # enable interactive plotting throughout iterations

        self.model.eval()
        gt_list = []
        res = {
            "index": [],
            "Prob": [],
            "preds": [],
            "coord": []
        }
        with torch.set_grad_enabled(False):
            for batch_ind, (xemit, yemit, z, S, Nphotons, s_mask, gt) in enumerate(
                    self.validation_generator):
                gt = torch.squeeze(gt)
                imgs_sim = self.DataGen.simulatedImg(S,xemit,yemit,z,Nphotons)
                P, xyzi_est,_ = self.model(imgs_sim)


                res["index"].append(batch_ind * self.batch_size)
                res["Prob"].append(P)
                res["preds"].append(xyzi_est)
                gt_list.append(cpu(gt))
                res["coord"].append([0, 0])

                # loss = loss + loss_total
        res = self.EvalM.inferlist_eval(res)
        perf_dict,_ = assess(res, gt_list, scale=self.EvalM.scale, limit_x=self.EvalM.limited_x, limit_y= self.EvalM.limited_y)
        for k in self.recorder.keys():
            if k in perf_dict:
                self.recorder[k][self.start_epoch] = perf_dict[k]
                print('{} : {:0.3f}'.format(k, float(self.recorder[k][self.start_epoch])))
        # return loss/self.validNum


    def loadConfig(self):
        print("=> loading checkpoint to resume training")
        checkpoint = torch.load(self.pathSave + "best_checkpoint.pth.tar")
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.last_epoch = self.start_epoch
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))


    def saveStatus(self,name=None):
        if not (os.path.isdir(self.pathSave)):
            os.mkdir(self.pathSave)
        checkpoint = {"state_dict": self.model.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "epoch": self.start_epoch}
        if name:
            path_checkpoint = self.pathSave + name+'checkpoint.pth.tar'
            path_result = self.pathSave + name+'result.pkl'

        else:
            path_checkpoint = self.pathSave + 'checkpoint.pth.tar'
            path_result = self.pathSave + 'result.pkl'
        torch.save(checkpoint, path_checkpoint)
        with open(path_result, 'wb') as f:
            pickle.dump(self.recorder, f)


    def printResult(self):
        print('{}{:0.3f}'.format('JoR: ', float(self.recorder['jor'][self.start_epoch])), end='')
        print('{}{}{:0.3f}'.format(' || ', 'Eff_3d: ', self.recorder['eff_3d'][self.start_epoch]), end='')
        print('{}{}{:0.3f}'.format(' || ', 'Jaccard: ', self.recorder['jaccard'][self.start_epoch]), end='')
        print('{}{}{:0.3f}'.format(' || ', 'RMSE_lat: ', self.recorder['rmse_lat'][self.start_epoch]),end='')
        print('{}{}{:0.3f}'.format(' || ', 'RMSE_ax: ', self.recorder['rmse_ax'][self.start_epoch]), end='')
        print('{}{}{:0.3f}'.format(' || ', 'Cost: ', self.recorder['cost_hist'][self.start_epoch]), end='')
        print('{}{}{:0.3f}'.format(' || ', 'Recall: ', self.recorder['recall'][self.start_epoch]), end='')
        print('{}{}{:0.3f}'.format(' || ', 'Precision: ', self.recorder['precision'][self.start_epoch]),end='')
        print('{}{}{}'.format(' || ', 'BatchNr.: ', self.interval*(self.start_epoch)), end='')
        print('{}{}{:0.1f}{}'.format(' || ', 'Time Upd.: ', self.recorder['update_time'][self.start_epoch], ' ms '))





