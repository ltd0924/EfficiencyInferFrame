import numpy as np
import torch
import os
import scipy.io as scio
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PSF_vector_gpu.PsfSimulation import *
from utils.perlin_noise import *

# xyz and photons turned online to fourier phases dataset
class LocalizeDataset(Dataset):

    # initialization of the dataset
    def __init__(self, list_IDs, labels):
        self.list_IDs = list_IDs
        self.labels = labels

    # total number of samples in the dataset
    def __len__(self):
        return len(self.list_IDs)

    # sampling one example from the data
    def __getitem__(self, index):
        # select sample
        ID = self.list_IDs[index]
        # associated number of photons
        dict = self.labels[ID]
        Nphotons = dict['ints']
        xemit= dict['x_os']
        yemit= dict['y_os']
        z= dict['z']
        locs = dict['locs']
        s_mask = dict['s_mask']
        gt = dict['gt']
        return xemit,yemit,z,locs, Nphotons,s_mask,gt






'''
generate evaluation dataset
'''

class DataGenerator:

    def __init__(self,data_params,camera_params,psf_params, type):
        self.path_train = data_params['valid_data_path']
        self.batch_size = data_params['batch_size']
        self.valid_size = data_params['nvalid']
        self.num_particles = data_params['num_particles']
        self.train_size_x = data_params['size_x']
        self.train_size_y = data_params['size_y']
        self.z_scale = psf_params['z_scale']
        self.min_ph = data_params['min_ph']
        self.nvalid_batches = int(data_params['nvalid']/data_params['batch_size'])
        self.PSFsimu = PsfSimulation(psf_params, type)
        self.camera_params = camera_params

    def genValidData(self):
        # random seed for repeatability

        if not (os.path.isdir(self.path_train)):
            os.mkdir(self.path_train)

        # print status
        print('=' * 50)
        print('Sampling examples for validation')
        print('=' * 50)

        labels_dict = {}

        # sample validation examples
        for i in range(self.nvalid_batches):
            # sample a training example
            while (True):
                S, X_os, Y_os, Z, I, s_mask, gt, _ = self.generateBatch(self.batch_size, val=True, local_context=False)
                X_os = torch.squeeze(X_os)
                Y_os = torch.squeeze(Y_os)
                Z = torch.squeeze(Z)
                I = torch.squeeze(I)
                if S.sum() > 0 and X_os.size():
                    labels_dict[str(i)] = {'locs': S, 'x_os': X_os, 'y_os': Y_os, 'z': Z, 'ints': I,
                                                            'gt': gt, 's_mask': s_mask}
                    break
            # print number of example
            # print('Validation Example [%d / %d]' % (i + 1, self.nvalid_batches))

        # save all xyz's dictionary as a pickle file
        path_labels = self.path_train + 'validLabels.pickle'
        self.labels = labels_dict
        with open(path_labels, 'wb') as handle:
            pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.getValidDataset()

    def genValidData_local(self):
        # random seed for repeatability

        if not (os.path.isdir(self.path_train)):
            os.mkdir(self.path_train)

        # print status
        print('=' * 50)
        print('Sampling examples for validation')
        print('=' * 50)

        labels_dict = {}

        # sample validation examples
        for i in range(self.nvalid_batches):
            # sample a training example
            while (True):
                S, X_os, Y_os, Z, I, s_mask, gt = self.generateBatch_val(self.batch_size, val=True, local_context=True)
                X_os = torch.squeeze(X_os)
                Y_os = torch.squeeze(Y_os)
                Z = torch.squeeze(Z)
                I = torch.squeeze(I)
                if S.sum() > 0 and X_os.size():
                    labels_dict[str(i)] = {'locs': S, 'x_os': X_os, 'y_os': Y_os, 'z': Z, 'ints': I,
                                                            'gt': gt, 's_mask': s_mask}
                    break
            # print number of example
            # print('Validation Example [%d / %d]' % (i + 1, self.nvalid_batches))

        # save all xyz's dictionary as a pickle file
        path_labels = self.path_train + 'validLabels.pickle'
        self.labels = labels_dict
        with open(path_labels, 'wb') as handle:
            pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.getValidDataset()

    def readValidFile(self):
        with open( self.path_train +'validLabels.pickle', 'rb') as handle:
            self.labels = pickle.load(handle)
        return self.getValidDataset()

    def getValidDataset(self):
        ind_all = np.arange(0, self.nvalid_batches, 1)
        list_all = ind_all.tolist()
        list_IDs = [str(i) for i in list_all]

        # instantiate the data class and create a data loader for validation
        validation_set = LocalizeDataset(list_IDs, self.labels)
        validation_generator = DataLoader(validation_set,shuffle=False)
        return validation_generator,self.nvalid_batches

    def simulatedImg(self, S, X, Y, Z, I):
        img = self.PSFsimu.sim_psf_vector(X, Y, Z)
        img /= img.sum(-1).sum(-1)[:, None, None]  # photon normalization
        I = torch.squeeze(I)  # 仅在用activation模拟数据时注释
        I = torch.reshape(I, (I.size(0), 1, 1))
        S = torch.reshape(S, (-1, self.train_size_x, self.train_size_y))
        img *= I
        img_sim = self.PSFsimu.place_psfs(img, S)
        imgs_sim = img_sim.reshape([-1, 1, self.train_size_x, self.train_size_y])

        imgs_sim = self.sim_noise(imgs_sim)

        return imgs_sim

    def generateBatch(self, size, val, local_context=False):
        # if we're testing then seed the random generator
        # if self.seed is not None:
        #     np.random.seed(self.seed)
        # randomly vary the number of emitters
        #fy
        if val:
            M = np.ones([1, self.train_size_y, self.train_size_x])
            M[0, int(self.camera_params['margin_empty'] * self.train_size_y):int((1-self.camera_params['margin_empty']) * self.train_size_y), int(self.camera_params['margin_empty'] * self.train_size_x):int((1-self.camera_params['margin_empty']) * self.train_size_x)] += 9
        else:
            M = np.zeros([1, self.train_size_y, self.train_size_x])
            M[0, int(self.camera_params['margin_empty'] * self.train_size_y):int((1-self.camera_params['margin_empty']) * self.train_size_y),
            int(self.camera_params['margin_empty'] * self.train_size_x):int((1-self.camera_params['margin_empty']) * self.train_size_x)] += 1
        M = M / M.sum() * self.num_particles

        blink_p = torch.cuda.FloatTensor(M)
        blink_p = blink_p.reshape(1, 1, blink_p.shape[-2], blink_p.shape[-1]).repeat_interleave(size, 0)
        while True:
            locs = torch.distributions.Binomial(1, blink_p).sample().to('cuda')  # p * M.sum() = density
            u = 0
            for i in range(self.batch_size):
                if locs[i].sum():
                    u = u + 1
            if u == self.batch_size:
                break

        if local_context:
            surv_p = self.camera_params['surv_p']
            a11 = 1 - (1 - blink_p) * (1 - surv_p)
            locs2 = torch.distributions.Binomial(1, (1 - locs) * blink_p + locs * a11).sample().to('cuda')
            locs3 = torch.distributions.Binomial(1, (1 - locs2) * blink_p + locs2 * a11).sample().to('cuda')
            locs = torch.cat([locs, locs2, locs3], 1)


        zeros = torch.zeros_like(locs).to('cuda')
        # z position follows a uniform distribution with predefined range
        z = torch.distributions.Uniform(zeros - 1,
                                        zeros + 1).sample().to('cuda')
        # xy offset follow uniform distribution
        x_os = torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample().to('cuda')
        y_os = torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample().to('cuda')

        ints = torch.distributions.Uniform(torch.zeros_like(locs) + self.min_ph,
                                           torch.ones_like(locs)).sample().to('cuda')
        z *= locs

        x_os *= locs
        y_os *= locs

        ints *= locs

        xyzit = torch.cat([x_os[:, :, None], y_os[:, :, None], z[:, :, None], ints[:, :, None]], 2)
        xyzi = torch.cat([x_os.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          y_os.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          z.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          ints.reshape([-1, 1, self.train_size_x, self.train_size_y])], 1)

        X_os, Y_os, Z, I = self.transform_offsets(self.z_scale, locs.reshape([-1, self.train_size_x, self.train_size_y]), xyzi)
        xyzi_gt = torch.zeros([size, 0, 4]).type(torch.cuda.FloatTensor)
        s_mask = torch.zeros([size, 0]).type(torch.cuda.FloatTensor)

        xyzit = xyzit[:, 1] if local_context else xyzit[:, 0]
        # get all molecules' discrete pixel positions [number_in_batch, row, column]
        #S = S.reshape([-1, self.train_size_x, self.train_size_y])
        S = locs  # （n, 3, size_x, size_y）
        S = S[:, 1] if local_context else S[:, 0]  # （n, 3, size_x, size_y）
        s_inds = tuple(S.nonzero().transpose(1, 0))
        # get these molecules' sub-pixel xy offsets, z positions and photons
        xyzi_true = xyzit[s_inds[0], :, s_inds[1], s_inds[2]]
        # get the xy continuous pixel positions
        xyzi_true[:, 0] += s_inds[2].type(torch.cuda.FloatTensor) + 0.5
        xyzi_true[:, 1] += s_inds[1].type(torch.cuda.FloatTensor) + 0.5
        # return the gt numbers of molecules on each training images of this batch
        # (if local_context, return the number of molecules on the middle frame)
        s_counts = torch.unique_consecutive(s_inds[0], return_counts=True)[1]
        s_max = s_counts.max()
        # for each training images of this batch, build a molecule list with length=s_max
        xyzi_gt_curr = torch.cuda.FloatTensor(size, s_max, 4).fill_(0)
        s_mask_curr = torch.cuda.FloatTensor(size, s_max).fill_(0)
        s_arr = torch.cat([torch.arange(c) for c in s_counts], dim=0)
        # put the gt in the molecule list, with remaining=0
        xyzi_gt_curr[s_inds[0], s_arr] = xyzi_true
        s_mask_curr[s_inds[0], s_arr] = 1

        xyzi_gt = torch.cat([xyzi_gt, xyzi_gt_curr], 1)
        s_mask = torch.cat([s_mask, s_mask_curr], 1)

        locs = locs.reshape([-1, self.train_size_x, self.train_size_y])
        return locs, X_os, Y_os, Z, I, s_mask, xyzi_gt, S

    def generateBatch_val(self, size, val, local_context=False):
        # if we're testing then seed the random generator
        # if self.seed is not None:
        #     np.random.seed(self.seed)
        # randomly vary the number of emitters
        #fy
        if val:
            M = np.ones([1, self.train_size_y, self.train_size_x])
            M[0, int(self.camera_params['margin_empty'] * self.train_size_y):int((1-self.camera_params['margin_empty']) * self.train_size_y), int(self.camera_params['margin_empty'] * self.train_size_x):int((1-self.camera_params['margin_empty']) * self.train_size_x)] += 9
        else:
            M = np.zeros([1, self.train_size_y, self.train_size_x])
            M[0, int(self.camera_params['margin_empty'] * self.train_size_y):int((1-self.camera_params['margin_empty']) * self.train_size_y),
            int(self.camera_params['margin_empty'] * self.train_size_x):int((1-self.camera_params['margin_empty']) * self.train_size_x)] += 1
        M = M / M.sum() * self.num_particles

        blink_p = torch.cuda.FloatTensor(M)
        blink_p = blink_p.reshape(1, 1, blink_p.shape[-2], blink_p.shape[-1]).repeat_interleave(size, 0)
        while True:
            locs = torch.distributions.Binomial(1, blink_p).sample().to('cuda')  # p * M.sum() = density
            u = 0
            for i in range(self.batch_size):
                if locs[i].sum():
                    u = u + 1
            if u == self.batch_size:
                break

        zeros = torch.zeros_like(locs).to('cuda')
        # z position follows a uniform distribution with predefined range
        z = torch.distributions.Uniform(zeros - 1,
                                        zeros + 1).sample().to('cuda')
        # xy offset follow uniform distribution
        x_os = torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample().to('cuda')
        y_os = torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample().to('cuda')

        if local_context:
            surv_p = self.camera_params['surv_p']
            a11 = 1 - (1 - blink_p) * (1 - surv_p)
            locs2 = torch.distributions.Binomial(1, (1 - locs) * blink_p + locs * a11).sample().to('cuda')
            locs3 = torch.distributions.Binomial(1, (1 - locs2) * blink_p + locs2 * a11).sample().to('cuda')
            locs = torch.cat([locs, locs2, locs3], 1)
            x_os = x_os.repeat_interleave(3, 1)  # 直接复制 == 连续三帧的偏移量相同，但坐标不同 --> 全局坐标不同
            y_os = y_os.repeat_interleave(3, 1)
            z = z.repeat_interleave(3, 1)

        ints = torch.distributions.Uniform(torch.zeros_like(locs) + self.min_ph,
                                           torch.ones_like(locs)).sample().to('cuda')
        z *= locs

        x_os *= locs
        y_os *= locs

        ints *= locs

        # xyzit = torch.cat([x_os[:, :, None], y_os[:, :, None], z[:, :, None], ints[:, :, None]], 2)
        xyzi = torch.cat([x_os.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          y_os.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          z.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          ints.reshape([-1, 1, self.train_size_x, self.train_size_y])], 1)

        X_os, Y_os, Z, I = self.transform_offsets(self.z_scale, locs.reshape([-1, self.train_size_x, self.train_size_y]), xyzi)
        xyzi_gt = torch.zeros([size*3, 0, 4]).type(torch.cuda.FloatTensor)
        s_mask = torch.zeros([size*3, 0]).type(torch.cuda.FloatTensor)

        # xyzit = xyzit[:, 1] if local_context else xyzit[:, 0]
        # get all molecules' discrete pixel positions [number_in_batch, row, column]
        #S = S.reshape([-1, self.train_size_x, self.train_size_y])
        S = locs.reshape([-1, self.train_size_x, self.train_size_y])  # （n, 3, size_x, size_y）
        # S = S[:, 1] if local_context else S[:, 0]  # （n, 3, size_x, size_y）
        s_inds = tuple(S.nonzero().transpose(1, 0))
        # get these molecules' sub-pixel xy offsets, z positions and photons
        xyzi_true = xyzi[s_inds[0], :, s_inds[1], s_inds[2]]
        # get the xy continuous pixel positions
        xyzi_true[:, 0] += s_inds[2].type(torch.cuda.FloatTensor) + 0.5
        xyzi_true[:, 1] += s_inds[1].type(torch.cuda.FloatTensor) + 0.5
        # return the gt numbers of molecules on each training images of this batch
        # (if local_context, return the number of molecules on the middle frame)
        s_counts = torch.unique_consecutive(s_inds[0], return_counts=True)[1]
        s_max = s_counts.max()
        # for each training images of this batch, build a molecule list with length=s_max
        xyzi_gt_curr = torch.cuda.FloatTensor(size*3, s_max, 4).fill_(0)
        s_mask_curr = torch.cuda.FloatTensor(size*3, s_max).fill_(0)
        s_arr = torch.cat([torch.arange(c) for c in s_counts], dim=0)
        # put the gt in the molecule list, with remaining=0
        xyzi_gt_curr[s_inds[0], s_arr] = xyzi_true
        s_mask_curr[s_inds[0], s_arr] = 1

        xyzi_gt = torch.cat([xyzi_gt, xyzi_gt_curr], 1)
        s_mask = torch.cat([s_mask, s_mask_curr], 1)

        locs = locs.reshape([-1, self.train_size_x, self.train_size_y])
        return locs, X_os, Y_os, Z, I, s_mask, xyzi_gt

    def sim_noise(self, imgs_sim, add_noise=True):
        if self.camera_params['camera'] == 'EMCCD':
            bg_photons = (self.camera_params['backg'] - self.camera_params['baseline']) \
                         / self.camera_params['em_gain'] * self.camera_params['e_per_adu'] \
                         / self.camera_params['qe']
            if bg_photons < 0:
                print('converted bg_photons is less than 0, please check the parameters setting!')

            if self.camera_params['perlin_noise']:
                size_x, size_y = imgs_sim.shape[-2], imgs_sim.shape[-1]
                self.PN_res = self.camera_params['pn_res']
                self.PN_octaves_num = 1
                space_range_x = size_x / self.PN_res
                space_range_y = size_y / self.PN_res
                self.PN = PerlinNoiseFactory(dimension=2, octaves=self.PN_octaves_num,
                                             tile=(space_range_x, space_range_y),
                                             unbias=True)
                PN_tmp_map = np.zeros([size_x, size_y])
                for x in range(size_x):
                    for y in range(size_y):
                        cal_PN_tmp = self.PN(x / self.PN_res, y / self.PN_res)
                        PN_tmp_map[x, y] = cal_PN_tmp
                PN_noise = PN_tmp_map * bg_photons * self.camera_params['pn_factor']
                bg_photons += PN_noise
                bg_photons = gpu(bg_photons)

            imgs_sim += bg_photons

            if add_noise:
                imgs_sim = torch.distributions.Poisson(
                    imgs_sim * self.camera_params['qe'] + self.camera_params['spurious_c']).sample()

                imgs_sim = torch.distributions.Gamma(imgs_sim, 1 / self.camera_params['em_gain']).sample()

                RN = self.camera_params['sig_read']
                zeros = torch.zeros_like(imgs_sim)
                read_out_noise = torch.distributions.Normal(zeros, zeros + RN).sample()

                imgs_sim = imgs_sim + read_out_noise
                imgs_sim = torch.clamp(imgs_sim / self.camera_params['e_per_adu'] + self.camera_params['baseline'],
                                       min=0)

        elif self.camera_params['camera'] == 'sCMOS':
            bg_photons = (self.camera_params['backg'] - self.camera_params['baseline']) \
                         * self.camera_params['e_per_adu'] / self.camera_params['qe']
            if bg_photons < 0:
                print('converted bg_photons is less than 0, please check the parameters setting!')

            if self.camera_params['perlin_noise']:
                size_x, size_y = imgs_sim.shape[-2], imgs_sim.shape[-1]
                self.PN_res = self.camera_params['pn_res']
                self.PN_octaves_num = 1
                space_range_x = size_x / self.PN_res
                space_range_y = size_y / self.PN_res
                self.PN = PerlinNoiseFactory(dimension=2, octaves=self.PN_octaves_num,
                                             tile=(space_range_x, space_range_y),
                                             unbias=True)
                PN_tmp_map = np.zeros([size_x, size_y])
                for x in range(size_x):
                    for y in range(size_y):
                        cal_PN_tmp = self.PN(x / self.PN_res, y / self.PN_res)
                        PN_tmp_map[x, y] = cal_PN_tmp
                PN_noise = PN_tmp_map * bg_photons * self.camera_params['pn_factor']
                bg_photons += PN_noise
                bg_photons = gpu(bg_photons)

            imgs_sim += bg_photons

            if add_noise:
                imgs_sim = torch.distributions.Poisson(
                    imgs_sim * self.camera_params['qe'] + self.camera_params['spurious_c']).sample()

                RN = self.camera_params['sig_read']
                zeros = torch.zeros_like(imgs_sim)
                read_out_noise = torch.distributions.Normal(zeros, zeros + RN).sample()

                imgs_sim = imgs_sim + read_out_noise
                imgs_sim = torch.clamp(imgs_sim / self.camera_params['e_per_adu'] + self.camera_params['baseline'],
                                       min=0)
        else:
            print('wrong camera types! please choose EMCCD or sCMOS!')
        return imgs_sim

    def transform_offsets(self,z, S, XYZI):
        n_samples = S.shape[0] // XYZI.shape[0]
        XYZI_rep = XYZI.repeat_interleave(n_samples, 0)

        s_inds = tuple(S.nonzero().transpose(1, 0))
        x_os_vals = (XYZI_rep[:, 0][s_inds])[:, None, None]
        y_os_vals = (XYZI_rep[:, 1][s_inds])[:, None, None]
        z_vals = z * XYZI_rep[:, 2][s_inds][:, None, None]
        i_vals = (XYZI_rep[:, 3][s_inds])[:, None, None]

        return x_os_vals, y_os_vals, z_vals, i_vals

    def genTestData(self):
        # don't know what to do

        if not (os.path.isdir(self.path_train)):
            os.mkdir(self.path_train)

        # print status
        print('=' * 50)
        print('Sampling examples for validation')
        print('=' * 50)

        labels_dict = {}
        S, X_os, Y_os, Z, I, s_mask, gt = [], [], [], [], [], [], []
        # sample validation examples
        for i in range(self.nvalid_batches):
            # sample a training example
            S_, X_os_, Y_os_, Z_, I_, s_mask_, gt_, _ = self.generateBatch(self.batch_size, val=True)
            X_os_ = torch.squeeze(X_os_)
            Y_os_ = torch.squeeze(Y_os_)
            Z_ = torch.squeeze(Z_)
            I_ = torch.squeeze(I_)

        return S_, X_os_, Y_os_, Z_, I_, s_mask_, gt_

