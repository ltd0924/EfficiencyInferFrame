import argparse

import pandas as pd
import torch
from tqdm import tqdm
import tifffile as tiff
import numpy as np
from PSF_vector_gpu.freedipolepsf_torch import *
from utils.parameter_setting import *
from network.infer_utils import *
from utils.perlin_noise import *


class cal_crlb:
    def __init__(self,eval_params,net_path,camera_par,train_size=128, scale = [1,1,1,1]):



        self.eval_utils = Eval(eval_params, num=0, scale=scale)
        self.model = LocalizationCNN(True, False)
        self.model.cuda()
        self.loadmodel(net_path)
        self.camera_params = camera_par
        self.train_size=train_size

    def loadmodel(self,path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print(checkpoint['epoch'])
        print("=> loaded checkpoint")

    def inferdata(self,data_buffer,ground_truth):
        self.model.eval()
        res = {
            "index": [],
            "coord": [],
            "Prob": [],
            "preds": []
        }
        gt = []
        with torch.set_grad_enabled(False):
            with autocast():
                for i in range(len(data_buffer)):
                    img = data_buffer[i]
                    img = img.reshape([-1, 1, self.train_size, self.train_size])
                    img_gpu = gpu(img)

                    P, xyzi_est, _ = self.model(img_gpu)

                    res["index"].append(i)
                    res["coord"].append([0,0])
                    res["Prob"].append(cpu(P))
                    res["preds"].append(xyzi_est)
                    gt.append([ground_truth[i]])
                self.eval_utils.image_num = len(data_buffer)
                self.eval_utils.batch_size = 1
                self.eval_utils.inferlist_eval(res)
                # perf_dict, pred_lists = self.EvalM.assess(gt_list)

                perf_dict, matches = self.eval_utils.assess(gt)

        print("predict finish")
        return perf_dict, matches

    def test_local_CRLB(self,test_pos, test_photons, test_bg, psf_par, type, Nmol=25, use_train_cam=False, test_num=3000):

        print('compare with PSF CRLB at this position', test_pos)

        zernike_aber = np.array([2, -2, 0, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                                 4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                                 5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                                dtype=np.float32).reshape([21, 3])
        if type == 'TE':
            self.aber_C = np.array([
                -1.05559140e-01, 9.06649333e-03, -8.81815663e-03, 1.21348319e-02, 5.57566371e-02,
                -2.03187763e-02, 2.66371799e-03, 1.72353124e-01, -5.38016406e-03, -6.87511903e-03,
                4.77506823e-03, 9.06943125e-03, 9.55213928e-03, 1.45853790e-02, 9.67666903e-04,
                -3.04284761e-03, -2.31390327e-04, -1.22579141e-04, -3.17219939e-03, 1.40159223e-03,
                1.05907613e-02
            ])
        elif type == 'SA':
            self.aber_C = np.array([0.12148456, 0.00108541, -0.01320412, -0.0140747, -0.03182168, -0.01063529, -0.01636443,
                           -0.05272095, -0.00969356, 0.0241506, -0.0160361, -0.00260201, -0.00078228, -0.00255994,
                           0.0055783, 0.00828868, 0.04018608, 0.00207674, -0.02517455, 0.02524482, -0.01050033])
        elif type == 'AS':
            self.aber_C = np.zeros([21])
            self.aber_C[1] = 70 / psf_par['wavelength']
        else:
            self.aber_C = np.array([5.2760e+01, -0.0000e+00, -1.3000e-01, -1.3000e-01, -1.5000e-01, -5.1000e-01, 5.1000e-01,
                           -2.6921e+02, 0.0000e+00, 1.5000e+00, 1.5000e+00, -2.2000e-01, -0.0000e+00, -2.0000e-02,
                           -0.0000e+00, 0.0000e+00, -2.7070e+01, 0.0000e+00, 5.2000e-01, 5.2000e-01, -4.0000e-0])
            self.aber_C = np.array(self.aber_C) / psf_par['wavelength']

        zernike_aber[:, 2] = self.aber_C * psf_par['wavelength']

        psf_par['zernike_aber'] = zernike_aber
        # Npixels = model.dat_generator.psf_pars['psf_size']
        psf_par['Nphotons'] = test_photons * np.ones(Nmol)
        psf_par['bg']= test_bg * np.ones(Nmol)
        psf_par['xemit'] = 0 * np.ones(Nmol)
        psf_par['yemit'] = 0 * np.ones(Nmol)
        psf_par['zemit'] = 1 * np.linspace(-psf_par['z_scale'], psf_par['z_scale'], Nmol)
        psf_par['objstage'] = 0 * np.linspace(-psf_par['z_scale'], psf_par['z_scale'], Nmol)
        psf_par['otf_rescale'] = [0.5, 0.5]
        psf_par['zemit0'] = -1 * psf_par['refmed'] / psf_par['refimm'] * psf_par['initial_obj_stage']


        # instantiate the PSF model
        PSF_torch = FreeDipolePSF_torch(psf_par, req_grad=False)



        # calculate CRLB
        y_crlb, x_crlb, z_crlb, model_torch = PSF_torch.cal_crlb()
        x_crlb_np = x_crlb.detach().cpu().numpy()
        y_crlb_np = y_crlb.detach().cpu().numpy()
        z_crlb_np = z_crlb.detach().cpu().numpy()
        print('average 3D CRLB is:', np.sum(x_crlb_np ** 2 + y_crlb_np ** 2 + z_crlb_np ** 2) / PSF_torch.Nmol)

        # generate test single emitter set for CRLB comparison
        oldzemit = psf_par['zemit']
        dz = np.abs(oldzemit[2] - oldzemit[1])
        data_buffer = np.zeros([1, psf_par['Npixels'],psf_par['Npixels']])
        ground_truth = []
        frame_count = 1
        print('simulating single-emitter images for CRLB test')
        for i in range(test_num):
            PSF_torch.xemit = gpu((np.ones(Nmol) - 2 * np.random.rand(1)) * psf_par['pixel_size_x'])  # nm
            PSF_torch.yemit = gpu((np.ones(Nmol) - 2 * np.random.rand(1)) * psf_par['pixel_size_y'])  # nm
            PSF_torch.zemit = gpu(oldzemit + (np.ones(Nmol) - 2 * np.random.rand(1)) * (dz / 2 - 1))  # nm

            for j in range(Nmol):
                ground_truth.append([frame_count,
                                     cpu(PSF_torch.yemit[j]) / psf_par['pixel_size_y']+ psf_par['Npixels'] / 2  ,
                                     cpu(PSF_torch.xemit[j]) / psf_par['pixel_size_y'] + psf_par['Npixels'] / 2 ,
                                     cpu(PSF_torch.zemit[j]) / psf_par['z_scale'] + 0,
                                     psf_par['Nphotons'][j] / psf_par['ph_scale'] ])

                frame_count += 1

            if use_train_cam:
                data_model = PSF_torch.gen_psf() * PSF_torch.Nphotons
                data_tmp = cpu(self.sim_noise(data_model))
            else:
                data_model = PSF_torch.gen_psf() * PSF_torch.Nphotons + PSF_torch.bg
                data_tmp = np.random.poisson(cpu(data_model))

            data_buffer = np.concatenate((data_buffer, data_tmp), axis=0)

            print('{}{}{}{}{}{}'.format('\r', 'simulated ', (i + 1) * Nmol, '/', test_num * Nmol, ' images'), end='')

        data_buffer = data_buffer[1:]
        print('\n')
        self.train_size = psf_par['Npixels']


        print('start inferring')
        # compare network's prediction with CRLB
        perf_dict, matches = self.inferdata(data_buffer,ground_truth)

        rmse_xyz = np.zeros([3, Nmol])

        for i in range(Nmol):
            z = psf_par['zemit'][i]
            ind = np.where(((z - dz / 2) < matches[:, 2]) & (matches[:, 2] < (z + dz / 2)))
            tmp = np.squeeze(matches[ind, :])
            if tmp.shape[0]:
                rmse_xyz[0, i] = np.sqrt(np.mean(np.square(tmp[:, 0] - tmp[:, 4])))
                rmse_xyz[1, i] = np.sqrt(np.mean(np.square(tmp[:, 1] - tmp[:, 5])))
                rmse_xyz[2, i] = np.sqrt(np.mean(np.square(tmp[:, 2] - tmp[:, 6])))

        return psf_par['zemit'], x_crlb_np, y_crlb_np, z_crlb_np, rmse_xyz

    def sim_noise(self, imgs_sim):
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
            imgs_sims = imgs_sim + bg_photons

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

            imgs_sim = torch.distributions.Poisson(
                imgs_sim * self.camera_params['qe'] + self.camera_params['spurious_c']).sample()

            RN = self.camera_params['sig_read']
            zeros = torch.zeros_like(imgs_sim)
            read_out_noise = torch.distributions.Normal(zeros, zeros + RN).sample()

            imgs_sim = imgs_sim + read_out_noise

            imgs_sim = imgs_sim / self.camera_params['e_per_adu']

            #ssn = cal_snr(cpu(imgs_sims),cpu(imgs_sim))

            imgs_sim = torch.clamp(imgs_sim + self.camera_params['baseline'],
                                   min=0)
        else:
            print('wrong camera types! please choose EMCCD or sCMOS!')

        return imgs_sim


def main():
    setup_params = parameters_set1()
    net_path = setup_params['infer_params']['net_path']


    test_pos_list = [[int(51 * 0.5), int(51 * 0.5)],
                     [int(51 * 0.95), int(51 * 0.95)]]

    bg = (setup_params['camera_params']['backg'] - setup_params['camera_params']['baseline']) \
         / setup_params['camera_params']['em_gain'] * setup_params['camera_params']['e_per_adu'] \
         / setup_params['camera_params']['qe']
    mol_photons = (setup_params['data_params']['min_ph'] + 1) / 2 * setup_params['psf_params']['ph_scale']
    print('{}{}{}{}{}'.format('The average signal/background used for training are: ', int(mol_photons), '/',
                              int(bg), ' photons'))

    cc = cal_crlb(setup_params['eval_params'], net_path, setup_params['camera_params'], 52, [setup_params['psf_params']['pixel_size_x'],
                                                                                                   setup_params['psf_params']['pixel_size_y'],
                                                                                                   setup_params['psf_params']['z_scale'],
                                                                                                   setup_params['psf_params']['ph_scale']])

    zpos, x_crlb, y_crlb, z_crlb, rmse_xyz = cc.test_local_CRLB(test_pos_list[1], mol_photons, bg,
                                                                setup_params['psf_params'], 'AS',
                                                                25, True, 500)

    plt.figure(constrained_layout=True)
    plt.plot(zpos, x_crlb, 'b', zpos, y_crlb, 'g', zpos, z_crlb, 'r')
    plt.scatter(zpos, rmse_xyz[0, :], c="b", marker="o")
    plt.scatter(zpos, rmse_xyz[1, :], c="g", marker="o")
    plt.scatter(zpos, rmse_xyz[2, :], c="r", marker="o")

    plt.legend(('$CRLB_x^{1/2}$', '$CRLB_y^{1/2}$', '$CRLB_z^{1/2}$',
                '$DialtedLoc RMSE_x$', '$DialtedLoc RMSE_y$', '$DialtedLoc RMSE_z$'
                ), ncol=3,
               loc='upper center')
    plt.xlim([np.min(zpos), np.max(zpos)])
    plt.show()


if __name__ == '__main__':
    main()

    # # parameters initialization
    # setup_params = parameters_set1()
    # net_path = setup_params['train_params']['results_path']+'best_checkpoint.pth.tar'
    #
    #
    # cc = cal_crlb(setup_params['eval_params'], net_path,  setup_params['camera_params'], 51)
    # test_pos_list = [[int(51 * 0.5), int(51 * 0.5)],
    #                  [int(51 * 0.95), int(51 * 0.95)]]
    #
    # bg = (setup_params['camera_params']['backg'] - setup_params['camera_params']['baseline']) \
    #      / setup_params['camera_params']['em_gain'] * setup_params['camera_params']['e_per_adu'] \
    #      / setup_params['camera_params']['qe']
    # mol_photons = (setup_params['data_params']['min_ph'] + 1) / 2 * setup_params['psf_params']['ph_scale']
    # print('{}{}{}{}{}'.format('The average signal/background used for training are: ', int(mol_photons), '/',
    #                           int(bg), ' photons'))
    #
    # zpos, x_crlb, y_crlb, z_crlb, rmse_xyz = cc.test_local_CRLB(test_pos_list[1], mol_photons, bg, setup_params['psf_params'], 'TE',
    #                                                             25, True, 1000)
    #
    #
    # plt.figure(constrained_layout=True)
    # plt.plot(zpos, x_crlb, 'b', zpos, y_crlb, 'g', zpos, z_crlb, 'r')
    # plt.scatter(zpos, rmse_xyz[0, :], c="b", marker="o")
    # plt.scatter(zpos, rmse_xyz[1, :], c="g", marker="o")
    # plt.scatter(zpos, rmse_xyz[2, :], c="r", marker="o")
    #
    # plt.legend(('$CRLB_x^{1/2}$', '$CRLB_y^{1/2}$', '$CRLB_z^{1/2}$',
    #             '$DialtedLoc RMSE_x$', '$DialtedLoc RMSE_y$', '$DialtedLoc RMSE_z$'
    #             ), ncol=3,
    #            loc='upper center')
    # plt.xlim([np.min(zpos), np.max(zpos)])
    # plt.show()

















