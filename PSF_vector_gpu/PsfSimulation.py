import numpy as np
import torch.nn as nn
from ctypes import *
import os

from utils.help_utils import *



class PsfSimulation:

    def __init__(self,psf_pars,type):
        self.psf = cdll.LoadLibrary(os.getcwd() + '/PSF_vector_gpu/libPSF.so')

        self.psf_pars = psf_pars
        if type =='DMO-SaddlePoint':
            self.aber_C = [0.12148456,  0.00108541, -0.01320412, -0.0140747 , -0.03182168, -0.01063529, -0.01636443,
                           -0.05272095, -0.00969356,  0.0241506, -0.0160361 , -0.00260201, -0.00078228, -0.00255994,
                           0.0055783, 0.00828868,  0.04018608,  0.00207674, -0.02517455,  0.02524482, -0.01050033,
                           0.6749699,  0.57691746]  # SaddlePoint_1.4um
        elif type == 'DMO-Tetrapod':
            self.aber_C = [-1.05559140e-01,  9.06649333e-03, -8.81815663e-03,  1.21348319e-02, 5.57566371e-02,
                           -2.03187763e-02,  2.66371799e-03,  1.72353124e-01, -5.38016406e-03, -6.87511903e-03,
                           4.77506823e-03,  9.06943125e-03, 9.55213928e-03,  1.45853790e-02,  9.67666903e-04,
                           -3.04284761e-03, -2.31390327e-04, -1.22579141e-04, -3.17219939e-03,  1.40159223e-03,
                           1.05907613e-02,  5.00000000e-01,  5.00000000e-01]  # demo3-PSF-DMO-Tetropod

        elif type == 'Tetrapod_6um':
            self.aber_C = [5.2760e+01, -0.0000e+00, -1.3000e-01, -1.3000e-01, -1.5000e-01, -5.1000e-01, 5.1000e-01,
                           -2.6921e+02, 0.0000e+00, 1.5000e+00, 1.5000e+00, -2.2000e-01, -0.0000e+00, -2.0000e-02,
                           -0.0000e+00, 0.0000e+00, -2.7070e+01, 0.0000e+00, 5.2000e-01, 5.2000e-01, -4.0000e-02,
                           5.00000000e-01,  5.00000000e-01]  # PSF_6um_from_sw
        elif type == 'DMO-SaddlePoint_no_aberration':
            self.aber_C = [114.416657666606, 2.37976209525321e-14, 0.0489922420513842, 0.0489922420514215,
                           -0.0274013937789169, 0.00865544996383093, -0.00865544996384615, -155.549529180759,
                           -3.08799113938076e-14, -0.00332799767600908, -0.00332799767603682, 0.00476053999137116,
                           3.68879797100896e-15, 0.0323209261736741, -0.0124523190759750, 0.0124523190760056,
                           -11.8551314547040, -5.52177089982253e-15, -0.0111239941329578, -0.0111239941329364,
                           -0.0220720134616587, 5.00000000e-01,  5.00000000e-01] # from DMO matlab
        elif type == 'Tetrapod_6um_NPC':
            self.aber_C = [i * 680 for i in [1.22207655e-01, -2.74923490e-02, -1.06193016e-02,  8.83247573e-05,
                           -4.28437818e-02,  9.37111459e-03,  1.81628855e-02, -3.70426472e-01,
                           -1.73960628e-02, -2.17575053e-02,  3.82627330e-03, -1.56144661e-02,
                           -3.47422408e-02, -1.12367523e-03, -1.07680364e-02, -1.59017438e-02,
                           -4.43905757e-02, -1.60009040e-02, -9.22835703e-03,  5.46488574e-04,
                           -5.84247911e-03,  5.00000000e-01,  5.00000000e-01]]
            self.aber_C[21] = 0.5
            self.aber_C[22] = 0.5
        elif type == 'DMO-SaddlePoint_2um':
            self.aber_C = [7.8440e+01, 0.0000e+00, 3.0000e-02, 3.0000e-02,
                            0.0000e+00, 3.0000e-02, -3.0000e-02,-1.0811e+02,
                            -0.0000e+00, -3.0000e-02, -3.0000e-02, -0.0000e+00,
                            0.0000e+00, -1.0000e-02,-0.0000e+00, 0.0000e+00,
                            -5.6500e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
                            5.00000000e-01,  5.00000000e-01]
        elif type == 'DMO-Tetrapod_ideal':
            self.aber_C = [1.0717e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,0.0000e+00, -0.0000e+00,
                           -1.5568e+02, -0.0000e+00, -1.0000e-02, -1.0000e-02, 1.0000e-02, 0.0000e+00, -1.0000e-02,
                           -0.0000e+00, 0.0000e+00, -1.0470e+01, -0.0000e+00, -1.0000e-02, -1.0000e-02, 1.0000e-02,
                           5.00000000e-01,  5.00000000e-01]
        elif type == 'Tetrapod_6um_NPC_fs':
            self.aber_C = [-0.03578469, -0.003296, 0.04482551, -0.01907026, 0.01316782, 0.08564793, 0.02510556,
                           0.20640889, -0.00664444, -0.02939664, -0.00886532, -0.02858315, 0.00524031, -0.01406109,
                           -0.01945552, -0.02767655, -0.03012324, -0.01944731, -0.02043111, -0.01169918, -0.01958496,
                           5.00000000e-01, 5.00000000e-01]

        else:
            self.aber_C = np.zeros([23])
            self.aber_C[1] = 70
            self.aber_C[21] = 0.5
            self.aber_C[22] = 0.5

    def sim_psf_vector(self, X_os, Y_os, Z):
        """Generate the PSFs based on C code

        Parameters
        ----------
        S:
            Coordinates of the pixel which has a molecule [row, column]
        X_os:
            Sub_pixel x offset
        Y_os:
            Sub_pixel y offset
        Z:
            Z position
        """


        class sysParas_(Structure):
            _fields_ = [
                ('aberrations_', POINTER(c_float)),
                ('NA_', c_float),
                ('refmed_', c_float),
                ('refcov_', c_float),
                ('refimm_', c_float),
                ('lambdaX_', c_float),
                ('objStage0_', c_float),
                ('zemit0_', c_float),
                ('pixelSizeX_', c_float),
                ('pixelSizeY_', c_float),
                ('sizeX_', c_float),
                ('sizeY_', c_float),
                ('PupilSize_', c_float),
                ('Npupil_', c_float),
                ('zernikeModesN_', c_int),
                ('xemit_', POINTER(c_float)),
                ('yemit_', POINTER(c_float)),
                ('zemit_', POINTER(c_float)),
                ('objstage_', POINTER(c_float)),
                ('aberrationsParas_', POINTER(c_float)),
                ('psfOut_', POINTER(c_float)),
                ('aberrationOut_', POINTER(c_float)),
                ('Nmol_', c_int),
                ('showAberrationNumber_', c_int)
            ]

        sP = sysParas_()

        # In pyInterfacePSFfSimu.cPSFf(), the x y positions should be inverse to ensure the input X_os corresponds to
        # the column
        npXemit = np.array(np.squeeze(cpu(Y_os)) * self.psf_pars['pixel_size_y'], dtype=np.float32)  # nm
        npYemit = np.array(np.squeeze(cpu(X_os)) * self.psf_pars['pixel_size_x'], dtype=np.float32)
        npZemit = np.array(1 * np.squeeze(cpu(Z)), dtype=np.float32)
        npObjStage = np.array(0 * np.squeeze(cpu(Z)), dtype=np.float32)

        npSizeX = self.psf_pars['Npixels']
        npSizeY = self.psf_pars['Npixels']
        sP.Npupil_ = 64
        sP.Nmol_ = npXemit.size
        sP.NA_ = self.psf_pars['NA']
        sP.refmed_ = self.psf_pars['refmed']
        sP.refcov_ = self.psf_pars['refcov']
        sP.refimm_ = self.psf_pars['refimm']
        sP.lambdaX_ = self.psf_pars['wavelength']
        sP.objStage0_ = self.psf_pars['initial_obj_stage']
        sP.zemit0_ = -1 * sP.refmed_ / sP.refimm_ * (sP.objStage0_)
        sP.pixelSizeX_ = self.psf_pars['pixel_size_x']
        sP.pixelSizeY_ = self.psf_pars['pixel_size_y']
        sP.zernikeModesN_ = 21
        sP.sizeX_ = npSizeX
        sP.sizeY_ = npSizeY
        sP.PupilSize_ = 1.0
        sP.showAberrationNumber_ = 1
        zernikeModes = np.array([2, 2, 3, 3, 4, 3, 3, 4, 4, 5, 5, 6, 4, 4, 5, 5, 6, 6, 7, 7, 8,
                         -2, 2, -1, 1, 0, -3, 3, -2, 2, -1, 1, 0, -4, 4, -3, 3, -2, 2, 1, -1, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        dtype=np.float32)
 


        sP.aberrations_ = zernikeModes.ctypes.data_as(POINTER(c_float))

        Nmol = sP.Nmol_
        # npXYZOBuffer = np.empty((4, Nmol), dtype=np.float32, order='C')
        npPSFBuffer = np.empty((Nmol, npSizeX, npSizeY), dtype=np.float32, order='C')
        npWaberrationBuffer = np.zeros((int(sP.Npupil_), int(sP.Npupil_)), dtype=np.float32, order='C')
        aberrationsParas = np.empty((Nmol, sP.zernikeModesN_), dtype=np.float32, order='C')


        sP.xemit_ = npXemit.ctypes.data_as(POINTER(c_float))
        sP.yemit_ = npYemit.ctypes.data_as(POINTER(c_float))
        sP.zemit_ = npZemit.ctypes.data_as(POINTER(c_float))
        sP.objstage_ = npObjStage.ctypes.data_as(POINTER(c_float))

        aber_C = np.array(self.aber_C)

        # robust_training means training data will randomly generate aberrations to the PSF, as we can not
        # measure the aberration accurately, and model mismatch will cause many artifacts, we think generate
        # data with some random aberrations can help network generalize
        if self.psf_pars['robust_training']:
            for n in range(0, Nmol):
                aberrationsParas[n] = aber_C[:21] + \
                                      np.random.normal(loc=0, scale=0.01, size=21)
        else:
            for n in range(0, Nmol):
                aberrationsParas[n] = aber_C[:21]

        aberrationsParas = aberrationsParas# * sP.wavelengthX_
        sP.aberrationsParas_ = aberrationsParas.ctypes.data_as(POINTER(c_float))
        sP.psfOut_ = npPSFBuffer.ctypes.data_as(POINTER(c_float))

        sP.aberrationOut_ = npWaberrationBuffer.ctypes.data_as(POINTER(c_float))
        if sP.Nmol_ != 0:  # 为什么要判断
            genPsf = self.psf.vectorPSFF1
            genPsf.argtypes = [POINTER(sysParas_)]
            genPsf(byref(sP))
        
        if np.size(np.where(np.isnan(npPSFBuffer))) != 0:
            print('nan in the gpu psf!!!', np.where(np.isnan(npPSFBuffer)))

        # otf rescale

        npPSFBuffer = gpu(npPSFBuffer)
        for i in range(len(npPSFBuffer)):
            h = gpu(self.otf_gauss2D(shape=[5, 5], Isigmax=self.aber_C[21],
                                Isigmay=self.aber_C[22])).reshape([1, 1, 5, 5])

            tmp = nn.functional.conv2d(
                npPSFBuffer[i].reshape(1, 1, self.psf_pars['Npixels'], self.psf_pars['Npixels'])
                , h, padding=2, stride=1)
            npPSFBuffer[i] = tmp.reshape(self.psf_pars['Npixels'], self.psf_pars['Npixels'])

        return npPSFBuffer

    def otf_gauss2D(self,shape=(3, 3), Isigmax=0.5, Isigmay=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's otf rescale
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x) / (2. * Isigmax * Isigmax + 1e-6) - (y * y) / (2. * Isigmay * Isigmay + 1e-6))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def place_psfs(self, W, S):

        recs = torch.zeros_like(S)
        h, w = S.shape[1], S.shape[2]
        # s_inds: [0, 0], [y2, y1], [x2, x1]
        s_inds = tuple(S.nonzero().transpose(1, 0))
        relu = nn.ReLU()
        # r_inds: [y2, x2], [y1, x1]
        r_inds = S.nonzero()[:, 1:]  # xy坐标
        uni_inds = S.sum(0).nonzero()

        x_rl = relu(uni_inds[:, 0] - self.psf_pars['Npixels'] // 2)  # y的位置
        y_rl = relu(uni_inds[:, 1] - self.psf_pars['Npixels'] // 2)  # x的位置

        x_wl = relu(self.psf_pars['Npixels'] // 2 - uni_inds[:, 0])
        x_wh = self.psf_pars['Npixels'] - (uni_inds[:, 0] + self.psf_pars['Npixels'] // 2 - h) - 1

        y_wl = relu(self.psf_pars['Npixels'] // 2 - uni_inds[:, 1])
        y_wh = self.psf_pars['Npixels'] - (uni_inds[:, 1] + self.psf_pars['Npixels'] // 2 - w) - 1

        r_inds_r = h * r_inds[:, 0] + r_inds[:, 1]
        uni_inds_r = h * uni_inds[:, 0] + uni_inds[:, 1]

        for i in range(len(uni_inds)):
            curr_inds = torch.nonzero(r_inds_r == uni_inds_r[i])[:, 0]
            w_cut = W[curr_inds, x_wl[i]: x_wh[i], y_wl[i]: y_wh[i]]

            recs[s_inds[0][curr_inds], x_rl[i]:x_rl[i] + w_cut.shape[1], y_rl[i]:y_rl[i] + w_cut.shape[2]] += w_cut

        return recs * self.psf_pars['ph_scale']

