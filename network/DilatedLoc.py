import torch
import torch.nn as nn
from torch.nn.functional import interpolate,elu
import torch.nn.functional as func
import time
import numpy as np
import thop

def add_coord(input, field_xy, aber_map_size):
    """ concatenate global coordinate channels to the input data

    Parameters
    ----------
    input:
         tensors with shape [batchsize, channel, height, width]
    field_xy:
        [xstart, xend, ystart, yend], the global position of the input sub-area image in the big aberration map.
        should satisfies this relationship: xstart - xend + 1 = input.width
    aber_map_size:
        [sizex sizey], the size of the aberration map, sizex corresponds to column, sizey corresponds to row.
    """
    x_start = field_xy[0].float()
    y_start = field_xy[2].float()
    x_end = field_xy[1].float()
    y_end = field_xy[3].float()

    batch_size = input.size()[0]
    x_dim = input.size()[3]
    y_dim = input.size()[2]

    x_step = 1 / (aber_map_size[0] - 1)
    y_step = 1 / (aber_map_size[1] - 1)

    xx_range = torch.arange(x_start / (aber_map_size[0] - 1), x_end / (aber_map_size[0] - 1) + 1e-6, step=x_step,
                            dtype=torch.float32).repeat([y_dim, 1]).reshape([1, y_dim, x_dim])

    xx_range = xx_range.repeat_interleave(repeats=batch_size, dim=0).reshape([batch_size, 1, y_dim, x_dim])

    yy_range = torch.arange(y_start / (aber_map_size[1] - 1), y_end / (aber_map_size[1] - 1) + 1e-6, step=y_step,
                            dtype=torch.float32).repeat([x_dim, 1]).transpose(1, 0).reshape([1, y_dim, x_dim])

    yy_range = yy_range.repeat_interleave(repeats=batch_size, dim=0).reshape([batch_size, 1, y_dim, x_dim])

    xx_range = xx_range.cuda()
    yy_range = yy_range.cuda()

    ret = torch.cat([input, xx_range, yy_range], dim=1)

    return ret

class CoordConv(nn.Module):
    """ CoordConv class, add coordinate channels to the data,
    apply extra 2D convolution on the coordinate channels and add the result"""
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(CoordConv, self).__init__()
        self.conv2d_im = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding)
        self.conv2d_coord = nn.Conv2d(in_channels=2, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, input, field_xy, aber_map_size):
        y = add_coord(input, field_xy, aber_map_size)
        ret_1 = self.conv2d_im(y[:, 0:-2])
        ret_2 = self.conv2d_coord(y[:, -2:])
        return ret_1 + ret_2

class OutnetCoordConv(nn.Module):
    """output module"""
    def __init__(self, n_filters, pred_sig=False, pred_bg=False, pad=1, ker_size=3, use_coordconv=True):
        super(OutnetCoordConv, self).__init__()

        self.pred_bg = pred_bg
        self.pred_sig = pred_sig
        self.use_coordconv = use_coordconv

        if self.use_coordconv:
            self.p_out1 = CoordConv(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                    padding=pad).cuda()
            self.p_out2 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()  # fu
            self.xyzi_out1 = CoordConv(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                       padding=pad).cuda()
            self.xyzi_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()  # fu

            nn.init.kaiming_normal_(self.p_out1.conv2d_im.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.p_out1.conv2d_coord.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.p_out2.weight, mode='fan_in', nonlinearity='sigmoid')
            nn.init.constant_(self.p_out2.bias, -6.)  # -6

            nn.init.kaiming_normal_(self.xyzi_out1.conv2d_im.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.xyzi_out1.conv2d_coord.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.xyzi_out2.weight, mode='fan_in', nonlinearity='tanh')
            nn.init.zeros_(self.xyzi_out2.bias)

            if self.pred_sig:
                self.xyzis_out1 = CoordConv(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                            padding=pad).cuda()
                self.xyzis_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()

                nn.init.kaiming_normal_(self.xyzis_out1.conv2d_im.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.xyzis_out1.conv2d_coord.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='sigmoid')
                nn.init.zeros_(self.xyzis_out2.bias)

            if self.pred_bg:
                self.bg_out1 = CoordConv(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                         padding=pad).cuda()
                self.bg_out2 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()

                nn.init.kaiming_normal_(self.bg_out1.conv2d_im.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.bg_out1.conv2d_coord.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.bg_out2.weight, mode='fan_in', nonlinearity='sigmoid')
                nn.init.zeros_(self.bg_out2.bias)
        else:
            self.p_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                    padding=pad).cuda()
            self.p_out2 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()  # fu
            self.xyzi_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                       padding=pad).cuda()
            self.xyzi_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()  # fu

            nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.p_out2.weight, mode='fan_in', nonlinearity='sigmoid')
            nn.init.constant_(self.p_out2.bias, -6.)  # -6

            nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.xyzi_out2.weight, mode='fan_in', nonlinearity='tanh')
            nn.init.zeros_(self.xyzi_out2.bias)

            if self.pred_sig:
                self.xyzis_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                            padding=pad).cuda()
                self.xyzis_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()

                nn.init.kaiming_normal_(self.xyzis_out1.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='sigmoid')
                nn.init.zeros_(self.xyzis_out2.bias)

            if self.pred_bg:
                self.bg_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                         padding=pad).cuda()
                self.bg_out2 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()

                nn.init.kaiming_normal_(self.bg_out1.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.bg_out2.weight, mode='fan_in', nonlinearity='sigmoid')
                nn.init.zeros_(self.bg_out2.bias)

    def forward(self, x, field_xy, aber_map_size):

        outputs = {}

        if self.use_coordconv:
            p = func.elu(self.p_out1(x, field_xy, aber_map_size))
            outputs['p'] = self.p_out2(p)

            xyzi = func.elu(self.xyzi_out1(x, field_xy, aber_map_size))
            outputs['xyzi'] = self.xyzi_out2(xyzi)

            if self.pred_sig:
                xyzis = func.elu(self.xyzis_out1(x, field_xy, aber_map_size))
                outputs['xyzi_sig'] = self.xyzis_out2(xyzis)

            if self.pred_bg:
                bg = func.elu(self.bg_out1(x, field_xy, aber_map_size))
                outputs['bg'] = self.bg_out2(bg)
        else:
            p = func.elu(self.p_out1(x))
            outputs['p'] = self.p_out2(p)

            xyzi = func.elu(self.xyzi_out1(x))
            outputs['xyzi'] = self.xyzi_out2(xyzi)

            if self.pred_sig:
                xyzis = func.elu(self.xyzis_out1(x))
                outputs['xyzi_sig'] = self.xyzis_out2(xyzis)

            if self.pred_bg:
                bg = func.elu(self.bg_out1(x))
                outputs['bg'] = self.bg_out2(bg)

        return outputs

class UnetCoordConv(nn.Module):
    """used for frame analysis module and temporal context module"""
    def __init__(self, n_inp, n_filters=64, n_stages=5, pad=1, ker_size=3, use_coordconv=True):
        super(UnetCoordConv, self).__init__()
        curr_N = n_filters
        self.n_stages = n_stages
        self.layer_path = nn.ModuleList()
        self.use_coordconv = use_coordconv

        if self.use_coordconv:
            self.layer_path.append(
                CoordConv(in_channels=n_inp, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())
        else:
            self.layer_path.append(
                nn.Conv2d(in_channels=n_inp, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        self.layer_path.append(
            nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        for i in range(n_stages):
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=2, stride=2, padding=0).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N * 2, kernel_size=ker_size, padding=pad).cuda())
            curr_N *= 2
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        for i in range(n_stages):
            self.layer_path.append(nn.UpsamplingNearest2d(scale_factor=2).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N // 2, kernel_size=ker_size, padding=pad).cuda())

            curr_N = curr_N // 2

            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N * 2, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        for m in self.layer_path:
            if isinstance(m, CoordConv):
                nn.init.kaiming_normal_(m.conv2d_im.weight, mode='fan_in', nonlinearity='relu')  # 初始化卷积层权重
                nn.init.kaiming_normal_(m.conv2d_coord.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, xy_field, aber_map_size):
        #print(aber_map_size)

        n_l = 0
        x_bridged = []

        if self.use_coordconv:
            x = func.elu(list(self.layer_path)[n_l](x, xy_field, aber_map_size))
        else:
            x = func.elu(list(self.layer_path)[n_l](x))
        n_l += 1
        x = func.elu(list(self.layer_path)[n_l](x))
        n_l += 1

        x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(3):
                if isinstance(list(self.layer_path)[n_l], CoordConv):
                    x = func.elu(list(self.layer_path)[n_l](x, xy_field, aber_map_size))
                else:
                    x = func.elu(list(self.layer_path)[n_l](x))
                n_l += 1
                if n == 2 and i < self.n_stages - 1:
                    x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(4):
                if isinstance(list(self.layer_path)[n_l], CoordConv):
                    x = func.elu(list(self.layer_path)[n_l](x, xy_field, aber_map_size))
                else:
                    x = func.elu(list(self.layer_path)[n_l](x))
                n_l += 1
                if n == 1:
                    x = torch.cat([x, x_bridged.pop()], 1)  # concatenate


        return x

class Conv2DReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation,  stride=1):
        super(Conv2DReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, stride, padding, dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        self.bn = nn.BatchNorm2d(layer_width)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)

        return out

class OutnetCoordConv128(nn.Module):
    """output module"""#128 专用
    def __init__(self, n_filters, pad=1, ker_size=3):
        super(OutnetCoordConv128, self).__init__()

        self.p_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                padding=pad).cuda()

        self.xyzi_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                   padding=pad).cuda()
        self.p_out1 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()  # fu

        self.xyzi_out1 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()  # fu

        nn.init.kaiming_normal_(self.p_out.weight, mode='fan_in', nonlinearity='relu')  # all positive
        nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='sigmoid')  # all in (0, 1)
        nn.init.constant_(self.p_out1.bias, -6.)  # -6

        nn.init.kaiming_normal_(self.xyzi_out.weight, mode='fan_in', nonlinearity='relu')  # all positive
        nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='tanh')  # all in (-1, 1)
        nn.init.zeros_(self.xyzi_out1.bias)

        self.xyzis_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                    padding=pad).cuda()
        self.xyzis_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()

        nn.init.kaiming_normal_(self.xyzis_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.zeros_(self.xyzis_out2.bias)



    def forward(self, x):

        outputs = {}
        p = elu(self.p_out(x))
        outputs['p'] = self.p_out1(p)

        xyzi =elu(self.xyzi_out(x))
        outputs['xyzi'] = self.xyzi_out1(xyzi)
        xyzis = elu(self.xyzis_out1(x))
        outputs['xyzi_sig'] = self.xyzis_out2(xyzis)
        return outputs
        
class LocalizationCNN_Unet_downsample_128_Unet(nn.Module):
    def __init__(self):
        super(LocalizationCNN_Unet_downsample_128_Unet, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer0 = Conv2DReLUBN(1, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # downsample the input image size
        self.layer1 = Conv2DReLUBN(64, 64, 3, 1, 1)
        self.layer2 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer3 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer30 = Conv2DReLUBN(128, 64, 3, 1, 1)

        self.layer4 = Conv2DReLUBN(128, 64, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
        self.layer5 = Conv2DReLUBN(128, 64, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
        self.layer6 = Conv2DReLUBN(128, 64, 3, (8, 8), (8, 8))
        self.layer7 = Conv2DReLUBN(128, 64, 3, (16, 16), (16, 16))

        self.deconv1 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layerU1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU2 = Conv2DReLUBN(64, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU3 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD3 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD2 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD0 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD00 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.pool = nn.AvgPool2d(2, stride=2)
        self.pred = OutnetCoordConv128(64, 1, 3)

        diag = 0
        self.p_Conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.p_Conv.bias = None
        self.p_Conv.training = False
        self.p_Conv.weight.data = torch.Tensor([[[[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]]])



    def forward(self, im, test=False, index = -1, coord = [], threhold = 0.6):
        # time.sleep(0.0015)

        # extract multi-scale features
        img_h, img_w, batch_size = im.shape[-2], im.shape[-1], im.shape[0]
        im0 = self.norm(im)  # (10, 1, 128, 128)
        im1 = self.layer0(im0)  # (10, 64, 128, 128)
        im = self.pool1(im1)  # (10, 64, 64, 64)

        out = self.layer1(im)  # (10, 64, 64, 64)
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)

        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)
        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)
        out = self.layer30(features) + out
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)

        out4 = self.layer4(features) + out
        features = torch.cat((out4, im), 1)  # (10, 128, 64, 64)
        out = self.layer5(features) + out4
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)
        out6 = self.layer6(features) + out
        features = torch.cat((out6, im), 1)  # (10, 128, 64, 64)
        out7 = self.layer7(features) + out4 + out6  # (10, 64, 64, 64)
        features = torch.cat((out7, im), 1)  # (10, 128, 64, 64)

        out1 = self.deconv1(features)  # (10, 64, 64, 64)

        # Unet Stage
        out = self.pool(out1)  # (10, 64, 32, 32)
        out = self.layerU1(out)  # (10, 64, 32, 32)
        out = self.layerU2(out)  # (10, 128, 32, 32)
        out = self.layerU3(out)  # (10, 128, 32, 32)

        out = interpolate(out, scale_factor=2)  # (10, 128, 64, 64)
        out = self.layerD3(out)  # (10, 64, 64, 64)
        out = torch.cat([out, out1], 1)  # (10, 128, 64, 64)
        out = self.layerD2(out)  # (10, 64, 64, 64)
        out = self.layerD1(out)  # (10, 64, 64, 64)
        out = interpolate(out, scale_factor=2)  # (10, 64, 128, 128)
        out = self.layerD0(out)  # (10, 64, 128, 128)
        out = self.layerD00(out)  # (10, 64, 128, 128)

        out = self.pred(out)
        p = torch.sigmoid(torch.clamp(out['p'], -16., 16.))

        xyzi_est = out['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # photon
        xyzi_sig = torch.sigmoid(out['xyzi_sig']) + 0.001
        p = p[:, 0]
        if test:
            p_clip = torch.where(p > 0.3, p, torch.zeros_like(p))[:, None]
            # localize maximum values within a 3x3 patch
            pool = func.max_pool2d(p_clip, 3, 1, padding=1)
            max_mask1 = torch.eq(p[:, None], pool).float()
            # Add probability values from the 4 adjacent pixels
            conv = self.p_Conv(p[:, None])
            p_ps1 = max_mask1 * conv
            # In order do be able to identify two fluorophores in adjacent pixels we look for probablity values > 0.6 that are not part of the first mask
            p_copy = p * (1 - max_mask1[:, 0])
            # p_clip = torch.where(p_copy > 0.6, p_copy, torch.zeros_like(p_copy))[:, None]  # fushuang
            max_mask2 = torch.where(p_copy > 0.6, torch.ones_like(p_copy), torch.zeros_like(p_copy))[:,
                        None]  # fushuang
            p_ps2 = max_mask2 * conv
            # This is our final clustered probablity which we then threshold (normally > 0.7) to get our final discrete locations
            p = p_ps1 + p_ps2

            xyzi_est[:, 0] += 0.5 + coord[0]
            xyzi_est[:, 1] += 0.5 + coord[1]
            p = p[:,0]
            
            p_index = torch.where(p>threhold)
            temp_len = len(p_index[0])

            self.results_array[: temp_len,0] = p_index[0] + index
            self.results_array[: temp_len,1] = ((xyzi_est[:,0])[p_index] + p_index[2]) 
                   #self.results_array[self.curindex:self.curindex + temp_len,1] = x
            self.results_array[:temp_len,2] = ((xyzi_est[:,1])[p_index] + p_index[1])  

            self.results_array[:temp_len,3] = (xyzi_est[:,2])[p_index] 
            self.results_array[:temp_len,4] = (xyzi_est[:,3])[p_index]
            self.results_array[:temp_len,5] = p[p_index]
            
            return self.results_array[:temp_len],temp_len
            
        return p, xyzi_est, xyzi_sig  

        
    def get_parameter_number(self,dummy_input):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')
        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        self.eval()
        torch.cuda.synchronize()
        t0 = time.time()
        cnum = 0
        for i in range(1000):
            # dummy_input = torch.randn(10, 1, 1024, 1024).cuda()
            self.forward(dummy_input)
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0) / 1000:.4f} s')

        

class FdDeeploc(nn.Module):
    def __init__(self, net_pars, feature=64):
        super(FdDeeploc, self).__init__()
        self.net_pars = net_pars

        self.local_context = net_pars['local_flag']
        self.sig_pred = net_pars['sig_pred']
        self.psf_pred = net_pars['psf_pred']
        self.n_filters = net_pars['n_filters']
        # self.p_Conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        self.n_inp = 3 if self.local_context else 1
        n_features = self.n_filters * self.n_inp
        self.frame_module = UnetCoordConv(n_inp=1, n_filters=self.n_filters, n_stages=2,
                                          use_coordconv=self.net_pars['use_coordconv'])
        self.context_module = UnetCoordConv(n_inp=n_features, n_filters=self.n_filters, n_stages=2,
                                            use_coordconv=self.net_pars['use_coordconv'])
        self.out_module = OutnetCoordConv(self.n_filters, self.sig_pred, self.psf_pred, pad=self.net_pars['padding'],
                                          ker_size=self.net_pars['kernel_size'],
                                          use_coordconv=self.net_pars['use_coordconv'])

        diag = 0
        self.p_Conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.p_Conv.bias = None
        self.p_Conv.training = False
        self.p_Conv.weight.data =  torch.Tensor([[[[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]]])




    def forward(self, X, test=False,field_xy=[0,0],aber_map_size=[0,0]):

        # extract multi-scale features
        # tt =time.time()
        img_h, img_w = X.shape[-2], X.shape[-1]

        # simple normalization
        scaled_x = (X - self.net_pars['offset']) / self.net_pars['factor']

        if X.ndimension() == 3:  # when test, X.ndimension = 3
            scaled_x = scaled_x[:, None]
            fm_out = self.frame_module(scaled_x, field_xy, aber_map_size)
            if self.local_context:
                zeros = torch.zeros_like(fm_out[:1])
                h_t0 = fm_out
                h_tm1 = torch.cat([zeros, fm_out], 0)[:-1]
                h_tp1 = torch.cat([fm_out, zeros], 0)[1:]
                fm_out = torch.cat([h_tm1, h_t0, h_tp1], 1)
        else:  # when train, X.ndimension = 4
            fm_out = self.frame_module(scaled_x.reshape([-1, 1, img_h, img_w]), field_xy, aber_map_size) \
                .reshape(-1, self.n_filters * self.n_inp, img_h, img_w)

        # cm_in = fm_out

        # layernorm
        fm_out_LN = nn.functional.layer_norm(fm_out, normalized_shape=[self.n_filters * self.n_inp, img_h, img_w])
        cm_in = fm_out_LN

        cm_out = self.context_module(cm_in, field_xy, aber_map_size)
        outputs = self.out_module.forward(cm_out, field_xy, aber_map_size)

        if self.sig_pred:
            xyzi_sig = torch.sigmoid(outputs['xyzi_sig']) + 0.001
        else:
            xyzi_sig = 0.2 * torch.ones_like(outputs['xyzi'])

        probs = torch.sigmoid(torch.clamp(outputs['p'], -16., 16.))

        xyzi_est = outputs['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # ph
        psf_est = torch.sigmoid(outputs['bg'])[:, 0] if self.psf_pred else None


        p = probs[:, 0]
        if test:
            diag = 0
            p_clip = torch.where(p > 0.3, p, torch.zeros_like(p))[:, None]

            # localize maximum values within a 3x3 patch

            pool = func.max_pool2d(p_clip, 3, 1, padding=1)
            max_mask1 = torch.eq(p[:, None], pool).float()

            # Add probability values from the 4 adjacent pixels

            conv = self.p_Conv(p[:, None])
            p_ps1 = max_mask1 * conv

            # In order do be able to identify two fluorophores in adjacent pixels we look for probablity values > 0.6 that are not part of the first mask

            p_copy = p * (1 - max_mask1[:, 0])
            p_clip = torch.where(p_copy > 0.6, p_copy, torch.zeros_like(p_copy))[:, None]  # fushuang
            max_mask2 = torch.where(p_copy > 0.6, torch.ones_like(p_copy), torch.zeros_like(p_copy))[:,
                        None]  # fushuang
            p_ps2 = max_mask2 * conv

            # This is our final clustered probablity which we then threshold (normally > 0.7) to get our final discrete locations
            p = p_ps1 + p_ps2

        return p, xyzi_est, xyzi_sig

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(10, 1, 512,512).cuda()

        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        self.eval()
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(100):
            # dummy_input = torch.randn(1, 1, 256, 1024).cuda()
            self.forward(dummy_input)
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0) / 100:.4f} s')
