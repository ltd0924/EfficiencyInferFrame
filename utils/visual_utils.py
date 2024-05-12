import matplotlib.pyplot as plt

from PSF_vector_gpu.PsfSimulation import *
from utils.help_utils import *
from utils.dataGenerator import *
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
import tifffile as tiff


def ShowSamplePSF(psf_pars, type, interval = 12):

    psf_pars['robust_training'] = False
    PSF = PsfSimulation(psf_pars,type)


    x_offset = np.zeros([interval, 1, 1])
    y_offset = np.zeros([interval, 1, 1])
    z = np.linspace(-psf_pars['z_scale'], psf_pars['z_scale'], interval).reshape(interval, 1, 1)

    psf_samples = PSF.sim_psf_vector(X_os=x_offset, Y_os=y_offset, Z=z)
    psf_samples /= psf_samples.sum(-1).sum(-1)[:, None, None]
    psf_samples = cpu(psf_samples)
    plt.imshow(psf_samples[0])
    plt.savefig('pic.png')
    
    #plt.figure(dpi=400)
    #plt.clf()
    #plt.suptitle(type+'PSF')
    #start = int(np.sqrt(interval))
    #while interval % start != 0:
    #    start = start - 1

    #for j in range(interval):
    #    plt.subplot(start, int(interval / start), j + 1)
    #    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.05)
    #    plt.xticks([])
    #    plt.yticks([])
    #    plt.imshow(psf_samples[j])
    #    plt.title(str(np.round(z[j, 0, 0], 0)) + ' nm', fontdict={'size': 10})

    #plt.show()

def ShowTrainImg(image_num, data_params,camera_params,psf_params,type='DMO'):
    DataGen = DataGenerator(data_params,camera_params,psf_params,type)
    DataGen.batch_size = 1
    S, X, Y, Z, I, s_mask, xyzi_gt,_ = DataGen.generateBatch(size=image_num, val=False)
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
        plt.imshow(np.squeeze(cpu(S[i])))
    plt.show()
    imgs_sim = DataGen.simulatedImg(S, X, Y, Z, I)
    plt.figure()
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
        plt.imshow(np.squeeze(cpu(imgs_sim[i][0])))
    plt.show()



# plot_emitter_distance_distribution (fy)
def plot_emitter_distance_distribution(data_path):
    activation = pd.read_csv(data_path)

    group_activation = activation.groupby("frame")

    dis_count = 0
    frame = np.unique(activation.values[:, 0])
    dis_record = [[] for j in range(len(frame))]
    
    count = 0
    for i in frame:
        index = group_activation.indices[i]  # 这里修改过，注意！！！i+1 --> i
        emitter_x = activation['x'][index]
        emitter_y = activation['y'][index]
        emitter = torch.tensor([emitter_x.values, emitter_y.values]).transpose(1, 0)
        dis_matrix = torch.norm(emitter[:, None]-emitter, dim=2, p=2)
        # dis_matrix = dis_matrix - torch.diag_embed(torch.diag(dis_matrix))
        if len(dis_matrix) != 1:
            for k in range(len(dis_matrix)):
                dis_matrix[k][k] = 1e5
        dis_min = torch.min(dis_matrix, dim=1).values  # dim=1表示在每行上进行操作
        # if dis_min.max() > 12800:
        #     print(dis_matrix)
        # dis_matrix = F.pdist(emitter, p=2)
        dis_count += (dis_min < 990).sum()  # math.sqrt(2) * (1400) / 2 ~= 990
        # dis_record[i] = dis_matrix
        dis_record[count] = dis_min
        count += 1

    dis_record = np.array(torch.cat(dis_record, dim=0))
    # pd.DataFrame(dis_record).to_csv('calculate_distance_Astigmatism_density=2.csv', index=False, header=False)

    print('end')
    mean = dis_record.mean()
    variance = dis_record.std()
    fig, ax = plt.subplots()

    # n, bins, patches = ax.hist(dis_record, bins=range(0, 8000, 120), density=True)
    n, bins, patches = ax.hist(dis_record, bins=50, density=True)

    y = norm.pdf(bins, mean, variance)  # y = ((1 / (np.sqrt(2 * np.pi) * variance)) *np.exp(-0.5 * (1 / variance * (bins - mean))**2))
    plt.plot(bins, y, 'r')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Probability density')
    ax.set_title('His. of normal distribution: '
                 fr'counts_lt={dis_count:.0f}, $\mu={mean:.0f}$, $\sigma={variance:.0f}$')
    fig.tight_layout()
    plt.show()


def plot_train(data_path, metric = 'cost_hist'):
    result = open(data_path, 'rb')
    checkpoint = pickle.load(result)

    recall = checkpoint[metric]
    recall_list = list(recall.values())
    index = list(recall)

    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.plot(index, recall_list)
    plt.show()


def GenerateData(psf, img_num, data_params, camera_params, psf_params, type='DMO', z_scale=700):
    datagene = DataGenerator(data_params, camera_params, psf_params, type)
    datagene.batch_size = 1
    S, X, Y, Z, I, s_mask, xyzi_gt, _ = datagene.generateBatch(size=img_num, val=False, local_context=True)
    img_sim = datagene.simulatedImg(S, X, Y, Z, I)

    img_gene = list(np.array(torch.unsqueeze(img_sim, dim=1).cpu()))
    tiff.imwrite(psf + '_generate_randomdata_128-4-100_local_context.tif', img_gene)

    frame, x, y, z, In, img_gene = [], [], [], [], [], []
    # output, counts = torch.unique_consecutive(S.nonzero()[:, 0], return_counts=True)
    for i in range(img_num*300):
        if min(S[i].nonzero().shape) == 0:
            continue
        else:
            # xyzi = np.array(xyzi_gt[i].cpu())
            s_ind = S[i].nonzero()
            for j in range(len(S[i].nonzero())):
                frame.append(i + 1)
                x.append((X[j] + s_ind[0]) * 110)
                y.append((Y[j] + s_ind[1]) * 110)
                z.append(Z[j] * z_scale)
                In.append(In[j] * 6000)
    activation = [frame, x, y, z, In]
    activation_df = pd.DataFrame(activation).transpose()
    activation_df = activation_df.rename(columns={0: 'frame', 1: 'x', 2: 'y', 3: 'z', 4: 'intensity'})
    activation_df.to_csv(psf + '_generate_randomdata_128-4-100_local_context.csv', mode='w', index=False)

