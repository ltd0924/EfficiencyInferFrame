import time
import torch
from tqdm import trange
from torch.cuda.amp import autocast as autocast
import numpy as np
from multiprocessing import Queue
from threading import *

from PSFLocModel import *
from utils.local_tifffile import *
from utils.record_utils import *


def sample_consumer(queue_local, tif_file, share_mem, read_pos, read_index, write_pos, write_index):
    vv = time.time()
    share_mem_size = share_mem.buf.nbytes
    mm = share_mem.buf

    while 1:
        data = queue_local.get()
        start_pos = write_pos.value
        if data[0] == -1:
            mm[start_pos:start_pos + 4] = int(-1).to_bytes(4, "little", signed=True)
            write_pos.value += 4
            break
        tif_img = np.array(TiffFile(tif_file[data[0]]).asarray(key=slice(data[1][0], data[1][1]), series=0),
                           dtype=np.float16)
        img_size = tif_img.nbytes
        if start_pos + img_size + 20 > share_mem_size:
            mm[start_pos:start_pos + 4] = int(0).to_bytes(4, "little")
            write_pos.value = 0
            write_index.value += 1
            start_pos = 0
        elif start_pos != 0:
            mm[start_pos:start_pos + 4] = (start_pos + 4).to_bytes(4, "little")
            write_pos.value += 4
            start_pos += 4
        while read_index.value < write_index.value and write_pos.value + img_size + 24 > read_pos.value:
            continue
        mm[start_pos:start_pos + 4] = img_size.to_bytes(4, "little")
        mm[start_pos + 4:start_pos + 8] = data[0].to_bytes(4, "little")
        mm[start_pos + 8:start_pos + 12] = data[1][0].to_bytes(4, "little")
        mm[start_pos + 12:start_pos + 16] = data[2][0].to_bytes(4, "little")
        mm[start_pos + 16:start_pos + 20] = data[2][1].to_bytes(4, "little")
        mm[start_pos + 20:start_pos + 20 + img_size] = tif_img.tobytes()
        write_pos.value += (20 + img_size)
    print(time.time() - vv)


def get_img_info(infer_par, batch_size):
    tif_file = []
    tif_file_name = []
    path_list = os.listdir(infer_par['img_path'])

    win_size = infer_par['win_size']
    padding = infer_par['padding']
    for f in path_list:
        if os.path.splitext(f)[1] == '.tif':
            tif_file_name.append(infer_par['img_path'] + f)
            tif_file.append(TiffFile(infer_par['img_path'] + f, is_ome=True))
    q = Queue()
    total_shape = tif_file[0].series[0].shape
    for g in range(len(tif_file)):
        for i in range(0, total_shape[0], batch_size):
            for j in range(int(np.ceil(total_shape[-1] / win_size))):
                for k in range(int(np.ceil(total_shape[-2] / win_size))):
                    q.put((g, [i, i + batch_size], [j * win_size, k * win_size]))
    for i in range(torch.cuda.device_count()):
        q.put([-1, -1, -1])
    return q, tif_file_name


class PsfInfer:
    def __init__(self, infer_par, shm, tif_file, device, net_params, fd=False):
        self.win_size = infer_par['win_size']
        self.shm = shm
        if fd:
            self.model = FdDeeploc(net_params).to(torch.device('cuda'))
        else:
            self.model = LocalizationCNN_Unet_downsample_128_Unet()
        self.loadmodel(infer_par['net_path'])
        self.device = device
        self.model = self.model.to(device)
        self.result_name_list = tif_file
        self.curindex = 0
        self.res = {}


    def loadmodel(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")

    # 或许可改成多进程写入
    def write_csv(self):
        for i in range(len(self.result_name_list)):
            with open(self.result_name_list[i] + '.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                if i not in self.res.keys():
                    print(self.result_name_list[i])
                    continue
                num = 0
                for x in self.res[i]:
                    num += len(x)
                    csvwriter.writerows(x.cpu().numpy())
                # print(self.result_name_list[i],num)

    def inferdata(self, r_pos, r_index, w_pos, w_index, scale_factor, threhold=0.6):
        self.model.eval()
        time_forward = 0
        time_cc = 0
        buf = self.shm.buf
        coord = [0, 0]

        with torch.set_grad_enabled(False):
            with autocast():

                while 1:
                    time_c = time.time()
                    cur_pos = r_pos.value
                    if cur_pos == -1:
                        break
                    img_size = int.from_bytes(buf[cur_pos:cur_pos + 4], "little")
                    while w_index.value < r_index.value or (
                            w_index.value == r_index.value and w_pos.value < cur_pos + 24 + img_size):
                        continue
                    findex = int.from_bytes(buf[cur_pos + 4:cur_pos + 8], "little")
                    index = int.from_bytes(buf[cur_pos + 8:cur_pos + 12], "little")
                    coord[0] = int.from_bytes(buf[cur_pos + 12:cur_pos + 16], "little")
                    coord[1] = int.from_bytes(buf[cur_pos + 16:cur_pos + 20], "little")
                    img = np.frombuffer(buf[cur_pos + 20:cur_pos + 20 + img_size], dtype=np.float16).reshape(
                        (-1, 1, self.win_size, self.win_size))

                    cur_pos = cur_pos + 20 + img_size
                    r_pos.value = int.from_bytes(buf[cur_pos:cur_pos + 4], "little")
                    if r_pos.value == 0:
                        r_index.value += 1
                    # print(self.result_name_list[findex],index)
                    time_cc += (time.time() - time_c)
                    time_c = time.time()

                    img = torch.Tensor(img).to(self.device)
                    ans, num = self.model(img, test=True, index=index + 1, coord=coord)  # ,threhold = threhold)
                    ans[1] *= scale_factor[0]
                    ans[2] *= scale_factor[1]
                    ans[3] *= scale_factor[2]
                    ans[4] *= scale_factor[3]

                    if findex not in self.res.keys():
                        # print(self.result_name_list[findex],index)
                        self.res[findex] = []

                    # self.res[findex].append(torch.cat(ans,dim=1))
                    self.curindex += num

                    time_forward = time_forward + (time.time() - time_c) * 1000
                    if findex == -1:
                        break

                print("time for network forward is {0} ，{2},num：{1}".format(time_forward, self.curindex, time_cc))




