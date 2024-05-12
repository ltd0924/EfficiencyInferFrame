import numpy as np
import csv
import pandas as pd
import torch
import os
from scipy.spatial.distance import cdist
from utils.help_utils import *


def check_csv(name):
    ind = 0
    if os.path.exists(name):
        with open(name) as csvfile:
            mLines = csvfile.readlines()

        targetLine = mLines[-1]
        ind = targetLine.split(',')[0]
    else:
        with open(name, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'xnano', 'ynano', 'znano', 'intensity','pnms'])
    return 0

def write_dict_csv(pred_list, name):
    """Writes a csv_file with columns: 'localizatioh', 'frame', 'x', 'y', 'z', 'intensity','x_sig','y_sig','z_sig'

    Parameters
    ----------
    pred_list : list
        List of localizations
    name: str
        File name
    """

    with open(name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for _,v in pred_list.items():
            csvwriter.writerows(v)

def write_csv(pred_list, name):
    """Writes a csv_file with columns: 'localizatioh', 'frame', 'x', 'y', 'z', 'intensity','x_sig','y_sig','z_sig'

    Parameters
    ----------
    pred_list : list
        List of localizations
    name: str
        File name
    """
    with open(name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for p in pred_list:
            csvwriter.writerow([repr(f) for f in p])


def list_to_dict(test_csv, pred = False, scale=[100,100,700,10000], limit_x=[0,6400], limit_y=[0,6400]):
    test_list = {}
    num = 0
    if isinstance(test_csv, str):
        with open(test_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                if pred:
                    row = row[:-1]
                try:
                    if float(row[-4]) < limit_x[0] or float(row[-4]) > limit_x[1]:
                        continue
                    if float(row[-3]) < limit_y[0] or float(row[-3]) > limit_y[1]:
                        continue
                    if int(float(row[-5])) not in test_list:
                        test_list[int(float(row[-5]))] = []
                    test_list[int(float(row[-5]))].append(
                        [float(row[-5]), float(row[-4]), float(row[-3]), float(row[-2]), float(row[-1])])
                    num += 1
                except:
                    print("error")
    elif isinstance(test_csv,dict):
        return test_csv, len(test_csv)
    else:
        for i in range(len(test_csv)):
            for j in range(len(test_csv[0])):
                for r in test_csv[i][j]:
                    if not isinstance(r, int):
                        if r[-1] == 0:
                            break
                        k = i * len(test_csv[0]) + j + 1
                        if k not in test_list:
                            test_list[k] = []
                        if float(r[-4]) < limit_x[0] or float(r[-4]) > limit_x[1]:
                            continue
                        if float(r[-3]) < limit_y[0] or float(r[-3]) > limit_y[1]:
                            continue

                        test_list[k].append([k, r[-4] * scale[0],
                                             r[-3] * scale[1],
                                             r[-2] * scale[2],
                                             r[-1] * scale[3]])
                    else:
                        r = test_csv[i][j]
                        k = i * len(test_csv[0]) + j + 1
                        if k not in test_list:
                            test_list[k] = []
                        if float(r[-4]) < limit_x[0] or float(r[-4]) > limit_x[1]:
                            break
                        if float(r[-3]) < limit_y[0] or float(r[-3]) > limit_y[1]:
                            break
                        test_list[k].append([k, r[-4] * scale[0],
                                             r[-3] * scale[1],
                                             r[-2] * scale[2],
                                             r[-1] * scale[3]])
                        break

                    num += 1
    return test_list, num

def assess(pred_data, test_data, tolerance = 200,tolerance_ax = 500, min_int = 1000,scale=[100,100,700,10000],  limit_x=[0,6400], limit_y=[0,6400]):
    # time_postprocess = time.time()
    test_list, num = list_to_dict(test_data,scale=scale, limit_x=limit_x, limit_y = limit_y)
    print('\nevaluation on {0} images,contain ground truth: {1}'.format(len(test_list), num))
    pred_list, num = list_to_dict(pred_data, True,scale=scale, limit_x=limit_x, limit_y = limit_y)
    print('\nevaluation on {0} images,contain ground truth: {1}'.format(len(pred_list), num))
    perf_dict, matches = limited_matching(test_list, pred_list, tolerance, tolerance_ax, min_int)

    return perf_dict, matches

def limited_matching(gt_list, pred_list, tolerance ,tolerance_ax, min_int):
    matches = []
    perf_dict = {'recall': 0, 'precision': 0, 'jaccard': 0, 'rmse_lat': 0,
                 'rmse_ax': 0, 'rmse_vol': 0, 'jor': 0, 'eff_lat': 0, 'eff_ax': 0,
                 'eff_3d': 0}

    TP = 0
    FP = 0.0001
    FN = 0.0001
    MSE_lat = 0
    MSE_ax = 0
    MSE_vol = 0
    truth_num = 0
    pred_num = 0
    for k in range(len(gt_list) + 1):
        # if preds is empty, it means no detection on the frame, all tests are FN

        if k not in gt_list and k not in pred_list:
            continue  # no need to calculate metric
        if k in gt_list:
            FN += len(gt_list[k])
            truth_num += len(gt_list[k])
        if k in pred_list:
            FP += len(pred_list[k])
            pred_num += len(pred_list[k])
        if k not in gt_list or k not in pred_list:
            continue

        tests = gt_list[k]
        preds = pred_list[k]
        # calculate the Euclidean distance between all gt and preds, get a matrix [number of gt, number of preds]
        dist_arr = cdist(np.array(tests)[:, 1:3], np.array(preds)[:, 1:3])
        ax_arr = cdist(np.array(tests)[:, 3:4], np.array(preds)[:, 3:4])
        tot_arr = np.sqrt(dist_arr ** 2 + ax_arr ** 2)

        if tolerance_ax == np.inf:
            tot_arr = dist_arr

        if dist_arr.size > 0:
            while dist_arr.min() < tolerance:
                r, c = np.where(tot_arr == tot_arr.min())
                if len(r) == 0:
                    break  # select the positions pair with shortest distance
                r = r[0]
                c = c[0]
                if ax_arr[r, c] < tolerance_ax and dist_arr[
                    r, c] < tolerance:  # compare the distance and tolerance
                    if tests[r][-1] > min_int:  # photons should be larger than min_int

                        MSE_lat += dist_arr[r, c] ** 2
                        MSE_ax += ax_arr[r, c] ** 2
                        MSE_vol += dist_arr[r, c] ** 2 + ax_arr[r, c] ** 2
                        TP += 1
                        FP -= 1
                        FN -= 1

                        matches.append([tests[r][1], tests[r][2], tests[r][3], tests[r][4],
                                        preds[c][1], preds[c][2], preds[c][3], preds[c][4]])

                    dist_arr[r, :] = np.inf
                    dist_arr[:, c] = np.inf
                    tot_arr[r, :] = np.inf
                    tot_arr[:, c] = np.inf

                dist_arr[r, c] = np.inf
                tot_arr[r, c] = np.inf

    print('after FOV and border segmentation,truth: {0},preds: {1}'.format(truth_num, pred_num))
    if len(matches) == 0:
        print('matches is empty!')
        return perf_dict, matches

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    jaccard = TP / (TP + FP + FN)
    rmse_lat = np.sqrt(MSE_lat / ((TP + 0.00001) * 2))
    rmse_ax = np.sqrt(MSE_ax / (TP + 0.00001))
    rmse_vol = np.sqrt(MSE_vol / (TP + 0.00001))
    jor = 100 * jaccard / rmse_lat

    eff_lat = 100 - np.sqrt((100 - 100 * jaccard) ** 2 + 1 ** 2 * rmse_lat ** 2)
    eff_ax = 100 - np.sqrt((100 - 100 * jaccard) ** 2 + 0.5 ** 2 * rmse_ax ** 2)
    eff_3d = (eff_lat + eff_ax) / 2

    matches = np.array(matches)
    perf_dict = {'recall': recall, 'precision': precision, 'jaccard': jaccard, 'rmse_lat': rmse_lat,
                 'rmse_ax': rmse_ax, 'rmse_vol': rmse_vol, 'jor': jor, 'eff_lat': eff_lat, 'eff_ax': eff_ax,
                 'eff_3d': eff_3d}

    return perf_dict, matches