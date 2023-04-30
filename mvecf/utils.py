import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import pickle
import numpy as np

import os

__all__ = [
    "check_stop_flag",
    "save_pickle",
    "load_pickle",
    "cal_mean_variance",
    "cal_correlation",
    "plot_loss"
]


def check_stop_flag(dirpath):
    load_path = None
    stop_flag = False
    if not os.path.exists(dirpath):
        return stop_flag, load_path
    tmp = os.listdir(dirpath)
    order = [int(x.split("epoch_")[1].split(".pkl")[0]) for x in tmp]
    order = np.argsort(order)
    tmp = list(np.array(tmp)[order])

    if not len(tmp) == 0:
        load_path = os.path.join(dirpath, tmp[-1])

        data = load_pickle(load_path)
        list_stop_cri = data["list_stop_cri"][-30:]
        if len(list_stop_cri) >= 30 and (
                list_stop_cri[0] <= min(list_stop_cri) or abs(list_stop_cri[0] - list_stop_cri[-1]) < 1
        ):
            stop_flag = True
        if np.isinf(list_stop_cri[-1]) or np.isnan(list_stop_cri[-1]):
            stop_flag = True
        if tmp[-1] == "epoch_500.pkl":
            stop_flag = True
        if stop_flag:
            if list_stop_cri[0] <= min(list_stop_cri):
                load_path = os.path.join(dirpath, tmp[-30])
            else:
                load_path = os.path.join(dirpath, tmp[-1])
    return stop_flag, load_path


def save_pickle(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)
    f.close()


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    f.close()
    return data


def cal_mean_variance(factor_params):
    beta = factor_params["beta"]
    mu = np.matmul(factor_params["factor_mean"], beta.T)
    sig2_factor = factor_params["factor_variance"]
    sig2_eps = np.diag(factor_params["sig_eps_square"])
    cov = np.matmul(np.matmul(beta, sig2_factor), beta.T) + sig2_eps
    return mu, cov


def cal_correlation(factor_params):
    _, cov = cal_mean_variance(factor_params)

    diag = np.sqrt(np.diag(cov))
    invdiag = diag ** -1
    correlation = np.clip(invdiag.reshape(-1, 1) * cov * invdiag, -1, 1)
    return correlation


def plot_loss(dirpath, name=None, total_loss=False, is_train=True):
    if is_train:
        train_or_val = "train"
    else:
        train_or_val = "val"
    tmp = os.listdir(dirpath)
    order = [int(x.split("epoch_")[1].split(".pkl")[0]) for x in tmp]
    order = np.argsort(order)
    tmp = list(np.array(tmp)[order])
    print(dirpath)
    result_dict = defaultdict(list)

    # result_dict = {"loss": [], "epoch": []}
    for epoch, file in enumerate(tmp):
        data = load_pickle(os.path.join(dirpath, file))

        result_dict["epoch"].append(epoch)
        for value in [f"{train_or_val}_loss", f"{train_or_val}_loss_mv", f"{train_or_val}_loss_rec"]:
            if value in data.keys():
                result_dict[value].append(data[value])

    result_df = pd.DataFrame.from_dict(result_dict)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    if total_loss:
        loss_type = f"{train_or_val}_loss"
    else:
        loss_type = f"{train_or_val}_loss_rec"
    ax1.set_ylabel('min loss = {:.2f}'.format(result_df[loss_type].min()),
                   color=color)  # we already handled the x-label with ax1
    ax1.set_xlabel('epoch')
    ax1.plot(result_df[loss_type], color=color, marker='.')
    ax1.tick_params(axis='y', labelcolor=color)

    if not total_loss and f"{train_or_val}_loss_mv" in result_df.columns:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('mv: min loss = {:.2f}'.format(result_df[f"{train_or_val}_loss_mv"].min()),
                       color=color)  # we already handled the x-label with ax1
        ax2.plot(result_df[f"{train_or_val}_loss_mv"], color=color, marker='.')
        ax2.tick_params(axis='y', labelcolor=color)

    if name is None:
        name = dirpath.split("results/")[-1]
    plt.suptitle(name)
    plt.gcf().subplots_adjust(left=0.16, right=0.825)
    plt.show()
    plt.close(fig)
