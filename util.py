import numpy as np
import pandas as pd
from tqdm import tqdm
import gzip


def normalize(x):
    x = np.array(x)
    mean_x = x.mean(axis=1, keepdims=True)
    std_x = x.std(axis=1, keepdims=True)
    return (x - mean_x) / std_x


def extract_data(data: str):
    if data == "mnist":
        df = pd.read_csv(f"./data/{data}/train.csv").to_numpy(dtype=np.float32)
        t_mat = pd.get_dummies(df[:, 0]).to_numpy()
        x = df[:, 1:]
    else:
        x, t = [], []
        with gzip.open(f"./data/{data}/fer2013.tar.gz", "rb") as file:
            row_data = file.readlines()
        n = len(row_data)
        print("Extracting images...")
        for ind in tqdm(range(n)):
            try:
                row = row_data[ind].split(b",")
                t.append(int(row[0].decode("utf-8")))
                x.append([int(num) for num in row[1].decode("utf-8").split()])
            except ValueError:
                pass
        print("\n")
        x, t = np.array(x, dtype=np.float32), np.array(t)
        t_mat = pd.get_dummies(t).to_numpy()
    x_mat = normalize(x=x)
    return x_mat, t_mat


def split_data_th(x, split_percent):
    n = np.shape(x)[0]
    return int(n * split_percent / 100)


def i_o_layers(data, target):
    d = np.shape(data)[1]
    k = np.shape(target)[1]
    return d, k


def split_2_batch(x, t, ind, batch_size):
    xb = x[ind * batch_size: (ind + 1) * batch_size, :]
    tb = t[ind * batch_size: (ind + 1) * batch_size, :]
    return xb, tb


def linear_regression(z, w, b):
    return np.array(z).dot(w) + b


def softmax(a):
    max_a = np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a - max_a)
    return exp_a / exp_a.sum(axis=1, keepdims=True)


def active(a, activation: str):
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-a))
    elif activation == "tanh":
        return np.tanh(a)
    elif activation == "relu":
        return np.where(a > 0, a, 0)
    else:
        raise Exception("Not valid activation function")


def accuracy_per_set(y, t):
    t_vec = np.argmax(t, axis=1)
    y_vec = np.argmax(y, axis=1)
    return 100 * np.array(t_vec == y_vec).mean()


def grad_activation(z, activation: str):
    if activation == "sigmoid":
        return z * (1 - z)
    elif activation == "tanh":
        return 1 - z ** 2
    elif activation == "relu":
        return np.where(z > 0, 1, 0)
    else:
        raise Exception("Not valid activation function")


def grad_reg(reg: dict, w, gradient_type: str):
    g_reg = 0
    if reg["R"] in [1, 2]:
        lam = reg["lambda"]
        alpha = -1 if gradient_type == "descent" else 1
        g_reg += alpha * lam * np.sign(w) if reg["R"] == 1 else lam * w
    return g_reg
