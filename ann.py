from colorama import Fore
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from util import *


class ANN:
    def __init__(self, data, target, split_percent, hidden_size: list):
        np.random.seed(2020)
        th = split_data_th(x=data, split_percent=split_percent)
        self.X = {
            "train": data[:th, :],
            "test": data[th:, :]
        }
        self.T = {
            "train": target[:th, :],
            "test": target[th:, :]
        }
        self.D, self.K = i_o_layers(data=data, target=target)
        self.M = [self.D] + hidden_size + [self.K]
        self.params_size = len(self.M) - 1
        self.W = []
        self.v_w = []
        self.m_w = []
        self.grad_w = []
        self.b = []
        self.v_b = []
        self.m_b = []
        self.grad_b = []
        self.weights_init()
        self.cost = {}
        self.accuracy = {}
        self.outputs_init()
        self.t = 0
        self.plt_num = 2

    def weights_refresh(self):
        self.W = []
        self.v_w = []
        self.m_w = []
        self.grad_w = []
        self.b = []
        self.v_b = []
        self.m_b = []
        self.grad_b = []

    def weights_init(self):
        for ind in range(self.params_size):
            a0, a1 = self.M[ind], self.M[ind + 1]
            w = np.random.randn(a0, a1) / np.sqrt(a0 * a1)
            w0 = np.zeros((a0, a1))
            b = np.zeros((1, a1))
            #
            self.W.append(w)
            self.v_w.append(w0)
            self.m_w.append(w0)
            #
            self.b.append(b)
            self.v_b.append(b)
            self.m_b.append(b)

    def outputs_init(self):
        for set_type in ["train", "test"]:
            self.cost[set_type] = []
            self.accuracy[set_type] = []

    def shuffle_data(self, set_type: str):
        self.X[set_type], self.T[set_type] = shuffle(self.X[set_type],
                                                     self.T[set_type])

    def batch_num(self, set_type: str, batch_size):
        n = np.shape(self.X[set_type])[0]
        if set_type == "test":
            batch_size = n
        if batch_size == "FGD":
            batch_size = n
        elif batch_size == "SGD":
            batch_size = 1
        elif batch_size > n:
            raise Exception(f"Not valid batch size, batch size <= {n}")
        return int(np.ceil(n / batch_size)), batch_size

    def forward(self, x, activation: str):
        z = [x]
        y = 0
        for ind in range(self.params_size):
            w = self.W[ind]
            b = self.b[ind]
            a = linear_regression(z=z[-1], w=w, b=b)
            if ind == self.params_size - 1:
                y = softmax(a=a)
            else:
                z.append(active(a=a, activation=activation))
        return z, y

    def loss_reg(self, reg: dict):
        loss_r = 0
        if reg["R"] not in [0, 1, 2]:
            raise Exception("Not valid regularization")
        if reg["R"] in [1, 2]:
            lam = reg["lambda"]
            lam /= reg["R"]
            for ind in range(self.params_size):
                w_vec = np.array(self.W[ind]).flatten()
                if reg["R"] == 1:
                    loss_r += lam * np.absolute(w_vec).sum()
                else:
                    loss_r += lam * np.array(w_vec ** 2).sum()
        return loss_r

    def cross_entropy(self, t, y, reg: dict, gradient_type: str):
        l_temp = - np.sum(t * np.log(y), axis=1, keepdims=True)
        loss = np.mean(l_temp) + self.loss_reg(reg=reg)
        if gradient_type == "ascent":
            loss *= -1
        return loss

    def forward_per_set(self, set_type: str, gradient_type: str,
                        activation: str, reg: dict, ind, batch_size):
        xb, tb = split_2_batch(x=self.X[set_type], t=self.T[set_type],
                               batch_size=batch_size, ind=ind)
        zb, yb = self.forward(x=xb, activation=activation)
        lb = self.cross_entropy(t=tb, y=yb, reg=reg, gradient_type=gradient_type)
        self.cost[set_type].append(lb)
        acc_b = accuracy_per_set(y=yb, t=tb)
        self.accuracy[set_type].append(acc_b)
        return zb, yb, tb, lb, acc_b

    def symbol(self, set_type: str):
        if len(self.cost[set_type]) == 1:
            return np.round(self.cost[set_type][-1], 4)
        temp_min = np.min(self.cost[set_type][:-1])
        temp_max = np.max(self.cost[set_type][:-1])
        if self.cost[set_type][-1] > temp_max:
            return "↗"
        elif self.cost[set_type][-1] < temp_min:
            return "↘"
        else:
            return "🔃"

    def append_gradients(self, grad_w, grad_b):
        self.grad_w.append(grad_w)
        self.grad_b.append(grad_b)

    def weights_reversed(self):
        self.grad_w.reverse()
        self.grad_b.reverse()

    def gradients(self, z, y, t, activation: str,
                  gradient_type: str, lr: dict, reg: dict):
        delta = t - y
        if gradient_type == "ascent":
            delta *= -1
        grad_w = np.transpose(z[-1]).dot(delta) + grad_reg(w=self.W[-1], reg=reg,
                                                           gradient_type=gradient_type)
        grad_b = np.sum(delta, axis=0, keepdims=True)
        self.append_gradients(grad_w=grad_w, grad_b=grad_b)
        for ind in reversed(range(self.params_size - 1)):
            grad_a = grad_activation(z=z[ind + 1], activation=activation)
            if lr["type"] == "nesterov":
                mu = lr["mu"]
                w = self.W[ind + 1] - mu * self.v_w[ind + 1]
            else:
                w = self.W[ind + 1]
            delta = np.dot(delta, np.transpose(w)) * grad_a
            g_r = grad_reg(w=self.W[ind], reg=reg, gradient_type=gradient_type)
            grad_w = np.transpose(z[ind]).dot(delta) + g_r
            grad_b = np.sum(delta, axis=0, keepdims=True)
            self.append_gradients(grad_w=grad_w, grad_b=grad_b)
        self.weights_reversed()

    def update_rule(self, ind, eta_w, eta_b, alpha):
        self.W[ind] -= alpha * eta_w * self.grad_w[ind]
        self.b[ind] -= alpha * eta_b * self.grad_b[ind]

    def momentum(self, ind, eta, mu, alpha):
        self.v_w[ind] = mu * self.v_w[ind] + eta * self.grad_w[ind]
        self.v_b[ind] = mu * self.v_b[ind] + eta * self.grad_b[ind]
        #
        self.W[ind] -= alpha * self.v_w[ind]
        self.b[ind] -= alpha * self.v_b[ind]

    def adagrad(self, ind, eta, eps, alpha):
        self.v_w[ind] = np.power(self.grad_w[ind], 2)
        self.v_b[ind] = np.power(self.grad_b[ind], 2)
        #
        self.W[ind] -= alpha * eta * self.grad_w[ind] / (np.sqrt(self.v_w[ind]) + eps)
        self.b[ind] -= alpha * eta * self.grad_b[ind] / (np.sqrt(self.v_b[ind]) + eps)

    def rmsprop(self, ind, eta, beta, eps, alpha):
        self.v_w[ind] += beta * self.v_w[ind] + (1 - beta) * (self.grad_w[ind] ** 2)
        self.v_b[ind] += beta * self.v_b[ind] + (1 - beta) * (self.grad_b[ind] ** 2)
        #
        self.W[ind] -= alpha * eta * self.grad_w[ind] / (np.sqrt(self.v_w[ind]) + eps)
        self.b[ind] -= alpha * eta * self.grad_b[ind] / (np.sqrt(self.v_b[ind]) + eps)

    def adam(self, ind, eta, beta1, beta2, eps, alpha):
        self.t += 1
        self.m_w[ind] = beta1 * self.m_w[ind] + (1 - beta1) * self.grad_w[ind]
        m_w = self.m_w[ind] / (1 - beta1 ** self.t)
        self.v_w[ind] = beta2 * self.v_w[ind] + (1 - beta2) * self.grad_w[ind] ** 2
        v_w = self.v_w[ind] / (1 - beta2 ** self.t)
        self.W[ind] -= alpha * eta * m_w / (np.sqrt(v_w) + eps)
        #
        self.m_b[ind] = beta1 * self.m_b[ind] + (1 - beta1) * self.grad_b[ind]
        m_b = self.m_b[ind] / (1 - beta1 ** self.t)
        self.v_b[ind] = beta2 * self.v_b[ind] + (1 - beta2) * self.grad_b[ind] ** 2
        v_b = self.v_b[ind] / (1 - beta2 ** self.t)
        self.b[ind] -= alpha * eta * m_b / (np.sqrt(v_b) + eps)

    def learning_rate_techniques(self, lr: dict, alpha):
        eta = lr["lr"]
        for ind in range(self.params_size):
            if lr["type"] == "none":
                self.update_rule(ind, eta, eta, alpha)
            elif lr["type"] in ["momentum", "nesterov"]:
                mu = lr["mu"]
                self.momentum(ind, eta, mu, alpha)
            elif lr["type"] == "adagrad":
                eps = lr["eps"]
                self.adagrad(ind, eta, eps, alpha)
            elif lr["type"] == "rmsprop":
                beta = lr["beta"]
                eps = lr["eps"]
                self.rmsprop(ind, eta, beta, eps, alpha)
            else:
                beta1 = lr["beta1"]
                beta2 = lr["beta2"]
                eps = lr["eps"]
                self.adam(ind, eta, beta1, beta2, eps, alpha)

    def backward_per_set(self, z, y, t, activation: str,
                         reg: dict, lr: dict, gradient_type: str):
        alpha = 1 if gradient_type == "ascent" else -1
        self.gradients(z=z, y=y, t=t, lr=lr, reg=reg,
                       activation=activation, gradient_type=gradient_type)

        self.learning_rate_techniques(lr=lr, alpha=alpha)

    def collect_figures(self, batch_size, plot_name):
        plt.figure(num=1)
        plt.plot(self.cost["train"], label=plot_name)
        plt.figure(num=2)
        if batch_size == 1:
            plt.scatter(self.accuracy["train"], label=plot_name)
        else:
            plt.plot(self.accuracy["train"], label=plot_name)
        self.plt_num += 1
        for set_type in ["train", "test"]:
            plt.figure(num=self.plt_num)
            plt.plot(self.cost[set_type], label=f"{set_type} - {plot_name}")

    def outputs_refresh(self):
        self.cost = {
            "train": [],
            "test": []
        }
        self.accuracy = {
            "train": [],
            "test": []
        }
        self.t = 0

    def train(self, gradient_type: str, activation: str, learning_rate: dict,
              regularization: dict, batch_size, epochs, plot_name: str):
        epoch_bar = tqdm(range(epochs))
        epoch_bar.colour = Fore.GREEN
        for epoch in epoch_bar:
            for set_type in ["train", "test"]:
                self.shuffle_data(set_type=set_type)
                b_num, batch_size = self.batch_num(set_type=set_type,
                                                   batch_size=batch_size)
                for b in range(b_num):
                    zb, yb, tb, lb, acc_b = self.forward_per_set(set_type=set_type, gradient_type=gradient_type,
                                                                 activation=activation, batch_size=batch_size,
                                                                 ind=b, reg=regularization)
                    if set_type == "train":
                        epoch_bar.set_description(f" epoch {epoch} / {epochs} |"
                                                  f" batch {b} / {b_num} | accuracy {np.round(acc_b, 2)}% "
                                                  f": {self.symbol(set_type=set_type)} ")
                        self.backward_per_set(z=zb, y=yb, t=tb, activation=activation, reg=regularization,
                                              lr=learning_rate, gradient_type=gradient_type)
        self.collect_figures(batch_size=batch_size, plot_name=plot_name)
        self.weights_refresh()
        self.weights_init()
        self.outputs_refresh()

    def show_results(self):
        plt.figure(num=1)
        plt.legend()
        plt.title("Cost function")
        plt.figure(num=2)
        plt.legend()
        plt.title("Accuracy")
        for num in range(3, self.plt_num + 1):
            plt.figure(num=num)
            plt.legend()
            plt.title("Cost function")
        plt.show()
