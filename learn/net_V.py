import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from utils.Config_B import CegisConfig


class Net(nn.Module):
    def __init__(self, config: CegisConfig):
        super(Net, self).__init__()
        self.config = config
        self.input_size = input_size = config.EXAMPLE.n
        hiddens = config.N_HIDDEN_NEURONS
        activate = config.ACTIVATION
        self.acts = config.ACTIVATION
        self.bias = bias = config.BIAS
        self.layers1 = []  # backbone network,such as square
        self.layers2 = []  # used for skip, mul and other networks
        k = 1
        n_prev = config.EXAMPLE.n
        for n_hid, act in zip(hiddens, activate):
            layer1 = nn.Linear(n_prev, n_hid, bias=bias)
            if act == 'SKIP':
                layer2 = nn.Linear(input_size, n_hid, bias=(bias & (k != len(hiddens))))
            else:
                layer2 = nn.Linear(n_prev, n_hid, bias=(bias & (k != len(hiddens))))

            self.register_parameter("W" + str(k), layer1.weight)
            self.register_parameter("W2" + str(k), layer2.weight)
            if bias:
                self.register_parameter("b" + str(k), layer1.bias)
                self.register_parameter("b2" + str(k), layer2.bias)
            self.layers1.append(layer1)
            self.layers2.append(layer2)

            n_prev = n_hid
            k = k + 1

        # free output layer
        layer1 = nn.Linear(n_prev, 1, bias=False)
        self.register_parameter("W" + str(k), layer1.weight)
        self.layers1.append(layer1)

        # generalisation of forward with tensors

    def forward(self, x, xdot):

        y = x
        jacobian = torch.diag_embed(torch.ones(x.shape[0], self.input_size))
        for idx, (layer1, layer2) in enumerate(zip(self.layers1[:-1], self.layers2)):
            if self.acts[idx] == 'SQUARE':
                z = layer1(y)
                y = z ** 2
                jacobian = torch.matmul(torch.matmul(2 * torch.diag_embed(z), layer1.weight), jacobian)

            elif self.acts[idx] == 'MUL':
                z1 = layer1(y)
                z2 = layer2(y)
                y = z1 * z2
                grad = torch.matmul(torch.diag_embed(z1), layer2.weight) + torch.matmul(torch.diag_embed(z2),
                                                                                        layer1.weight)
                jacobian = torch.matmul(grad, jacobian)


            elif self.acts[idx] == 'SKIP':
                z1 = layer1(y)
                z2 = layer2(x)
                y = z1 * z2
                jacobian = torch.matmul(torch.diag_embed(z1), layer2.weight) + torch.matmul(
                    torch.matmul(torch.diag_embed(z2), layer1.weight), jacobian)

        numerical_v = torch.matmul(y, self.layers1[-1].weight.T)
        jacobian = torch.matmul(self.layers1[-1].weight, jacobian)
        numerical_vdot = torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1)
        return numerical_v[:, 0], numerical_vdot

    def get_lyapunov(self):
        x = sp.symbols([['x{}'.format(i + 1) for i in range(self.input_size)]])
        y = x
        for idx, (layer1, layer2) in enumerate(zip(self.layers1[:-1], self.layers2)):
            if self.acts[idx] == 'SQUARE':
                w1 = layer1.weight.detach().numpy()
                z = np.dot(y, w1.T)
                if self.bias:
                    z = z + layer1.bias.detach().numpy()
                y = z ** 2
            elif self.acts[idx] == 'MUL':
                w1 = layer1.weight.detach().numpy()
                z1 = np.dot(y, w1.T)
                if self.bias:
                    z1 += layer1.bias.detach().numpy()

                w2 = layer2.weight.detach().numpy()
                z2 = np.dot(y, w2.T)
                if self.bias & (idx != len(self.layers2) - 1):
                    z2 += layer2.bias.detach().numpy()

                y = np.multiply(z1, z2)
            elif self.acts[idx] == 'SKIP':
                w1 = layer1.weight.detach().numpy()
                z1 = np.dot(y, w1.T)
                if self.bias:
                    z1 += layer1.bias.detach().numpy()

                w2 = layer2.weight.detach().numpy()
                z2 = np.dot(x, w2.T)
                if self.bias & (idx != len(self.layers2) - 1):
                    z2 += layer2.bias.detach().numpy()

                y = np.multiply(z1, z2)

        w1 = self.layers1[-1].weight.detach().numpy()
        y = np.dot(y, w1.T)
        y = sp.expand(y[0, 0])
        return y


class Learner(nn.Module):
    def __init__(self, config: CegisConfig):
        super(Learner, self).__init__()
        self.net = Net(config)
        self.loss_weight = config.LOSS_WEIGHT
        self.config = config

    def learn(self, optimizer, S, Sdot):
        """
        :param optimizer: torch optimiser
        :param S: tensor of data
        :param Sdot: tensor contain f(data)
        :param margin: performance threshold
        :return: --
        """

        assert (len(S) == len(Sdot))
        print('samples:', len(S))
        learn_loops = 500
        margin = self.config.MARGIN
        slope = 1e-3
        relu6 = torch.nn.ReLU6()
        for t in range(learn_loops):
            optimizer.zero_grad()

            V, Vdot = self.net(S, Sdot)
            circle = torch.pow(S, 2).sum(dim=1)

            accuracy_v = sum(V >= margin * circle).item() * 100 / len(S)
            accuracy_vdot = sum(Vdot < -margin * circle).item() * 100 / len(S)

            loss = self.loss_weight[0] * (torch.relu(-V + margin * circle) - slope * relu6(V - margin * circle)).mean()
            loss = loss + self.loss_weight[1] * (
                    torch.relu(Vdot + margin * circle) - slope * relu6(-Vdot - margin * circle)).mean()

            if t % int(learn_loops / 10) == 0 or (accuracy_v == 100 and accuracy_vdot == 100):
                print(t, "- loss:", loss.item(), '- accuracy V:', accuracy_v, 'accuracy Vdot:', accuracy_vdot)

            loss.backward()
            optimizer.step()
            if accuracy_v == 100 and accuracy_vdot == 100:
                break
