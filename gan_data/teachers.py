#!/usr/bin/env python3
#
# Models to train two-layer nets on (DCGAN, ResNet) generative model.
#
# Date: December 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

from abc import ABCMeta, abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models.resnet


SQRT2 = 1.414213562
SQRT2OVERPI = 0.797884560802865


def erfscaled(x):
    """
    Re-scaled error function activation function.
    """
    return torch.erf(x / SQRT2)


def linear(x):
    """
    Linear activation function.

    Useful to make the connection with an ODE description.
    """
    return x


def relu(x):
    """
    ReLU activation function.
    """
    return F.relu(x)


class Model(nn.Module, metaclass=ABCMeta):
    """
    Abstract class for all the models used in these experiments.
    """

    _input_dim = None

    @property
    def input_dim(self):
        """
        Input dimension of the network
        """
        return self._input_dim

    @property
    @abstractmethod
    def requires_2d_input(self):
        """
        True if the network requires inputs as 2D arrays (images).
        """

    @abstractmethod
    def preprocess(self, x):
        """
        Applies all the layers to the input before the last fully-connected D->K layer.
        """

    def forward(self, x):
        """
        Computes the output of the network.
        """
        x = self.nu(x)
        x = self.g(x)
        x = self.v(x)
        return x

    def nu(self, x):
        """
        Computes the pre-activation at the last hidden layer of neurons of the network.
        """
        x = self.preprocess(x)
        x = self.fc(x / math.sqrt(self.D))
        return x

    def nu_y(self, x):
        """
        Computes the pre-activation of the last hidden layer and the network's output.
        """
        nu = self.nu(x)
        y = self.g(nu)
        y = self.v(y)
        return nu, y

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def _norm_second_layer(self):
        """
        Normalises the weight vector of the 'second-layer'.
        """
        self.v.weight.data /= torch.sqrt(torch.sum(self.v.weight.data ** 2))


class TwoLayer(Model):
    """
    Plain two-layer neural network with ODE scaling.
    """

    requires_2d_input = False

    def __init__(self, g, N, K, std0w=None, std0v=None):
        """
        Parameters:
        -----------

        N : input dimension
        K : number of hidden nodes
        std0w :
           std dev of the initial FIRST-layer (Gaussian) weights.
        std0v :
           std dev of the initial SECOND-layer (Gaussian) weights.
        """
        super().__init__()
        self.g = g
        self._input_dim = N
        self.D = N
        self.K = K

        self.fc = nn.Linear(N, K, bias=False)
        self.v = nn.Linear(K, 1, bias=False)

        if std0w is not None:
            nn.init.normal_(self.fc.weight, mean=0.0, std=std0w)
        if std0v is not None:
            nn.init.normal_(self.v.weight, mean=0.0, std=std0v)
        self._norm_second_layer()

    def preprocess(self, x):
        return x


class SmallMLP(Model):
    """
    Multi-layer perceptron with three fully connected layers with skip connections,
    followed by a two-layer nn.
    """

    requires_2d_input = False

    def __init__(self, g, N, K):
        """
        Parameters:
        -----------

        N : input dimension
        K : number of hidden nodes
        """
        super().__init__()
        self._input_dim = N
        self.g = g
        self.D = N
        self.K = K

        self.preprocess1 = nn.Linear(N, self.D, bias=False)

        # add a batch-norm layer before the last fully connected layer
        self.bnz = nn.BatchNorm1d(self.D, affine=False, track_running_stats=False)

        self.fc = nn.Linear(self.D, K, bias=False)
        self.v = nn.Linear(K, 1, bias=False)

        nn.init.normal_(self.fc.weight, mean=0.0, std=1)
        nn.init.normal_(self.v.weight, mean=0.0, std=1)
        self._norm_second_layer()

    def preprocess(self, x):
        """
        Propagates the given inputs x through the MLP until we hit the fully-connected
        layer that projects things down to the M classes.
        """
        x = F.relu(self.preprocess1(x))

        x = self.bnz(x)  # an additional BatchNorm layer to ensure that z has zero mean

        return x


class MLP(Model):
    """
    Multi-layer perceptron with three fully connected layers with skip connections,
    followed by a two-layer nn.
    """

    requires_2d_input = False

    def __init__(self, g, N, K):
        """
        Parameters:
        -----------

        N : input dimension
        K : number of hidden nodes
        """
        super().__init__()
        self._input_dim = N
        self.g = g
        self.D = N
        self.K = K

        self.preprocess1 = nn.Linear(N, N, bias=False)
        self.preprocess2 = nn.Linear(N, N, bias=False)
        self.preprocess3 = nn.Linear(N, self.D, bias=False)

        # add a batch-norm layer before the last fully connected layer
        self.bnz = nn.BatchNorm1d(self.D, affine=False, track_running_stats=False)

        self.fc = nn.Linear(self.D, K, bias=False)
        self.v = nn.Linear(K, 1, bias=False)

        nn.init.normal_(self.fc.weight, mean=0.0, std=1)
        nn.init.normal_(self.v.weight, mean=0.0, std=1)
        self._norm_second_layer()

    def preprocess(self, x):
        """
        Propagates the given inputs x through the MLP until we hit the fully-connected
        layer that projects things down to the M classes.
        """
        x = F.relu(self.preprocess1(x))
        x = F.relu(self.preprocess2(x)) + x
        x = F.relu(self.preprocess3(x)) + x

        x = self.bnz(x)  # an additional BatchNorm layer to ensure that z has zero mean

        return x


class ConvNet(Model):
    """Simple convolutional network.

    The architecture and the pre-processing of the network are taken from the pyTorch
    tutorial:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
    """

    requires_2d_input = True

    def __init__(self, g, K, input_dim=[3, 32, 32]):
        super().__init__()

        self._input_dim = input_dim
        self.K = K
        self.g = g

        self.network = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),  # output: 64 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 5 x 5
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 2 x 2
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.ReLU(),
        )

        self.D = 784

        # add a batch-norm layer before the last fully connected layer
        self.bnz = nn.BatchNorm1d(self.D, affine=False, track_running_stats=False)

        self.fc = nn.Linear(self.D, K, bias=False)
        self.v = nn.Linear(K, 1, bias=False)

        nn.init.normal_(self.fc.weight, mean=0.0, std=1)
        nn.init.normal_(self.v.weight, mean=0.0, std=1)

        self._norm_second_layer()
        self.v.requires_grad = False

    def preprocess(self, x):
        """
        Propagates the given inputs x through the MLP until we hit the fully-connected
        layer that projects things down to the M classes.
        """
        x = self.network(x)

        x = self.bnz(x)  # an additional BatchNorm layer to ensure that z has zero mean

        return x


class ScalarResnet(Model):
    """
    A Resnet18 with a single output head.
    """

    requires_2d_input = True

    def __init__(self, g, num_classes=1, **kwargs):
        super().__init__()

        # Create an instance of a resnet to have the pre-processing
        self._resnet = torchvision.models.resnet.ResNet(
            torchvision.models.resnet.BasicBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
            **kwargs
        )

        self._input_dim = (1, 32, 32)
        self.num_classes = num_classes
        self.g = g
        self.D = max(self._resnet.fc.weight.data.shape)
        self.K = 1

        # Two tricks to get better accuracy, taken from Joost van Amersfoort
        # https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469
        self._resnet.conv1 = torch.nn.Conv2d(
            self._input_dim[0], 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self._resnet.maxpool = torch.nn.Identity()

        # add a batch-norm layer before the last fully connected layer
        self.bnz = nn.BatchNorm1d(self.D, affine=False, track_running_stats=False)

        # the two MLP layers where the GET magic happens
        self.fc = nn.Linear(self.D, self.K, bias=False)
        self.v = nn.Linear(self.K, 1, bias=False)

        nn.init.normal_(self.fc.weight, mean=0.0, std=1)
        nn.init.uniform_(self.v.weight.data, 1)
        self.v.requires_grad = False  # keep it constant

    def preprocess(self, x, block=4):
        """Propagates the given inputs x through the ResNet until we would hit its
        fully-connected layer that projects things down to the K classes.

        Code is taken
        directly from _forward_impl of pyTorch ResNet implementation, only x =
        self.fc(x) is missing at the end.

        """
        x = self._resnet.conv1(x)
        x = self._resnet.bn1(x)
        x = self._resnet.relu(x)
        x = self._resnet.maxpool(x)

        if block > 0:
            x = self._resnet.layer1(x)
        if block > 1:
            x = self._resnet.layer2(x)
        if block > 2:
            x = self._resnet.layer3(x)
        if block > 3:
            x = self._resnet.layer4(x)

        x = self._resnet.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.bnz(x)  # an additional BatchNorm layer to ensure that z has zero mean

        return x
