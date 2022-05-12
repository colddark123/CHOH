import torch.nn as nn
import torch
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv,GATConv
import torch.nn.functional as F
from model.DenseModel import Dense
from model.Cutoff import CosineCutoff
import math
import numpy as np
from dgl.nn.pytorch.conv import NNConv,CFConv
from model.GaussianSmear import GaussianSmearing
from torch.nn import functional
from model.MPAConv import MPGATConv

def shifted_softplus(x):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return functional.softplus(x) - np.log(2.0)
class ShiftedSoftplus(nn.Module):
    r"""

    Description
    -----------
    Applies the element-wise function:

    .. math::
        \text{SSP}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) - \log(\text{shift})

    Attributes
    ----------
    beta : int
        :math:`\beta` value for the mathematical formulation. Default to 1.
    shift : int
        :math:`\text{shift}` value for the mathematical formulation. Default to 2.
    """
    def __init__(self, beta=1, shift=2, threshold=20):
        super(ShiftedSoftplus, self).__init__()

        self.shift = shift
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, inputs):
        """

        Description
        -----------
        Applies the activation function.

        Parameters
        ----------
        inputs : float32 tensor of shape (N, *)
            * denotes any number of additional dimensions.

        Returns
        -------
        float32 tensor of shape (N, *)
            Result of applying the activation function to the input.
        """
        return self.softplus(inputs) - np.log(float(self.shift))


def getdis(pointa, pointb):
    dist_vec = pointa - pointb
    distances = torch.norm(dist_vec, 2, 1)
    return distances
class SchNetInteraction(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        out_dim,
        num_heads,
        n_gaussians,
        cutoffFunc
    ):
        super(SchNetInteraction, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.numheads = num_heads
        self.n_gaussians = n_gaussians
        self.activation = ShiftedSoftplus()
        self.mpaconv = MPGATConv(self.embedding_dim, self.n_gaussians, self.hidden_dim,
                                 num_heads=num_heads,activation=self.activation,
                                 cutoffFunc=cutoffFunc)
        self.MLP = nn.Sequential(
            Dense(embedding_dim*num_heads,embedding_dim, bias=True, activation=shifted_softplus),
            Dense(embedding_dim, embedding_dim,bias=True),
        )
    def forward(self,g, x, gs, rij):
        res = self.mpaconv(g, x, gs, rij)
        res = res.reshape(-1, self.out_dim * self.numheads)
        res = self.MLP(res)
        return res
class Standardize(nn.Module):
    def __init__(self, mean, stddev):
        super(Standardize, self).__init__()
        self.mean=mean
        self.stddev=stddev
    def forward(self, input):
        y = input * self.stddev + self.mean
        return y
class Representation(nn.Module):
    def __init__(self,
        embedding_dim,
        hidden_dim,
        mlp_dim,
        output_dim,
        device,
        num_heads,
        start,
        stop,
        n_gaussians,
        mean,
        stddev,
        n_interactions=3,
        embedding_input = 100,
        ):
        super(Representation, self).__init__()
        self.n_interactions=n_interactions
        self.device=device
        self.hidden_dim=hidden_dim
        self.getdis=getdis
        self.embedding_dim=embedding_dim
        self.start=start
        self.mean=mean
        self.stddev=stddev
        self.embedding = nn.Embedding(embedding_input, embedding_dim, padding_idx=0)
        self.linearAtom = Dense(embedding_dim, embedding_dim, bias=False)
        self.distance_expansion = GaussianSmearing(
            start, stop, n_gaussians, trainable=True
        )
        self.cutoff = CosineCutoff(stop)
        self.activation=ShiftedSoftplus()
        self.interactions = nn.ModuleList([
                SchNetInteraction(
                    embedding_dim = embedding_dim,
                    hidden_dim = hidden_dim,
                    out_dim = embedding_dim,
                    num_heads = num_heads,
                    n_gaussians = n_gaussians,
                    cutoffFunc = self.cutoff
                )
                for _ in range(n_interactions)
            ]
        )
        self.OutputMLP = nn.Sequential(
            Dense(embedding_dim, mlp_dim, bias=True, activation=shifted_softplus),
            Dense(mlp_dim, output_dim, bias=True),
        )
        self.standardize = Standardize(self.mean, self.stddev)
    def forward(self, g):
        R = g.ndata['R']  # get coordinates of each atom
        Z = g.ndata['Z']  # get atomic numbers of each atom
        g.ndata['Z']=Z.float()
        #print(dgl.khop_adj(g, 1))
        #g = dgl.add_self_loop(g)
        g.apply_edges(lambda edges: {'dis': self.getdis(edges.src['R'], edges.dst['R'])})

        x = self.embedding(Z)
        #x = self.linearAtom(x)
        r = g.edata['dis']
        rij = r
        r = r.unsqueeze(0)
        r = r.unsqueeze(0)
        gs = self.distance_expansion(r)
        gs = gs.squeeze()
        gs = gs.to(self.device)
        for interaction in self.interactions:
            v = interaction(g,x,gs,rij)
            x = x + v
        h = self.OutputMLP(x)
        h = self.standardize(h)
        g.ndata['item']=h
        out = dgl.sum_nodes(g, 'item')
        #print(out.shape)
        #out = out.squeeze()
        #print(out.shape)
        return out









