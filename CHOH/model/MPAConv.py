import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
import numpy as np
from torch.nn import functional
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

    def __init__(self, beta=1, shift=2, threshold=20):
        super(ShiftedSoftplus, self).__init__()

        self.shift = shift
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, inputs):
        return self.softplus(inputs) - np.log(float(self.shift))
class MPGATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 edge_feats,
                 out_feats,
                 num_heads,
                 cutoffFunc,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(MPGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._edge_feats =edge_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.cutoffFunc=cutoffFunc

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)

        self.project_edge1 = nn.Linear(edge_feats, out_feats)
        self.project_edge2 = nn.Linear(out_feats, out_feats * num_heads)
        self.shiftactivation = ShiftedSoftplus()
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_edge = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
            nn.init.xavier_normal_(self.project_edge1.weight, gain=gain)
            nn.init.xavier_normal_(self.project_edge2.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.project_edge1.weight, gain=gain)
            nn.init.xavier_normal_(self.project_edge2.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_edge, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, nfeat, efeat ,rij, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            #feat 6*5   6
            src_prefix_shape = dst_prefix_shape = nfeat.shape[:-1]
            edge_prefix_shape = efeat.shape[:-1]


            h_src = h_dst = self.feat_drop(nfeat)
            #6 6
            feat_src = feat_dst = self.fc(h_src).view(
            *src_prefix_shape, self._num_heads, self._out_feats)
            #*src_prefix_shape 6  [6,3,2]
            #false 太棒了
            #attn_l attn_l [1, 3, 2]
            #feat_dst   [6, 3, 2]
            edgefeat = self.project_edge1(efeat)
            edgefeat = self.shiftactivation(edgefeat)
            edgefeat = self.project_edge2(edgefeat)
            edgefeat = self.shiftactivation(edgefeat)
            # print(edgefeat.shape)
            # print(self.cutoffFunc(rij).shape)
            C=self.cutoffFunc(rij)
            C=C.unsqueeze(-1)
            edgefeat = C*edgefeat

            edgefeat = edgefeat.view(
                *edge_prefix_shape, self._num_heads, self._out_feats)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            eedge = (edgefeat * self.attn_edge).sum(dim=-1).unsqueeze(-1)

            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'dis': eedge})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            graph.edata['sum']=graph.edata['e']+graph.edata['dis']

            e = self.leaky_relu(graph.edata.pop('sum'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            #点乘注意力权重，在这
            graph.edata['a'] = graph.edata['a']*graph.edata['dis']
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
