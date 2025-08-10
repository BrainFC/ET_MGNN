import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F

from RWKV.RWKV4TS import RWKVBlock
from einops import rearrange


class ModuleTimestamping(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, t, sampling_endpoints):
        return self.rnn(t[:sampling_endpoints[-1]])[0][[p-1 for p in sampling_endpoints]]

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Lancelot39/KGSF/blob/a10c02cf31dd70d55e972c9ad046df16886f88e1/models/graph.py#L74
    https://github.com/HKUDS/SSLRec/blob/85216c5c426465ff6df5f296b7193666424188b3/models/multi_behavior/kmclr.py#L685
    https://github.com/subercui/pyGConvAT/blob/3f5953feca72d4bec79d2fca4916a36f04b6eb7e/layers.py#L7
    https://github.com/subercui/pyGAT/tree/34956c1ca1538108ea4db6ae091d99e4d180f98e  SpGAT
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha #parameter of leakyReLU
        self.concat = concat
        #Define trainable parameters
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        Wh = torch.mm(h, self.W) #((batch*slice*roi,hidden_dim))
        e = self._prepare_attentional_mechanism_input(Wh)#(batch*slice*roi,batch*slice*roi)
        #
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention1 = F.softmax(attention, dim=1)# to get normalized attention weight
        attention = F.dropout(attention1, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)#(batch*slice*roi,hidden_dim))

        if self.concat:
            return F.elu(h_prime),attention1
        else:
            return h_prime,attention1

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# https://github.com/phamvanhanh6720/PDDA/blob/6a56a54d9a4e8cf060fac6ffb1664323a8baa662/model/model.py#L52
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        #multi-head
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
    def forward(self, x, adj):#adj(8,116,116)
        adj=abs(adj).to_dense()
        x = F.dropout(x, self.dropout, training=self.training)
        aa= [att(x, adj) for att in self.attentions]
        att_list= []
        for i in range(len(aa)):
            att_list.append(aa[i][1])
        x_list = [] #list 5 each: (batch*slice*roi,hidden_dim)
        for i in range(len(aa)):
            x_list.append(aa[i][0])
        x = torch.cat(x_list, dim=1) #(batch*slice*roi,hidden_dim*nhead)
        x = F.dropout(x, self.dropout, training=self.training)
        x,learned_att = self.out_att(x, adj)  # (928,64)#(batch*slice*roi,hidden_dim)
        x = F.elu(x)
        return x
# MLP_GIN
# https://github.com/zdhNarsil/GFlowNet-CombOpt/blob/306bff0316a9c886fd7dfbeccf9e9c1cda104c9a/gflownet/network.py#L16
class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())

    #dyn_v=dyn_v 【b t n1 n2】,dyn_a=dyn_a: minibatchsiz (6)* sampling_endpoints 24,(150长度情下)， 246 * 246.
    def forward(self, v, a):
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += self.epsilon * v # assumes that the adjacency matrix includes self-loop
        v_combine = self.mlp(v_aggregate)
        return v_combine


class ModuleBARO(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_time, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.heads = num_heads
        self.head_dim = hidden_dim * num_time // self.heads
        self.embed_size = hidden_dim * num_time
        self.embed_query = nn.Linear(self.head_dim, round(self.head_dim * upscale))
        self.embed_key = nn.Linear(self.head_dim, round(self.head_dim * upscale))
        self.embed_value = nn.Linear(self.head_dim, self.embed_size)

        self.fc = nn.Linear(hidden_dim * num_time, hidden_dim * num_time)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]

        num_time, batch, nodes, feature = x.shape#([8, 16, 246, 16])
        x_mean = x.mean(node_axis)
        x = x.permute(1, 2, 3, 0).reshape(batch, nodes, feature * num_time)#([16,8, 246*16])
        x = x.reshape(batch, nodes, self.heads, self.head_dim)
        x_q = self.embed_query(x)
        x_k = self.embed_key(x)
        x_v = self.embed_value(x)
        energy = torch.einsum("nqhd,nkhd->nhqk", [x_q, x_k])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        attention_d = self.dropout(attention)
        out = torch.einsum("nhql,nlhd->nqhd", [attention_d, x_v]).reshape(batch, nodes, feature * num_time)
        out = self.fc(out)
        out = out.view(nodes, batch, num_time, feature).permute(2, 1, 0, 3).mean(node_axis)
        out = x_mean + self.dropout(out)

        return out, attention



class ModuleMixedAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0,**kwargs):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale * hidden_dim)),
                                   nn.BatchNorm1d(round(upscale * hidden_dim)), nn.GELU())
        self.attend_query = nn.Linear(hidden_dim, round(upscale * hidden_dim))
        self.attend_key = nn.Linear(hidden_dim, round(upscale * hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):  # x.shape: (segment_num, batch_size, 116, 64)
        global_context = x.mean(node_axis)  # global_context.shape: (segment_num, batch_size, 64)

        # Generate query and key vectors
        query_vectors = self.attend_query(global_context)  # (segment_num, batch_size, upscale_dim)
        key_vectors = self.attend_key(x)  # (segment_num, batch_size, 116, upscale_dim)

        # Reshape query_vectors to match dimensions for matrix multiplication
        query_vectors = query_vectors.unsqueeze(node_axis)  # (segment_num, batch_size, 1, upscale_dim)

        # Compute attention scores
        attention_scores = torch.matmul(query_vectors,
                                        key_vectors.transpose(-1, -2))  # (segment_num, batch_size, 1, 116)
        attention_scores = torch.softmax(attention_scores, dim=-1)  # Apply softmax along the last dimension
        # Expand attention_scores to match the dimensions of x
        attention_scores = attention_scores.expand(-1, -1, -1, x.shape[-1])  # (segment_num, batch_size, 246, 64)

        # Apply attention scores to the input
        x_new = x * self.dropout(attention_scores)  # (segment_num, batch_size, 246, 64)

        # Process the features with embedding
        x_embed = self.embed(x_new.reshape(-1, x_new.shape[-1]))  # Flatten and embed
        x_graphattention = torch.sigmoid(x_embed).view(*x_new.shape[:-1], -1)  # (segment_num, batch_size, 246)

        # Adjust dimensions
        permute_idx = list(range(node_axis)) + [len(x_graphattention.shape) - 1] + list(
            range(node_axis, len(x_graphattention.shape) - 1))
        x_graphattention = x_graphattention.permute(permute_idx)

        return (x_new * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1, 0, 2)

class ModuleCombinedAttention(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        # ModuleSERO-inspired components
        self.embed_sero = nn.Sequential(
            nn.Linear(hidden_dim, round(upscale * hidden_dim)),
            nn.BatchNorm1d(round(upscale * hidden_dim)),
            nn.GELU()
        )
        self.attend_sero = nn.Linear(round(upscale * hidden_dim), input_dim)

        # ModuleGARO-inspired components
        self.embed_query = nn.Linear(hidden_dim, round(upscale * hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale * hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        # print("===== ModuleCombinedAttention ===",x.shape)

        # SERO-inspired attention
        x_readout = x.mean(node_axis)  # Aggregating node information
        x_shape = x_readout.shape
        x_embed_sero = self.embed_sero(x_readout.reshape(-1, x_shape[-1]))
        x_graphattention_sero = torch.sigmoid(self.attend_sero(x_embed_sero)).view(*x_shape[:-1], -1)
        # print("## x_graphattention_sero ##",x_graphattention_sero.shape)

        # Reordering the dimensions to apply attention along the correct axis
        permute_idx = list(range(node_axis)) + [len(x_graphattention_sero.shape) - 1] + list(
            range(node_axis, len(x_graphattention_sero.shape) - 1))
        x_graphattention_sero = x_graphattention_sero.permute(permute_idx)
        # print("## x_graphattention_sero 2 ##",x_graphattention_sero.shape)

        # GARO-inspired attention
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention_garo = torch.sigmoid(
            torch.matmul(x_q, rearrange(x_k, '... n c -> ... c n')) / np.sqrt(x_q.shape[-1])
        ).squeeze(2)
        # print("## x_graphattention_garo ##",x_graphattention_garo.shape)

        # Adjusting dimensions for multiplication
        if x_graphattention_sero.shape[2] != x_graphattention_garo.shape[2]:
            # Ensure both tensors have the same size along the node axis (dimension 2)
            x_graphattention_garo = x_graphattention_garo.unsqueeze(-1).expand_as(x_graphattention_sero)
            # print("## x_graphattention_garo 2 ##", x_graphattention_garo.shape)

        # Combining SERO and GARO attention mechanisms
        combined_attention = x_graphattention_sero * x_graphattention_garo
        # print("## combined_attention  ##", combined_attention.shape)

        x_recalibrated = x * self.dropout(combined_attention.unsqueeze(-1))
        # print("## x_recalibrated  ##", x_recalibrated.shape)

        return x_recalibrated.mean(node_axis), combined_attention.permute(1, 0, 2)


class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32)


class ChannelAttentionGNN(nn.Module):
    def __init__(self, c, reduction=16):
        super(ChannelAttentionGNN, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(c, c // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(c // reduction, c, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (t, b, n, c)
        t, b, n, c = x.shape
        x_reshaped = x.view(t * b * n, c)
        # x_reshaped = x.reshape(t * b * n, c)  # 使用 reshape 代替 view

        avg_out = self.global_avg_pool(x_reshaped.unsqueeze(-1).unsqueeze(-1)).view(t * b * n, c)
        max_out = self.global_max_pool(x_reshaped.unsqueeze(-1).unsqueeze(-1)).view(t * b * n, c)
        avg_out = self.fc1(avg_out)
        max_out = self.fc1(max_out)
        out = self.fc2(self.relu(avg_out + max_out))
        attention = self.sigmoid(out).view(t, b, n, c)

        return x * attention, attention  # 返回加权后的特征和注意力权重
class SpatialAttentionGNN(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionGNN, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (t, b, n, c)
        t, b, n, c = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()  # (t, b, c, n)
        x_avg = torch.mean(x, dim=2, keepdim=True)
        x_max, _ = torch.max(x, dim=2, keepdim=True)
        x = torch.cat([x_avg, x_max], dim=2)  # (t, b, 2, n)
        attention = self.conv(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # (t, b, n, 1)
        attention = self.sigmoid(attention)

        return x.permute(0, 1, 3, 2) * attention, attention  # 返回加权后的特征和注意力权重


class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, node_axis=1):
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))# 节点维度，node_axis值是2
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n'))/np.sqrt(x_q.shape[-1])).squeeze(2)


        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)


class ModuleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))


    def forward(self, x):
        # print("=========",x.device)
        # x = x.to(next(self.parameters()).device)
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend) # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix


# Percentile class based on
# https://github.com/aliutkus/torchpercentile
class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)

    def forward(self, input, percentiles):
        input = torch.flatten(input) # find percentiles for flattened axis
        input_dtype = input.dtype
        input_shape = input.shape
        if isinstance(percentiles, int):
            percentiles = (percentiles,)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        input = input.double()
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)
        in_sorted, in_argsort = torch.sort(input, dim=0)
        positions = percentiles * (input.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0+d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)

    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector

        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred
class ModelRWKV(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, sparsity, dropout=0.5,
                 cls_token='sum', readout='sero', garo_upscale=1.0):
    # def __init__(self, input_dim, hidden_dim, num_heads, num_classes, num_time, num_layers, sparsity, dropout=0.5,
    #              cls_token='sum', readout='baro', garo_upscale=1.0):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token == 'sum':
            self.cls_token = lambda x: x.sum(0)
        elif cls_token == 'mean':
            self.cls_token = lambda x: x.mean(0)
        elif cls_token == 'param':
            self.cls_token = lambda x: x[-1]
        else:
            raise
        if readout == 'garo':
            readout_module = ModuleGARO
        elif readout == 'sero':
            readout_module = ModuleSERO
        elif readout == 'mean':
            readout_module = ModuleMeanReadout
        elif readout == 'baro':
            readout_module = ModuleBARO
        elif readout == 'gsero':
            readout_module = ModuleCombinedAttention
        else:
            raise

        self.token_parameter = nn.Parameter(
            torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token == 'param' else None

        self.num_classes = num_classes
        self.sparsity = sparsity

        # define modules
        self.percentile = Percentile()
        # self.timestamp_encoder = ModuleTimestamping(input_dim, hidden_dim, hidden_dim) #如果没有时间编码的话，这行可以注释掉
        # self.initial_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.initial_linear = nn.Linear(input_dim , hidden_dim)

        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            # self.gnn_layers.append(GAT(nfeat=hidden_dim, nhid=hidden_dim,  dropout=0.0,nheads=2,alpha=0.2))
            # self.gnn_layers.append(GraphSAGE(hidden_dim, hidden_dim))

            # attention original OK
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))

            # self.readout_modules.append(readout_module(hidden_dim=hidden_dim))

            # for BARO
            # self.readout_modules.append(readout_module(hidden_dim=hidden_dim, num_heads=num_heads, num_time=num_time, input_dim=input_dim, dropout=0.1))

            # === original === ST
            # self.transformer_modules.append(
            #     ModuleTransformer(hidden_dim, 2 * hidden_dim, num_heads=num_heads, dropout=0.1))

            # == RWKVBlock ==  OK
            self.transformer_modules.append(RWKVBlock(input_dim = hidden_dim,layer_id= 0))


            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))

    def _collate_adjacency(self, a, sparsity, sparse=True):
        i_list = []
        v_list = []

        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > self.percentile(_a, 100 - sparsity))
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)

        # return torch.sparse.FloatTensor(_i, _v,(a.shape[0] * a.shape[1] * a.shape[2], a.shape[0] * a.shape[1] * a.shape[3]))

        # 创建一个全零的密集矩阵，并根据索引填充值
        dense_shape = (a.shape[0] * a.shape[1] * a.shape[2], a.shape[0] * a.shape[1] * a.shape[3])
        dense_adj = torch.zeros(dense_shape, device=a.device)
        dense_adj[_i[0], _i[1]] = _v  # 根据索引填充值
        return dense_adj

    def forward(self, v, a, t, sampling_endpoints):
        # assumes shape [minibatch x time x node x feature] for v
        # assumes shape [minibatch x time x node x node] for a
        # print("==== RWKV ===v ",v.shape, "a ",a.shape," t",t.shape,"sampling_endpoints ",sampling_endpoints)
        modularityloss = 0.0
        logit = 0.0
        reg_ortho = 0.0
        epsilon = 1e-8
        reconstruct_loss = 0.0 # add 917 ljw

        attention = {'node-attention': [], 'time-attention': []}
        latent_list = []
        minibatch_size, num_timepoints, num_nodes = a.shape[:3]
        # print(" ===================", minibatch_size, num_timepoints, num_nodes) #  8 ,16, 246
        # time_encoding = self.timestamp_encoder(t, sampling_endpoints)
        # print("xxx RWKV after time_encoding:",time_encoding.shape)
        # time_encoding = repeat(time_encoding, 'b t c -> t b n c', n=num_nodes)
        # print("xxx RWKV after repeat:",time_encoding.shape)

        # h = torch.cat([v, time_encoding], dim=3)
        h = v

        # print("xxx RWKV h cat:",h.shape)

        h = rearrange(h, 'b t n c -> (b t n) c')
        # print("xxx RWKV h rearrange:",h.shape)

        h = self.initial_linear(h)
        # print("xxx RWKV h linear:",h.shape)

        a = self._collate_adjacency(a, self.sparsity)
        # print("xxx RWKV a lcollate_adjacency:",a.shape)
        # print(a)
        # print(a.shape)
        weight_mask = a.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))

        for layer, (G, R, M, L) in enumerate(
                zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers)):
            h = G(h, a)
            # print("xxx RWKV GNN:", h.shape)
            # print("######## ==gnn_layers ===h",h.shape, "===a",a.shape)

            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            h_readout, node_attn = R(h_bridge, node_axis=2)
            # print("######## ==readout_module ===h_readout",h_readout.shape, "===node_attn",node_attn.shape)

            # X = rearrange(h_bridge, 't b n c -> (t b) n c')#(batch*slice,roi, hidden) for modularity_loss
            # modularityloss += brain_modularity_loss2(X)
            # modularityloss +=  brain_modularity_loss(X)
            # modularityloss += brain_modularity_loss_cosine(X) # 使用cosine计算特征间差异性
            # 为了消融实验：先注释掉，with no Modularity,7/10
            # modularityloss += calculateloss(X) #不计算loss看会不会快一点


            A_pred = dot_product_decode(h)
            recon_loss = F.binary_cross_entropy(A_pred.view(-1).cuda(a.device), a.view(-1).cuda(a.device),
                                                       weight=weight_tensor.cuda(a.device))
            print("====== recon_loss   ===", recon_loss)
            reconstruct_loss += recon_loss

            # 修改，即去掉readout
            if self.token_parameter is not None: h_readout = torch.cat(
                [h_readout, self.token_parameter[layer].expand(-1, h_readout.shape[1], -1)])

            # print("h_readout ==",h_readout.shape)
            # print("node_attn ==", node_attn.shape)

            # print("h_bridge ==", h_bridge.shape)
            # print("h_readout ==", h_readout.shape)

            # h_state = M(h_readout) # 正确代码，执行了Mamba

            # logit = tt.mean(dim =1)
            # print("===logit",logit.shape)
            h_state = M(h_bridge.mean(dim =2)) # no Readout ,Only RWKV ，需要註釋掉

            # h_state = h_readout # 修改，即去掉RWKV层, no RWKV, 同時需要注釋  h_state = M(h_readout)
            # h_state = h_bridge   # 修改，即去掉readout no Readout and RWKV
            # print("h_state ==",h_state.shape)

            ortho_latent = rearrange(h_bridge, 't b n c -> (t b) n c')
            matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0, 2, 1))
            # reg_ortho += (matrix_inner/matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(num_nodes, device=matrix_inner.device)).triu().norm(dim=(1,2)).mean()

            max_vals = matrix_inner.max(-1)[0].unsqueeze(-1)
            reg_ortho += (matrix_inner /  (max_vals + epsilon) - torch.eye(num_nodes,device=matrix_inner.device)).triu().norm(dim=(1, 2)).mean()

            # print("xxxxxxxxxxxx=reg_ortho:",reg_ortho)
            # print("h_state ==",h_state[0].shape)
            # print("h_state ==",h_state[1].shape)
            # print("h_state ==",h_state[2].shape)

            latent = self.cls_token(h_state)
            # print("===latent",latent.shape)
            logit += self.dropout(L(latent)) #原始代码

            #将维度torch.Size([12, 246, 2])，降低到 torch.Size([12, 2])  #no RWKV ,no Readout 時需要下面這個代碼
            # tt = self.dropout(L(latent))
            # logit += tt.mean(dim =1)
            # print("===logit",logit.shape)

            attention['node-attention'].append(node_attn)
            attention['time-attention'].append(node_attn)
            latent_list.append(latent)

        logit = logit.squeeze(1)
        attention['node-attention'] = torch.stack(attention['node-attention'], dim=1).detach().cpu()
        attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()
        latent = torch.stack(latent_list, dim=1)

        # return logit, attention, latent, reg_ortho, modularityloss
        return logit, attention, latent, reg_ortho, reconstruct_loss


class LayerFusion(nn.Module):

    def __init__(self, num_views=2, fusion_type='weighted'):
        """
        :param fusion_type: include concatenate/average
        """
        super(LayerFusion, self).__init__()
        self.fusion_type = fusion_type
        self.num_views = num_views

        # define the attention weights for feature matrix and adjacent matrix
        # self.pai_fea = nn.Parameter(torch.ones(self.num_views) / self.num_views, requires_grad=True)
        # self.pai_adj = nn.Parameter(torch.ones(self.num_views) / self.num_views, requires_grad=True)
        #
        # self.pai_fea = nn.Parameter(torch.normal(mean=0.6, std=1, size=(self.num_views,)), requires_grad=True)
        # self.pai_adj = nn.Parameter(torch.normal(mean=0.6, std=1, size=(self.num_views,)), requires_grad=True)

        self.pai_fea = nn.Parameter(torch.rand(self.num_views), requires_grad=True)
        self.pai_adj = nn.Parameter(torch.rand(self.num_views), requires_grad=True)

    def forward(self, features, adjs):
        if self.fusion_type == "concatenate":
            combined_feature = torch.cat(features, dim=1)
        elif self.fusion_type == "average":
            pass
        elif self.fusion_type == "weighted":
            # 归一化权重，确保和为1
            norm_pai_fea = torch.softmax(self.pai_fea, dim=0)
            norm_pai_adj = torch.softmax(self.pai_adj, dim=0)

            # Sigmoid 可以将值映射到(0, 1) 区间
            # norm_pai_fea = torch.sigmoid(self.pai_fea)
            # norm_pai_adj = torch.sigmoid(self.pai_adj)

            # 使用温度参数进行权重归一化
            # norm_pai_fea = torch.softmax(self.pai_fea / self.temperature, dim=0)
            # norm_pai_adj = torch.softmax(self.pai_adj / self.temperature, dim=0)

            # 在计算时归一化权重：
            # norm_pai_fea = self.pai_fea / (self.pai_fea + self.pai_adj).sum()
            # norm_pai_adj = self.pai_adj / (self.pai_fea + self.pai_adj).sum()

            print("====exp_sum_pai_fea ===",norm_pai_fea)
            print("====exp_sum_pai_adj ===",norm_pai_adj)

            # combine the feature matrix
            combined_feature = sum(norm_pai_fea[i] * features[i] for i in range(self.num_views))

            # combine the adjacent matrix
            combined_adjacent = sum(norm_pai_adj[i] * adjs[i] for i in range(self.num_views))


        elif self.fusion_type == "weighted3":
            # combine the feature matrix
            exp_sum_pai_fea = 0
            for i in range(self.num_views):
                exp_sum_pai_fea += torch.exp(self.pai_fea[i])
            combined_feature = (torch.exp(self.pai_fea[0]) / exp_sum_pai_fea) * features[0]
            print("====exp_sum_pai_fea ===",(torch.exp(self.pai_fea[0]) / exp_sum_pai_fea))
            for i in range(1, self.num_views):
                combined_feature = combined_feature + (torch.exp(self.pai_fea[i]) / exp_sum_pai_fea) * features[i]

            # combine the adjacent matrix
            exp_sum_pai_adj = 0
            for i in range(self.num_views):
                exp_sum_pai_adj += torch.exp(self.pai_adj[i]) #下面这行需要解决稀疏张量的问题
            # combined_adjacent = (torch.exp(self.pai_adj[0]) / exp_sum_pai_adj) * adjs[0].coalesce() # adjs[0] 修改为 adjs[0].coalesce()
            combined_adjacent = (torch.exp(self.pai_adj[0]) / exp_sum_pai_adj) * adjs[0]
            print("====exp_sum_pai_adj ===",(torch.exp(self.pai_adj[0]) / exp_sum_pai_adj))
            for i in range(1, self.num_views):
                combined_adjacent = combined_adjacent + (torch.exp(self.pai_adj[i]) / exp_sum_pai_adj) * adjs[i]

        else:
            print("Please using a correct fusion type")
            exit()

        return combined_feature, combined_adjacent
class ModelRWKV_F(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, sparsity, dropout=0.5,
                 cls_token='sum', readout='sero', garo_upscale=1.0):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token == 'sum':
            self.cls_token = lambda x: x.sum(0)
        elif cls_token == 'mean':
            self.cls_token = lambda x: x.mean(0)
        elif cls_token == 'param':
            self.cls_token = lambda x: x[-1]
        else:
            raise
        if readout == 'garo':
            readout_module = ModuleGARO
        elif readout == 'sero':
            readout_module = ModuleSERO
        elif readout == 'mean':
            readout_module = ModuleMeanReadout
        elif readout == 'baro':
            readout_module = ModuleBARO
        elif readout == 'gsero':
            readout_module = ModuleCombinedAttention # ok

        else:
            raise

        self.token_parameter = nn.Parameter(
            torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token == 'param' else None

        self.num_classes = num_classes
        self.sparsity = sparsity

        # define modules
        self.percentile = Percentile()
        # self.timestamp_encoder = ModuleTimestamping(input_dim, hidden_dim, hidden_dim) #如果没有时间编码的话，这行可以注释掉
        # self.initial_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.initial_linear = nn.Linear(input_dim, hidden_dim)
        self.LL_layers = LayerFusion(2, 'weighted')

        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            # self.gnn_layers.append(GAT(nfeat=hidden_dim, nhid=hidden_dim,  dropout=0.0,nheads=2,alpha=0.2))
            # self.gnn_layers.append(GraphSAGE(hidden_dim, hidden_dim))

            # attention original OK
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))

            # self.readout_modules.append(readout_module(hidden_dim=hidden_dim))

            # for BARO
            # self.readout_modules.append(readout_module(hidden_dim=hidden_dim, num_heads=num_heads, num_time=num_time, input_dim=input_dim, dropout=0.1))

            # === original === ST
            # self.transformer_modules.append(
            #     ModuleTransformer(hidden_dim, 2 * hidden_dim, num_heads=num_heads, dropout=0.1))

            # === mamba ===
            # self.transformer_modules.append(ModuleMamba(hidden_dim, 2*hidden_dim))

            # == RWKVBlock ==  OK
            self.transformer_modules.append(RWKVBlock(input_dim = hidden_dim,layer_id= 0))

            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))

    def _collate_adjacency(self, a, sparsity, sparse=True):
        i_list = []
        v_list = []

        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > self.percentile(_a, 100 - sparsity))
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)

        # return torch.sparse.FloatTensor(_i, _v,(a.shape[0] * a.shape[1] * a.shape[2], a.shape[0] * a.shape[1] * a.shape[3]))

        # 创建一个全零的密集矩阵，并根据索引填充值 DFP-GNN  在RWKV_F中使用
        dense_shape = (a.shape[0] * a.shape[1] * a.shape[2], a.shape[0] * a.shape[1] * a.shape[3])
        dense_adj = torch.zeros(dense_shape, device=a.device)
        dense_adj[_i[0], _i[1]] = _v  # 根据索引填充值
        return dense_adj

    def forward(self, v, a, a2, t, sampling_endpoints):
        # assumes shape [minibatch x time x node x feature] for v
        # assumes shape [minibatch x time x node x node] for a
        # print("==== RWKV ===v ",v.shape, "a ",a.shape," t",t.shape,"sampling_endpoints ",sampling_endpoints)
        reconstruct_loss = 0.0
        logit = 0.0
        reg_ortho = 0.0
        epsilon = 1e-8

        attention = {'node-attention': [], 'time-attention': []}
        latent_list = []
        minibatch_size, num_timepoints, num_nodes = a.shape[:3]
        h = v

        h = rearrange(h, 'b t n c -> (b t n) c')
        h = self.initial_linear(h)

        h2 = rearrange(a2, 'b t n c -> (b t n) c')
        h2 = self.initial_linear(h2)

        hidden_list = []
        adjs = []
        hidden_list.append(h)
        hidden_list.append(h2)

        adjs.append(a)
        adjs.append(a2)#[18, 2, 246, 246]  weight_mask
        xxh, xxa = self.LL_layers(hidden_list, adjs)
        xxa = self._collate_adjacency(xxa,self.sparsity)
        weight_mask =xxa.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        for layer, (G, R, M, L) in enumerate(
                zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers)):
            h_new = G(xxh, xxa)#

            h_bridge = rearrange(h_new, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            h_readout, node_attn = R(h_bridge, node_axis=2)

            A_pred = dot_product_decode(h_new)
            recon_loss = F.binary_cross_entropy(A_pred.view(-1).cuda(a.device), xxa.view(-1).cuda(a.device),
                                                       weight=weight_tensor.cuda(a.device))
            reconstruct_loss += recon_loss
            # 修改，即去掉readout
            if self.token_parameter is not None: h_readout = torch.cat(
                [h_readout, self.token_parameter[layer].expand(-1, h_readout.shape[1], -1)])

            h_state = M(h_readout)

            ortho_latent = rearrange(h_bridge, 't b n c -> (t b) n c')
            matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0, 2, 1))
            # reg_ortho += (matrix_inner/matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(num_nodes, device=matrix_inner.device)).triu().norm(dim=(1,2)).mean()

            max_vals = matrix_inner.max(-1)[0].unsqueeze(-1)
            reg_ortho += (matrix_inner /  (max_vals + epsilon) - torch.eye(num_nodes,device=matrix_inner.device)).triu().norm(dim=(1, 2)).mean()

            latent = self.cls_token(h_state)
            # print("===latent",latent.shape)
            logit += self.dropout(L(latent)) #原始代码


            attention['node-attention'].append(node_attn)
            attention['time-attention'].append(node_attn)
            latent_list.append(latent)

        logit = logit.squeeze(1)
        attention['node-attention'] = torch.stack(attention['node-attention'], dim=1).detach().cpu()
        attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()
        latent = torch.stack(latent_list, dim=1)

        return logit, attention, latent, reg_ortho, reconstruct_loss
