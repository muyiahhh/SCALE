"""图卷积层定义"""

import torch
import math
import torch.nn as nn

from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    图卷积层 (Graph Convolutional Layer)

    功能：在图结构上进行卷积操作，聚合节点的邻居信息
    参考论文：https://arxiv.org/abs/1609.02907 (Kipf & Welling, ICLR 2017)

    核心原理：
    对于图中的每个节点，GCN通过以下方式更新其特征：
    H^(l+1) = σ(A * H^(l) * W^(l))
    其中：
    - H^(l): 第l层的节点特征矩阵 (n_nodes, in_features)
    - A: 归一化后的邻接矩阵 (n_nodes, n_nodes)
    - W^(l): 可学习的权重矩阵 (in_features, out_features)
    - σ: 激活函数

    在SafeDrug中的应用：
    - 用于GAMENet模型中建模药物之间的关系图（DDI图和EHR共现图）
    - 通过图卷积聚合药物之间的相互作用信息
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        初始化图卷积层

        参数:
            in_features (int): 输入特征维度
            out_features (int): 输出特征维度
            bias (bool): 是否使用偏置项，默认True
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重和偏置参数，使用均匀分布 U(-stdv, stdv)"""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        前向传播函数

        参数:
            input (Tensor): 节点特征矩阵，shape: (n_nodes, in_features)
            adj (Tensor): 归一化后的邻接矩阵，shape: (n_nodes, n_nodes)

        返回:
            output (Tensor): 更新后的节点特征，shape: (n_nodes, out_features)
        """
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """打印时显示层的信息"""
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
