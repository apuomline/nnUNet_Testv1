import torch
from torch import nn, Tensor
import numpy as np

"""
这段代码定义了两个PyTorch的损失函数类：
RobustCrossEntropyLoss和TopKLoss，它们都继承自nn.CrossEntropyLoss。
RobustCrossEntropyLoss是一个兼容性层，
因为它的目标张量是浮点数并且有一个额外的维度。
它的前向方法接受输入input和目标target张量作为参数。
如果目标张量target的形状与输入张量input的形状相同，
则将目标张量的第二个维度删除，确保目标张量的形状为(batch_size, )。
然后调用nn.CrossEntropyLoss的前向方法，将输入张量和目标张量的整数值作为参数，并返回损失值。
TopKLoss是一个基于RobustCrossEntropyLoss的损失函数，
它在计算交叉熵损失之前，使用torch.topk函数选择输入张量中的前k%（k是通过构造函数传递的参数）最大值。
这个操作可以帮助模型更加关注重要的像素，从而提高模型性能。
TopKLoss的前向方法接受输入input和目标target张量作为参数。
如果目标张量target的形状与输入张量input的形状相同，则将目标张量的第二个维度删除，
确保目标张量的形状为(batch_size, )。然后调用RobustCrossEntropyLoss的前向方法，返回交叉熵损失。
接着，使用torch.topk函数选择前k%的最大值，并取平均值作为最终的损失值。
"""
#这里为什么要将第二个维度删除？
class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

