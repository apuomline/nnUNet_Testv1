from typing import Tuple, Union, List

from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np


class DownsampleSegForDSTransform2(AbstractTransform):
    """
    这段代码定义了一个名为DownsampleSegForDSTransform2的类，它是AbstractTransform的子类。

该类用于将输入的分割数据进行下采样，生成多个分辨率的分割结果。具体来说，它接收以下参数：

- ds_scales: 下采样因子的列表或元组。每个因子指定了一个深度监督输出相对于原始数据的分辨率，例如0.25表示原始形状的四分之一。ds_scales也可以是一个元组的元组，例如((1, 1, 1), (0.5, 0.5, 0.5))，用于独立指定每个轴的下采样因子。
- order: 下采样时使用的插值阶数，默认为0。
- input_key: 输入数据在数据字典中的键，默认为"seg"。
- output_key: 输出数据在数据字典中的键，默认为"seg"。
- axes: 需要进行下采样的轴的索引元组，默认为None，表示对除了前两个轴以外的所有轴进行下采样。

该类的__call__方法实现了具体的下采样操作。首先根据输入数据的形状确定需要进行下采样的轴，然后遍历每个下采样因子。对于每个因子，如果所有轴的下采样因子都为1，则直接将输入数据添加到输出列表中；否则，根据下采样因子计算新的形状，并创建一个与新形状相同的全零数组。然后，遍历输入数据的每个通道和批次，将每个通道的分割结果进行插值，得到新的分割结果，并将其添加到输出列表中。最后，将输出列表添加到数据字典中，并返回数据字典。

总之，该类实现了将输入的分割数据按照指定的下采样因子进行下采样的功能，并将下采样结果存储在数据字典中。
    """
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self, ds_scales: Union[List, Tuple],
                 order: int = 0, input_key: str = "seg",
                 output_key: str = "seg", axes: Tuple[int] = None):
        """
        Downscales data_dict[input_key] according to ds_scales. Each entry in ds_scales specified one deep supervision
        output and its resolution relative to the original data, for example 0.25 specifies 1/4 of the original shape.
        ds_scales can also be a tuple of tuples, for example ((1, 1, 1), (0.5, 0.5, 0.5)) to specify the downsampling
        for each axis independently
        """
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        if self.axes is None:
            axes = list(range(2, len(data_dict[self.input_key].shape)))
        else:
            axes = self.axes

        output = []
        for s in self.ds_scales:
            if not isinstance(s, (tuple, list)):
                s = [s] * len(axes)
            else:
                assert len(s) == len(axes), f'If ds_scales is a tuple for each resolution (one downsampling factor ' \
                                            f'for each axis) then the number of entried in that tuple (here ' \
                                            f'{len(s)}) must be the same as the number of axes (here {len(axes)}).'

            if all([i == 1 for i in s]):
                output.append(data_dict[self.input_key])
            else:
                new_shape = np.array(data_dict[self.input_key].shape).astype(float)
                for i, a in enumerate(axes):
                    new_shape[a] *= s[i]
                new_shape = np.round(new_shape).astype(int)
                out_seg = np.zeros(new_shape, dtype=data_dict[self.input_key].dtype)
                for b in range(data_dict[self.input_key].shape[0]):
                    for c in range(data_dict[self.input_key].shape[1]):
                        out_seg[b, c] = resize_segmentation(data_dict[self.input_key][b, c], new_shape[2:], self.order)
                output.append(out_seg)
        data_dict[self.output_key] = output
        return data_dict
