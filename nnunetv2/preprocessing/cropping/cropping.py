import numpy as np


# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice


def create_nonzero_mask(data):#将图像中不需要的地方 用零掩码覆盖掉
    """

    :param data:
    :return: the mask is True where the data is nonzero

    data在传入进来时，不需要的地方的就已经被标0了？

    在图像分割后，可能会出现一些孔洞。需要进行填补。
    """
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    #针对于2维或者三维模型
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)#data.shape[1:]代表什么？？
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    #binary_fill_holes代码如何编写的 以及代码作用
    return nonzero_mask


#现在该如何做？？？、
"""
网络代码过多，如何看？？是将整个代码看懂后，再做还是扣去数据预处理与数据增强代码？
不同的网络架构，代码是不容易嫁接的。
不信了，就看nnUnet整体代码，不可能看不懂。
"""
def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    代码中的分割图像用来干什么的？
    这段代码实现了将数据和分割图像裁剪到非零区域的功能。
    
    具体解释如下：

    1. `crop_to_nonzero`函数接受三个参数：`data`表示输入数据，`seg`表示分割图像（可选），`nonzero_label`表示非零区域的标签。
    2. 首先调用`create_nonzero_mask`函数生成一个与输入数据形状相同的非零区域掩码。
    3. 然后调用`get_bbox_from_mask`函数根据非零区域掩码计算出一个边界框（bounding box）。
    4. 使用`bounding_box_to_slice`函数将边界框转换为切片（slice）对象。
    5. 将输入数据根据切片对象进行裁剪，得到裁剪后的数据。
    6. 如果分割图像`seg`不为空，则也根据切片对象对分割图像进行裁剪。
    7. 将非零区域掩码也根据切片对象进行裁剪，并添加一个新的维度。
    8. 如果分割图像`seg`不为空，则将分割图像中值为0且非零区域掩码为False的像素值赋为`nonzero_label`。
    9. 如果分割图像`seg`为空，则将非零区域掩码转换为`int8`类型的数组，并将值为0的像素赋为`nonzero_label`，将值大于0的像素赋为0。
    10. 最后返回裁剪后的数据、分割图像和边界框。
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)

    slicer = bounding_box_to_slice(bbox)
    data = data[tuple([slice(None), *slicer])]

    if seg is not None:
        seg = seg[tuple([slice(None), *slicer])]

    nonzero_mask = nonzero_mask[slicer][None]
    if seg is not None:
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(np.int8)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox


