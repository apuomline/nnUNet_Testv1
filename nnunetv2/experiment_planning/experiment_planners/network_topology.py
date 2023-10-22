from copy import deepcopy
import numpy as np


def get_shape_must_be_divisible_by(net_numpool_per_axis):
    return 2 ** np.array(net_numpool_per_axis)


def pad_shape(shape, must_be_divisible_by):
    """
    pads shape so that it is divisible by must_be_divisible_by
    :param shape:
    :param must_be_divisible_by:
    :return:
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(int)
    return new_shp


def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size, max_numpool):
    """
    this is the same as get_pool_and_conv_props_v2 from old nnunet

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :param max_numpool:
    :return:
    """
    """
    该函数用于计算UNet模型的池化和卷积属性。

输入参数包括：
- `spacing`：数据集的间距。
- `patch_size`：补丁的大小。
- `min_feature_map_size`：瓶颈层中特征图的最小边长。
- `max_numpool`：每个轴上的最大池化次数。

函数首先复制输入参数的值，然后初始化一些变量，包括池化操作的内核大小、卷积内核大小、每个轴上的池化次数和内核大小。

然后，它开始循环，直到无法进行更多池化操作。在每次循环中，函数会排除那些由于特征图大小的限制无法进行更多池化操作的轴。

接下来，函数会找到在最小间距的两倍范围内的轴，并将这些轴添加到可以进行池化操作的轴列表中。然后，函数会排除已经达到最大池化次数的轴。

如果只有一个轴可以进行池化操作，并且该轴的大小大于或等于3倍的最小特征图大小，则跳过此轮循环。否则，函数会退出循环。

如果没有轴可以进行池化操作，则函数会退出循环。

在找到可以进行池化操作的轴后，函数会初始化池化内核大小为2，并增加每个轴的池化次数。然后，函数会更新当前间距和大小，以反映池化操作的影响。

接下来，函数会计算池化操作的内核大小，并将其添加到`pool_op_kernel_sizes`列表中。然后，函数会将卷积内核大小添加到`conv_kernel_sizes`列表中。

最后，函数会计算补丁大小，以确保其可以被池化和卷积操作所处理。并返回池化和卷积属性、补丁大小和必须被整除的形状。
    """
    # todo review this code
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = [[1] * len(spacing)]
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim
    kernel_size = [1] * dim

    while True:
        # exclude axes that we cannot pool further because of min_feature_map_size constraint
        valid_axes_for_pool = [i for i in range(dim) if current_size[i] >= 2*min_feature_map_size]
        if len(valid_axes_for_pool) < 1:
            break

        spacings_of_axes = [current_spacing[i] for i in valid_axes_for_pool]

        # find axis that are within factor of 2 within smallest spacing
        min_spacing_of_valid = min(spacings_of_axes)
        valid_axes_for_pool = [i for i in valid_axes_for_pool if current_spacing[i] / min_spacing_of_valid < 2]

        # max_numpool constraint
        valid_axes_for_pool = [i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool]

        if len(valid_axes_for_pool) == 1:
            if current_size[valid_axes_for_pool[0]] >= 3 * min_feature_map_size:
                pass
            else:
                break
        if len(valid_axes_for_pool) < 1:
            break

        # now we need to find kernel sizes
        # kernel sizes are initialized to 1. They are successively set to 3 when their associated axis becomes within
        # factor 2 of min_spacing. Once they are 3 they remain 3
        for d in range(dim):
            if kernel_size[d] == 3:
                continue
            else:
                if current_spacing[d] / min(current_spacing) < 2:
                    kernel_size[d] = 3

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(deepcopy(kernel_size))
        #print(conv_kernel_sizes)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3]*dim)
    return num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, must_be_divisible_by
