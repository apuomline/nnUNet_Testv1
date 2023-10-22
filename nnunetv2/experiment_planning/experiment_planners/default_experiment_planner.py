import os.path
import shutil
from copy import deepcopy
from functools import lru_cache
from typing import List, Union, Tuple, Type

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.normalization.map_channel_name_to_normalization import get_normalization_scheme
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape, compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    get_filenames_of_train_images_and_targets

#nnUNet你看完了吗 你能从原理到代码讲清楚吗 会跑模型吗？
#还有很多好模型，你看了吗 原理你会吗？
class ExperimentPlanner(object):
    """
    这段代码定义了一个名为`ExperimentPlanner`的类，用于计划实验的参数和配置。

类的构造函数接受以下参数：
- `dataset_name_or_id`：数据集的名称或ID。
- `gpu_memory_target_in_gb`：目标GPU内存大小（以GB为单位），默认为8GB。
- `preprocessor_name`：预处理器的名称，默认为'DefaultPreprocessor'。
- `plans_name`：计划的名称，默认为'nnUNetPlans'。
- `overwrite_target_spacing`：要覆盖的目标间距，默认为None。
- `suppress_transpose`：一个布尔值，指示是否禁止转置，默认为False。

构造函数首先将`dataset_name_or_id`转换为数据集名称，并设置一些属性，如`suppress_transpose`、`raw_dataset_folder`和`preprocessed_folder`。

然后，它加载数据集的`dataset.json`文件，并获取训练图像和目标的文件名。

接下来，它加载数据集的指纹信息，并设置一些与UNet模型相关的属性，如`UNet_base_num_features`、`UNet_class`和`UNet_reference_val_3d`。

然后，它设置一些与GPU内存和特征图大小相关的属性，如`UNet_vram_target_GB`、`UNet_max_features_2d`和`UNet_max_features_3d`。

最后，它设置一些与预处理和计划相关的属性，如`preprocessor_name`、`plans_identifier`和`overwrite_target_spacing`。
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        """
        overwrite_target_spacing only affects 3d_fullres! (but by extension 3d_lowres which starts with fullres may
        also be affected
        """

        self.dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        self.suppress_transpose = suppress_transpose
        self.raw_dataset_folder = join(nnUNet_raw, self.dataset_name)
        preprocessed_folder = join(nnUNet_preprocessed, self.dataset_name)
        self.dataset_json = load_json(join(self.raw_dataset_folder, 'dataset.json'))
        self.dataset = get_filenames_of_train_images_and_targets(self.raw_dataset_folder, self.dataset_json)

        # load dataset fingerprint
        if not isfile(join(preprocessed_folder, 'dataset_fingerprint.json')):
            raise RuntimeError('Fingerprint missing for this dataset. Please run nnUNet_extract_dataset_fingerprint')

        self.dataset_fingerprint = load_json(join(preprocessed_folder, 'dataset_fingerprint.json'))

        self.anisotropy_threshold = ANISO_THRESHOLD

        self.UNet_base_num_features = 32
        self.UNet_class = PlainConvUNet
        # the following two numbers are really arbitrary and were set to reproduce nnU-Net v1's configurations as
        # much as possible
        self.UNet_reference_val_3d = 560000000  # 455600128  550000000
        self.UNet_reference_val_2d = 85000000  # 83252480
        self.UNet_reference_com_nfeatures = 32
        self.UNet_reference_val_corresp_GB = 8
        self.UNet_reference_val_corresp_bs_2d = 12
        self.UNet_reference_val_corresp_bs_3d = 2
        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_featuremap_min_edge_length = 4
        self.UNet_blocks_per_stage_encoder = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        self.UNet_blocks_per_stage_decoder = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        self.UNet_min_batch_size = 2
        self.UNet_max_features_2d = 512
        self.UNet_max_features_3d = 320

        self.lowres_creation_threshold = 0.25  # if the patch size of fullres is less than 25% of the voxels in the
        # median shape then we need a lowres config as well

        self.preprocessor_name = preprocessor_name
        self.plans_identifier = plans_name
        self.overwrite_target_spacing = overwrite_target_spacing
        assert overwrite_target_spacing is None or len(overwrite_target_spacing), 'if overwrite_target_spacing is ' \
                                                                                  'used then three floats must be ' \
                                                                                  'given (as list or tuple)'
        assert overwrite_target_spacing is None or all([isinstance(i, float) for i in overwrite_target_spacing]), \
            'if overwrite_target_spacing is used then three floats must be given (as list or tuple)'

        self.plans = None

    def determine_reader_writer(self):
        example_image = self.dataset[self.dataset.keys().__iter__().__next__()]['images'][0]
        return determine_reader_writer_from_dataset_json(self.dataset_json, example_image)

    @staticmethod
    @lru_cache(maxsize=None)
    def static_estimate_VRAM_usage(patch_size: Tuple[int],
                                   n_stages: int,
                                   strides: Union[int, List[int], Tuple[int, ...]],
                                   UNet_class: Union[Type[PlainConvUNet], Type[ResidualEncoderUNet]],
                                   num_input_channels: int,
                                   features_per_stage: Tuple[int],
                                   blocks_per_stage_encoder: Union[int, Tuple[int]],
                                   blocks_per_stage_decoder: Union[int, Tuple[int]],
                                   num_labels: int):
        """
        这段代码定义了一个名为`static_estimate_VRAM_usage`的函数，用于估计UNet模型的显存使用情况。

函数接受以下参数：
- `patch_size`：输入图像的补丁大小。
- `n_stages`：UNet模型的阶段数。
- `strides`：UNet模型的步幅。
- `UNet_class`：UNet模型的类别。
- `num_input_channels`：输入图像的通道数。
- `features_per_stage`：每个阶段的特征数量。
- `blocks_per_stage_encoder`：编码器每个阶段的块数。
- `blocks_per_stage_decoder`：解码器每个阶段的块数。
- `num_labels`：输出标签的数量。

函数首先根据输入参数创建一个UNet模型实例。

然后，它调用UNet模型的`compute_conv_feature_map_size`方法，计算给定补丁大小的特征图大小。

最后，它返回特征图的大小，用于估计显存使用情况。
        """
        """
        Works for PlainConvUNet, ResidualEncoderUNet
        """
        dim = len(patch_size)
        conv_op = convert_dim_to_conv_op(dim)
        norm_op = get_matching_instancenorm(conv_op)
        net = UNet_class(num_input_channels, n_stages,
                         features_per_stage,
                         conv_op,
                         3,
                         strides,
                         blocks_per_stage_encoder,
                         num_labels,
                         blocks_per_stage_decoder,
                         norm_op=norm_op)
        return net.compute_conv_feature_map_size(patch_size)

    def determine_resampling(self, *args, **kwargs):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

        determine_resampling is called within get_plans_for_configuration to allow for different functions for each
        configuration
        """
        resampling_data = resample_data_or_seg_to_shape
        resampling_data_kwargs = {
            "is_seg": False,
            "order": 3,
            "order_z": 0,
            "force_separate_z": None,
        }
        resampling_seg = resample_data_or_seg_to_shape
        resampling_seg_kwargs = {
            "is_seg": True,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        """
        function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
        used as target. current_spacing and new_spacing are merely there in case we want to use it somehow

        determine_segmentation_softmax_export_fn is called within get_plans_for_configuration to allow for different
        functions for each configuration

        """
        resampling_fn = resample_data_or_seg_to_shape
        resampling_fn_kwargs = {
            "is_seg": False,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
        return resampling_fn, resampling_fn_kwargs
#设置目标体素间距-->为了处理体素间距各向异性
    def determine_fullres_target_spacing(self) -> np.ndarray:
        """
        per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
        and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

        For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
        (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
        resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
        impact performance (due to the low number of slices).
        """
        """
            这段代码定义了一个名为`determine_fullres_target_spacing`的方法，用于确定完整分辨率的目标间距。
    方法首先检查是否指定了`overwrite_target_spacing`，如果指定了，则返回指定的目标间距。

    否则，它获取数据集的间距信息和裁剪后的形状信息。然后，它计算间距的50th百分位数作为初始目标间距。

    接下来，它检查是否存在具有不均匀间距的数据集。具有不均匀间距的数据集通常在某个轴上具有较低的分辨率，并且在该轴上的体素数量较少。为了避免插值引起的伪影，它将该轴的目标间距设置为其他轴上最大间距的值。

    最后，它返回确定的目标间距。
        """
        if self.overwrite_target_spacing is not None:
            return np.array(self.overwrite_target_spacing)

        spacings = self.dataset_fingerprint['spacings']
        sizes = self.dataset_fingerprint['shapes_after_crop']

        target = np.percentile(np.vstack(spacings), 50, 0)

        # todo sizes_after_resampling = [compute_new_shape(j, i, target) for i, j in zip(spacings, sizes)]

        target_size = np.percentile(np.vstack(sizes), 50, 0)
        # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
        # the following properties:
        # - one axis which much lower resolution than the others
        # - the lowres axis has much less voxels than the others
        # - (the size in mm of the lowres axis is also reduced)
        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (self.anisotropy_threshold * max(other_spacings))
        has_aniso_voxels = target_size[worst_spacing_axis] * self.anisotropy_threshold < min(other_sizes)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target

    def determine_normalization_scheme_and_whether_mask_is_used_for_norm(self) -> Tuple[List[str], List[bool]]:
        if 'channel_names' not in self.dataset_json.keys():
            print('WARNING: "modalities" should be renamed to "channel_names" in dataset.json. This will be '
                  'enforced soon!')
        modalities = self.dataset_json['channel_names'] if 'channel_names' in self.dataset_json.keys() else \
            self.dataset_json['modality']
        normalization_schemes = [get_normalization_scheme(m) for m in modalities.values()]
        if self.dataset_fingerprint['median_relative_size_after_cropping'] < (3 / 4.):
            use_nonzero_mask_for_norm = [i.leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true for i in
                                         normalization_schemes]
        else:
            use_nonzero_mask_for_norm = [False] * len(normalization_schemes)
            assert all([i in (True, False) for i in use_nonzero_mask_for_norm]), 'use_nonzero_mask_for_norm must be ' \
                                                                                 'True or False and cannot be None'
        normalization_schemes = [i.__name__ for i in normalization_schemes]
        return normalization_schemes, use_nonzero_mask_for_norm

    def determine_transpose(self):
        if self.suppress_transpose:
            return [0, 1, 2], [0, 1, 2]

        # todo we should use shapes for that as well. Not quite sure how yet
        target_spacing = self.determine_fullres_target_spacing()

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        transpose_forward = [max_spacing_axis] + remaining_axes
        transpose_backward = [np.argwhere(np.array(transpose_forward) == i)[0][0] for i in range(3)]
        return transpose_forward, transpose_backward

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...], List[int]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float) -> dict:
    #网络架构设计
        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        # print(spacing, median_shape, approximate_n_voxels_dataset)
        # find an initial patch size
        # we first use the spacing to get an aspect ratio
        tmp = 1 / np.array(spacing)

        # we then upscale it so that it initially is certainly larger than what we need (rescale to have the same
        # volume as a patch of size 256 ** 3)
        # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
        # ideal because large initial patch sizes increase computation time because more iterations in the while loop
        # further down may be required.
        if len(spacing) == 3:
            initial_patch_size = [round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]
        elif len(spacing) == 2:
            initial_patch_size = [round(i) for i in tmp * (2048 ** 2 / np.prod(tmp)) ** (1 / 2)]
        else:
            raise RuntimeError()

        # clip initial patch size to median_shape. It makes little sense to have it be larger than that. Note that
        # this is different from how nnU-Net v1 does it!
        # todo patch size can still get too large because we pad the patch size to a multiple of 2**n
        initial_patch_size = np.array([min(i, j) for i, j in zip(initial_patch_size, median_shape[:len(spacing)])])

        # use that to get the network topology. Note that this changes the patch_size depending on the number of
        # pooling operations (must be divisible by 2**num_pool in each axis)
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
                                                             self.UNet_featuremap_min_edge_length,
                                                             999999)

        # now estimate vram consumption
        """
        根据设定的显存使用的阈值-->确定最合适的patch_size--》更新网络架构
        """

        """
        这段代码重新计算了网络拓扑。以下是代码的逐步解释：
        1. 使用`get_pool_and_conv_props`方法重新计算网络拓扑。
        该方法使用以下参数：间距，块大小，特征图最小边长，999999。
        2. 更新`num_stages`为`pool_op_kernel_sizes`的长度。
        3. 使用`static_estimate_VRAM_usage`方法重新计算估计值。
        该方法使用以下参数：块大小，池化操作的数量，池化操作的内核大小，
        UNet类，通道数，特征图数量，编码器和解码器的块数，标签数。
        4. 确定批量大小。如果while循环执行过，则使用`UNet_min_batch_size`；
        否则，使用额外的VRAM余量来增加批量大小。
        5. 将批量大小限制在覆盖最多5%的整个数据集的范围内，以防止过拟合。
        但批量大小不能小于`UNet_min_batch_size`。
        6. 使用`determine_resampling`方法确定重采样数据和标签的参数。
        7. 使用`determine_segmentation_softmax_export_fn`方法确定分割的softmax输出函数和参数。
        8. 使用`determine_normalization_scheme_and_whether_mask_is_used_for_norm`
        方法确定归一化方案和是否使用掩码进行归一化。
        总的来说，这段代码的主要目的是重新计算网络拓扑，
        并确定批量大小、重采样参数、分割输出函数和参数以及归一化方案。
        """
        num_stages = len(pool_op_kernel_sizes)
        estimate = self.static_estimate_VRAM_usage(tuple(patch_size),
                                                   num_stages,
                                                   tuple([tuple(i) for i in pool_op_kernel_sizes]),
                                                   self.UNet_class,
                                                   len(self.dataset_json['channel_names'].keys()
                                                       if 'channel_names' in self.dataset_json.keys()
                                                       else self.dataset_json['modality'].keys()),
                                                   tuple([min(self.UNet_max_features_2d if len(patch_size) == 2 else
                                                              self.UNet_max_features_3d,
                                                              self.UNet_reference_com_nfeatures * 2 ** i) for
                                                          i in range(len(pool_op_kernel_sizes))]),
                                                   self.UNet_blocks_per_stage_encoder[:num_stages],
                                                   self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                                                   len(self.dataset_json['labels'].keys()))

        # how large is the reference for us here (batch size etc)?
        # adapt for our vram target
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        while estimate > reference:
            # print(patch_size)
            # patch size seems to be too large, so we need to reduce it. Reduce the axis that currently violates the
            # aspect ratio the most (that is the largest relative to median shape)
            axis_to_be_reduced = np.argsort(patch_size / median_shape[:len(spacing)])[-1]

            # we cannot simply reduce that axis by shape_must_be_divisible_by[axis_to_be_reduced] because this
            # may cause us to skip some valid sizes, for example shape_must_be_divisible_by is 64 for a shape of 256.
            # If we subtracted that we would end up with 192, skipping 224 which is also a valid patch size
            # (224 / 2**5 = 7; 7 < 2 * self.UNet_featuremap_min_edge_length(4) so it's valid). So we need to first
            # subtract shape_must_be_divisible_by, then recompute it and then subtract the
            # recomputed shape_must_be_divisible_by. Annoying.
            tmp = deepcopy(patch_size)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by = \
                get_pool_and_conv_props(spacing, tmp,
                                        self.UNet_featuremap_min_edge_length,
                                        999999)
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            # now recompute topology
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
            shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
                                                                 self.UNet_featuremap_min_edge_length,
                                                                 999999)

            num_stages = len(pool_op_kernel_sizes)
            estimate = self.static_estimate_VRAM_usage(tuple(patch_size),
                                                       num_stages,
                                                       tuple([tuple(i) for i in pool_op_kernel_sizes]),
                                                       self.UNet_class,
                                                       len(self.dataset_json['channel_names'].keys()
                                                           if 'channel_names' in self.dataset_json.keys()
                                                           else self.dataset_json['modality'].keys()),
                                                       tuple([min(self.UNet_max_features_2d if len(patch_size) == 2 else
                                                                  self.UNet_max_features_3d,
                                                                  self.UNet_reference_com_nfeatures * 2 ** i) for
                                                              i in range(len(pool_op_kernel_sizes))]),
                                                       self.UNet_blocks_per_stage_encoder[:num_stages],
                                                       self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                                                       len(self.dataset_json['labels'].keys()))

        # alright now let's determine the batch size. This will give self.UNet_min_batch_size if the while loop was
        # executed. If not, additional vram headroom is used to increase batch size
        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        batch_size = round((reference / estimate) * ref_bs)

        # we need to cap the batch size to cover at most 5% of the entire dataset. Overfitting precaution. We cannot
        # go smaller than self.UNet_min_batch_size though
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * 0.05 / np.prod(patch_size, dtype=np.float64))
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()
        num_stages = len(pool_op_kernel_sizes)
        plan = {
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': batch_size,
            'patch_size': patch_size,
            'median_image_size_in_voxels': median_shape,
            'spacing': spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': mask_is_used_for_norm,
            'UNet_class_name': self.UNet_class.__name__,
            'UNet_base_num_features': self.UNet_base_num_features,
            'n_conv_per_stage_encoder': self.UNet_blocks_per_stage_encoder[:num_stages],
            'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
            'num_pool_per_axis': network_num_pool_per_axis,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
            'unet_max_num_features': self.UNet_max_features_3d if len(spacing) == 3 else self.UNet_max_features_2d,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
        }
        return plan
#根据dataset_fingerprint确定模型的配置信息(2d,3d_fullers,3d_lowers)
#只有当3d_fullres 3d_lowers全不为空时，才会生成3d_cascade配置

    def plan_experiment(self):
        """

        """
        """
        MOVE EVERYTHING INTO THE PLANS. MAXIMUM FLEXIBILITY

        Ideally I would like to move transpose_forward/backward into the configurations so that this can also be done
        differently for each configuration but this would cause problems with identifying the correct axes for 2d. There
        surely is a way around that but eh. I'm feeling lazy and featuritis must also not be pushed to the extremes.

        So for now if you want a different transpose_forward/backward you need to create a new planner. Also not too
        hard.
        planer用来干什么的？
        """

        # first get transpose(这里的两种transpose方式哪里用到？)
        transpose_forward, transpose_backward = self.determine_transpose()

        # get fullres spacing and transpose it
        fullres_spacing = self.determine_fullres_target_spacing()
        fullres_spacing_transposed = fullres_spacing[transpose_forward]

        # get transposed new median shape (what we would have after resampling)
        new_shapes = [compute_new_shape(j, i, fullres_spacing) for i, j in
                      zip(self.dataset_fingerprint['spacings'], self.dataset_fingerprint['shapes_after_crop'])]
        new_median_shape = np.median(new_shapes, 0)
        new_median_shape_transposed = new_median_shape[transpose_forward]

        approximate_n_voxels_dataset = float(np.prod(new_median_shape_transposed, dtype=np.float64) *
                                             self.dataset_json['numTraining'])
        # only run 3d if this is a 3d dataset
        if new_median_shape_transposed[0] != 1:
            plan_3d_fullres = self.get_plans_for_configuration(fullres_spacing_transposed,
                                                               new_median_shape_transposed,
                                                               self.generate_data_identifier('3d_fullres'),
                                                               approximate_n_voxels_dataset)
            # maybe add 3d_lowres as well
            patch_size_fullres = plan_3d_fullres['patch_size']
            median_num_voxels = np.prod(new_median_shape_transposed, dtype=np.float64)
            num_voxels_in_patch = np.prod(patch_size_fullres, dtype=np.float64)

            plan_3d_lowres = None
            lowres_spacing = deepcopy(plan_3d_fullres['spacing'])

            spacing_increase_factor = 1.03  # used to be 1.01 but that is slow with new GPU memory estimation!

            while num_voxels_in_patch / median_num_voxels < self.lowres_creation_threshold:
                # we incrementally increase the target spacing. We start with the anisotropic axis/axes until it/they
                # is/are similar (factor 2) to the other ax(i/e)s.
                max_spacing = max(lowres_spacing)
                if np.any((max_spacing / lowres_spacing) > 2):
                    lowres_spacing[(max_spacing / lowres_spacing) > 2] *= spacing_increase_factor
                else:
                    lowres_spacing *= spacing_increase_factor
                median_num_voxels = np.prod(plan_3d_fullres['spacing'] / lowres_spacing * new_median_shape_transposed,
                                            dtype=np.float64)
                # print(lowres_spacing)
                plan_3d_lowres = self.get_plans_for_configuration(lowres_spacing,
                                                                  [round(i) for i in plan_3d_fullres['spacing'] /
                                                                   lowres_spacing * new_median_shape_transposed],
                                                                  self.generate_data_identifier('3d_lowres'),
                                                                  float(np.prod(median_num_voxels) *
                                                                        self.dataset_json['numTraining']))
                num_voxels_in_patch = np.prod(plan_3d_lowres['patch_size'], dtype=np.int64)
                print(f'Attempting to find 3d_lowres config. '
                      f'\nCurrent spacing: {lowres_spacing}. '
                      f'\nCurrent patch size: {plan_3d_lowres["patch_size"]}. '
                      f'\nCurrent median shape: {plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed}')
            if plan_3d_lowres is not None:
                plan_3d_lowres['batch_dice'] = False
                plan_3d_fullres['batch_dice'] = True
            else:
                plan_3d_fullres['batch_dice'] = False
        else:
            plan_3d_fullres = None
            plan_3d_lowres = None

        # 2D configuration
        plan_2d = self.get_plans_for_configuration(fullres_spacing_transposed[1:],
                                                   new_median_shape_transposed[1:],
                                                   self.generate_data_identifier('2d'), approximate_n_voxels_dataset)
        plan_2d['batch_dice'] = True

        print('2D U-Net configuration:')
        print(plan_2d)
        print()

        """
        中值体素，中值大小
        """
        # median spacing and shape, just for reference when printing the plans
        median_spacing = np.median(self.dataset_fingerprint['spacings'], 0)[transpose_forward]
        median_shape = np.median(self.dataset_fingerprint['shapes_after_crop'], 0)[transpose_forward]

        # instead of writing all that into the plans we just copy the original file. More files, but less crowded
        # per file.
        shutil.copy(join(self.raw_dataset_folder, 'dataset.json'),
                    join(nnUNet_preprocessed, self.dataset_name, 'dataset.json'))

        # json is stupid and I hate it... "Object of type int64 is not JSON serializable" -> my ass
        plans = {
            'dataset_name': self.dataset_name,
            'plans_name': self.plans_identifier,
            'original_median_spacing_after_transp': [float(i) for i in median_spacing],
            'original_median_shape_after_transp': [int(round(i)) for i in median_shape],
            'image_reader_writer': self.determine_reader_writer().__name__,
            'transpose_forward': [int(i) for i in transpose_forward],
            'transpose_backward': [int(i) for i in transpose_backward],
            'configurations': {'2d': plan_2d},
            'experiment_planner_used': self.__class__.__name__,
            'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': self.dataset_fingerprint[
                'foreground_intensity_properties_per_channel']
        }

        if plan_3d_lowres is not None:
            plans['configurations']['3d_lowres'] = plan_3d_lowres
            if plan_3d_fullres is not None:
                plans['configurations']['3d_lowres']['next_stage'] = '3d_cascade_fullres'
            print('3D lowres U-Net configuration:')
            print(plan_3d_lowres)
            print()
        if plan_3d_fullres is not None:
            plans['configurations']['3d_fullres'] = plan_3d_fullres
            print('3D fullres U-Net configuration:')
            print(plan_3d_fullres)
            print()
            if plan_3d_lowres is not None:
                plans['configurations']['3d_cascade_fullres'] = {
                    'inherits_from': '3d_fullres',
                    'previous_stage': '3d_lowres'
                }

        self.plans = plans
        self.save_plans(plans)
        return plans

    def save_plans(self, plans):
        recursive_fix_for_json_export(plans)

        plans_file = join(nnUNet_preprocessed, self.dataset_name, self.plans_identifier + '.json')

        # we don't want to overwrite potentially existing custom configurations every time this is executed. So let's
        # read the plans file if it already exists and keep any non-default configurations
        if isfile(plans_file):
            old_plans = load_json(plans_file)
            old_configurations = old_plans['configurations']
            for c in plans['configurations'].keys():
                if c in old_configurations.keys():
                    del (old_configurations[c])
            plans['configurations'].update(old_configurations)

        maybe_mkdir_p(join(nnUNet_preprocessed, self.dataset_name))
        save_json(plans, plans_file, sort_keys=False)
        print('Plans were saved to %s' % join(nnUNet_preprocessed, self.dataset_name, self.plans_identifier + '.json'))

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but differnet plans file can have configurations with the
        same name. In order to distinguish the assiciated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        return self.plans_identifier + '_' + configuration_name

    def load_plans(self, fname: str):
        self.plans = load_json(fname)


if __name__ == '__main__':
    ExperimentPlanner(2, 8).plan_experiment()
