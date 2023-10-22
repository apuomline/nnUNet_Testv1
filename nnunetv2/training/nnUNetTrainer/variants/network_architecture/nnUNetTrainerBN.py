from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn


class nnUNetTrainerBN(nnUNetTrainer):
    """
    这段代码是一个自定义的神经网络架构构建函数。
    它基于nnUNetTrainer类的build_network_architecture方法进行了扩展。
    首先，它从PlansManager中获取了数据集的标签管理器label_manager。
    然后，根据配置文件中的UNet_class_name属性选择要使用的分割网络类名。
    目前支持两种网络类名：PlainConvUNet和ResidualEncoderUNet。
    接下来，根据选择的网络类名，从映射字典mapping中获取对应的网络类。
    然后，根据网络类名选择对应的参数配置。这些参数包括是否使用偏置项conv_bias、
    使用的批归一化操作norm_op及其参数norm_op_kwargs、
    是否使用dropout操作dropout_op及其参数dropout_op_kwargs、
    非线性激活函数nonlin及其参数nonlin_kwargs。
    然后，根据配置文件中的参数，构建网络模型。模型的输入通道数为num_input_channels，
    总共有num_stages个阶段，每个阶段的特征数为UNet_base_num_features * 2^i（i为阶段索引），
    但不超过unet_max_num_features。卷积操作使用的卷积核大小为conv_kernel_sizes，
    池化操作使用的核大小为pool_op_kernel_sizes。
    模型的输出类别数为label_manager的num_segmentation_heads属性值。
    如果启用了深度监督（enable_deep_supervision为True），则模型会输出每个阶段的预测结果。
    最后，对模型应用He初始化权重的方法InitWeights_He，
    并根据网络类名为ResidualEncoderUNet的模型应用init_last_bn_before_add_to_0方法。
    总之，这段代码根据配置文件中的参数构建了一个指定类型的神经网络模型，并对模型进行了初始化。
    """
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)

        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)

        segmentation_network_class_name = configuration_manager.UNet_class_name
        mapping = {
            'PlainConvUNet': PlainConvUNet,
            'ResidualEncoderUNet': ResidualEncoderUNet
        }
        kwargs = {
            'PlainConvUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'ResidualEncoderUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            }
        }
        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                  'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                  'into either this ' \
                                                                  'function (get_network_from_plans) or ' \
                                                                  'the init of your nnUNetModule to accomodate that.'
        network_class = mapping[segmentation_network_class_name]

        conv_or_blocks_per_stage = {
            'n_conv_per_stage'
            if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
            'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
        }
        # network class name!!
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                    configuration_manager.unet_max_num_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        model.apply(InitWeights_He(1e-2))
        if network_class == ResidualEncoderUNet:
            model.apply(init_last_bn_before_add_to_0)
        return model
