from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn



"""
这段代码定义了一个函数 `get_network_from_plans`，用于根据给定的配置和数据集信息构建一个网络模型。

函数的输入参数包括：
- `plans_manager`：一个 `PlansManager` 对象，用于管理训练计划。
- `dataset_json`：一个包含数据集信息的字典。
- `configuration_manager`：一个 `ConfigurationManager` 对象，用于管理配置信息。
- `num_input_channels`：输入图像的通道数。
- `deep_supervision`：一个布尔值，表示是否使用深度监督，默认为 `True`。

函数的主要步骤如下：

1. 获取网络的阶段数 `num_stages`，即配置中的卷积核大小列表的长度。
2. 获取卷积操作的维度 `dim`，以及根据维度获取对应的卷积操作 `conv_op`。
3. 获取标签管理器 `label_manager`，通过调用 `plans_manager` 的 `get_label_manager` 方法，
并传入数据集信息。
4. 获取分割网络的类名 `segmentation_network_class_name`，
通过访问配置管理器的 `UNet_class_name` 属性。
5. 定义一个字典 `mapping`，用于将网络类名映射到对应的网络类。
6. 定义一个字典 `kwargs`，用于存储不同网络类的初始化参数。
7. 确保分割网络的类名在 `mapping` 的键中，如果不在则抛出异常。
8. 根据分割网络的类名从 `mapping` 中获取对应的网络类。
9. 定义一个字典 `conv_or_blocks_per_stage`，用于存储不同网络类的卷积或块数。
10. 创建网络模型，通过调用网络类的构造函数，并传入以下参数：
    - `input_channels`：输入图像的通道数。
    - `n_stages`：网络的阶段数。
    - `features_per_stage`：每个阶段的特征数列表。
    - `conv_op`：卷积操作。
    - `kernel_sizes`：卷积核大小列表。
    - `strides`：池化操作的核大小列表。
    - `num_classes`：分割头的数量。
    - `deep_supervision`：是否使用深度监督。
    - `**conv_or_blocks_per_stage`：卷积或块数的参数。
    - `**kwargs[segmentation_network_class_name]`：网络类的其他初始化参数。
11. 对模型应用 He 初始化权重的函数 `InitWeights_He`。
12. 如果网络类是 `ResidualEncoderUNet`，
则对模型应用一个初始化最后一个 BatchNorm 层的函数 `init_last_bn_before_add_to_0`。
13. 返回构建好的模型。

总的来说，这段代码的作用是根据给定的配置和数据集信息构建一个网络模型，
具体的网络类和初始化参数根据配置中的设定来确定。

构建模型的时候，怎么知道使用的是2D还是3D模型 或者级联模型？
具体构建2D 3D模型的细节在哪里？
对于3Dunet模型 分为 全分辨率 低分辨率模型-->代码在哪里体现的？
"""
def get_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
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
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
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
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    return model
