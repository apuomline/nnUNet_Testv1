import os
import socket
from typing import Union, Optional

import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn

# import sys
# sys.path.append('E:\nnUNet')

"""
我们这一代人注定不能与同龄人相似，只能低头赶路。我们这一辈做不出成绩来，下辈就得接着受苦。
除了奋斗，别无选择。
做事时平心静气，心态平和即可。
"""

"""
这段代码是一个函数，用于在本地主机上查找一个空闲的端口。

函数的实现如下：

1. 创建一个socket对象`s`，使用`socket.AF_INET`表示使用IPv4协议，`socket.SOCK_STREAM`
表示使用TCP协议。  
2. 调用`s.bind(("", 0))`将socket绑定到本地主机的一个随机可用端口。
空字符串`""`表示绑定到本地主机的所有网络接口，而0表示由操作系统自动选择一个可用端口。
3. 调用`s.getsockname()[1]`获取绑定的端口号。
4. 调用`s.close()`关闭socket连接。
5. 返回获取到的端口号。

这个函数的作用是在单节点训练时，在不连接到实际主节点的情况下找到一个空闲的端口号。
这通常用于设置`MASTER_PORT`环境变量。
"""
def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


"""
get_trainer_from_args：
这段代码是一个函数，它根据输入参数返回一个nnUNetTrainer对象。

函数的输入参数如下：

- `dataset_name_or_id`：数据集的名称或ID，可以是一个整数或字符串。
- `configuration`：配置参数，一个字符串。
- `fold`：交叉验证的折数，一个整数。
- `trainer_name`：nnUNetTrainer的名称，默认为'nnUNetTrainer'。
- `plans_identifier`：计划标识符，默认为'nnUNetPlans'。
- `use_compressed`：是否使用压缩数据集，默认为False。
- `device`：设备类型，默认为torch.device('cuda')。

以上参数都是用来初始化nnUNet_traniner

函数首先加载nnUNetTrainer类，并进行一些检查。
如果找不到指定的nnUNetTrainer类，将会引发一个运行时错误。
然后，函数将检查nnUNetTrainer类是否是nnUNetTrainer的子类，如果不是，将引发一个断言错误。

接下来，函数处理数据集输入。
如果`dataset_name_or_id`以'Dataset'开头，表示输入的是数据集名称，
不需要做任何处理。否则，函数将尝试将`dataset_name_or_id`转换为整数类型。
如果转换失败，将引发一个值错误。

然后，函数初始化nnunet_trainer对象。
它首先确定预处理数据集文件夹的路径，然后加载计划文件和数据集文件的JSON内容。
最后，使用这些参数初始化nnunet_trainer对象，并返回该对象。
"""
def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          use_compressed: bool = False,
                          device: torch.device = torch.device('cuda')):
    # load nnunet class and do sanity checks

    #根据训练模型名称获取指定的Unettrainer--实例化对象
    #为什么不直接调用类 为什么要在指定文件夹下去递归搜索？
    nnunet_trainer = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    if nnunet_trainer is None:#如果没有找到制定模型对象--则打印报错信息
        #代码中的路径表示没有看明白
        raise RuntimeError(f'Could not find requested nnunet trainer {trainer_name} in '
                           f'nnunetv2.training.nnUNetTrainer ('
                           f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
                           f'else, please move it there.')
    assert issubclass(nnunet_trainer, nnUNetTrainer), 'The requested nnunet trainer class must inherit from ' \
                                                    'nnUNetTrainer'
    #判断当前模型对象是否为nnUNetTraniner中的子类对象
    #并不知道为甚要写这段代码

    # handle dataset input. If it's an ID we need to convert to int from string
    #为什么要将输入数据集的ID号改为字符类？？
    """
    可能是输入数据集命名格式：
        1新数据集用名字命名
        2若为十大数据集中的某一个，则用数字代替
    """
    if dataset_name_or_id.startswith('Dataset'):#数据集的命名格式必须正确
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {dataset_name_or_id}')
                            #datasetXXX表示数据集ID号，自定义的数据集需要从指定的ID号开始
                            #YYY表示数据集名称
    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    #preprocessed_dataset_folder_base 确定预处理数据集文件夹路径
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)#？？？？？？？
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, unpack_dataset=not use_compressed, device=device)
    return nnunet_trainer

"""
训练方式：
1：从0开始训练--使用检查点
2：使用预训练--获取预训练文件
"""
"""
模型的训练时间一般很长，checkpoint文件保存的是当前模型的权重参数与优化器状态--方便之后恢复训练
若检查点文件为空，则模型加载checkpoint文件失败。
"""

#不太明白编写这个函数的真正目的
def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                           'be used at the beginning of the training.')
    if continue_training:
        """
        依次判断一下检查点文件是否存在
        最终检查点文件
        最新检查点文件
        最优检查点文件-->若当前检查点文件存在，则跳出if语句
        """
        #检查点文件存放在输出文件夹下--output_folder
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                               f"continue from. Starting a new training...")
            expected_checkpoint_file = None
    elif validation_only:
        """
        验证前提是训练结束即 检查点文件不为空
        验证集使用的是最终的检查点文件
        """
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(f"Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)
        #最终训练器加载检查点文件，前提是检查点文件路径不为空


def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)#初始化进程组


def cleanup_ddp():
    dist.destroy_process_group()

#进行分布式训练模型
"""
建议，查看常见分布式训练编写的代码
"""
def run_ddp(rank, dataset_name_or_id, configuration, fold, tr, p, use_compressed, disable_checkpointing, c, val,
            pretrained_weights, npz, val_with_best, world_size):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))
#前两行设置分布式环境
    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p,
                                           use_compressed)
#获取训练器--这里的训练器指定是？ #数据集名称，配置不同，使用的训练器也不同

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), f'Cannot set --c and --val flag at the same time. Dummy.'
#训练与验证不可以同步
    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)
    #若既不训练也不验证，则加载预训练模型

    if torch.cuda.is_available():#若当前有gpu环境，则进行gpu加速
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        nnunet_trainer.run_training()

    if val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
    nnunet_trainer.perform_actual_validation(npz)#这里的npz是什么？？
    #保存到npz中的内容到底是什么？不太清楚
    """
    因为模型进行了交叉验证，取5次交叉验证中模型结果最好的验证结果--保存的是模型的参数信息？
    然后使用验证结果最好的模型去验证没有见过的数据集？
    """

    cleanup_ddp()#关闭当前分布式huanjing


def run_training(dataset_name_or_id: Union[str, int],
                 configuration: str, fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 use_compressed_data: bool = False,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 val_with_best: bool = False,
                 device: torch.device = torch.device('cuda')):
    """
    三个if条件语句的作用
    这段代码包含了一些条件判断和断言语句，用于检查和设置一些参数的合法性和一些环境变量。

首先，代码检查变量fold的类型是否为字符串。
如果是字符串，那么进一步判断它是否等于'all'。
如果不等于'all'，则尝试将其转换为整数类型。如果转换失败，会打印一条错误信息并抛出一个ValueError异常。
这段代码的作用是将fold参数转换为整数类型，或者确保它的值为字符串'all'。
接下来，代码检查变量val_with_best的值是否为True。
如果是True，则断言disable_checkpointing的值为False，否则会抛出一个AssertionError异常。
这段代码的作用是确保当val_with_best为True时，disable_checkpointing必须为False。

然后，代码检查变量num_gpus的值是否大于1。
如果是大于1，那么断言device的类型为'cuda'，否则会抛出一个AssertionError异常。
这段代码的作用是确保当num_gpus大于1时，设备类型必须为'cuda'。

最后，代码设置了一个名为'MASTER_ADDR'的环境变量为'localhost'。
如果环境变量'MASTER_PORT'不存在，代码会调用find_free_network_port函数找到一个可用的网络端口，
并将其设置为'MASTER_PORT'的值。
这段代码的作用是为使用多个GPU进行分布式训练（DDP training）时设置主节点的地址和端口。
    """
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    if val_with_best:
        assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

    if num_gpus > 1:
        assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        mp.spawn(run_ddp,#好像是multiprocessing--多进程训练？？
                 args=(
                     dataset_name_or_id,
                     configuration,
                     fold,
                     trainer_class_name,
                     plans_identifier,
                     use_compressed_data,
                     disable_checkpointing,
                     continue_training,
                     only_run_validation,
                     pretrained_weights,
                     export_validation_probabilities,
                     val_with_best,
                     num_gpus),
                 nprocs=num_gpus,
                 join=True)

    else:
        
        #获取nnUNetTrainer类
        #这里的configuration应该就是：2d 3d_fullers 3d_lowers等信息
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, use_compressed_data, device=device)
        
        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

        #如果是训练模型的话--有可能需要加载模型的检查点文件--从中断的地方继续训练
        #每50个epoch可以保存一个checkpoint文件
        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)

        #如果当前有gpu环境 则开启gpu加速
        if torch.cuda.is_available():#当gpu个数为1时
            cudnn.deterministic = False# torch.backends.cudnn.deterministic->卷积层都使用通样的卷积算法
            cudnn.benchmark = True# 当模型的网络结构固定，输入图像的尺寸固定时，对于每一个卷积层选择最适合当前卷积层的卷积算法
            #卷积算法实现有很多种--不同的卷积层使用不同的卷积算法所需时间不同。

        if not only_run_validation:
            nnunet_trainer.run_training()

        if val_with_best:
            nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
            #这里的加载checkpoint文件与上面的maybe_load_checkpoint文件并不冲突
        nnunet_trainer.perform_actual_validation(export_validation_probabilities)
        """
        主要是要看训练器类的run_trianing函数，load_checkpoint函数以及preform_actual_validation函数
        """
"""
else 语句后代码的作用：
这段代码主要是根据参数和配置信息创建一个`nnunet_trainer`对象，并执行训练和验证的操作。

首先，代码调用`get_trainer_from_args`函数来创建一个`nnunet_trainer`对象。
这个函数根据给定的参数和配置信息，选择相应的训练器类，并返回一个实例化的训练器对象。

接下来，如果`disable_checkpointing`参数为True，
则将`nnunet_trainer`对象的`disable_checkpointing`属性设置为True，以禁用模型检查点的保存。

然后，代码检查`continue_training`和`only_run_validation`参数的值。
如果两者都为True，则抛出一个AssertionError异常，因为不能同时设置这两个参数。

接着，代码调用`maybe_load_checkpoint`函数来根据参数设置，加载预训练的模型权重或模型检查点。

如果系统支持CUDA，则将`cudnn.deterministic`设置为False（用于提高性能），
`cudnn.benchmark`设置为True（用于自动寻找最佳的卷积算法）。

如果不仅仅是进行验证，代码调用`nnunet_trainer`对象的`run_training`方法来执行训练过程。

如果`val_with_best`参数为True，
代码加载保存在`nnunet_trainer.output_folder`目录中的`checkpoint_best.pth`模型检查点。

最后，代码调用`nnunet_trainer`对象的`perform_actual_validation`方法来执行实际的验证操作，
并可以选择是否导出验证结果的概率值。

总体而言，这段代码根据参数和配置信息创建训练器对象，并根据参数的设置执行训练和验证的操作。
"""

#获取训练时，输入参数--设置以下信息：
"""
1 训练集名称
2 配置文件？
3交叉验证折数
4训练器
5
6使用预训练模型的权重与参数
7设置GPU数量
8是否使用压缩数据的训练方式
9是否将验证结果保存为npz文件-->方便后续选择最佳Unet模型
10？
11指定模型在验证时，使用最佳的checkpoint文件中保存的模型参数与优化器信息
12禁用检查点--减少内存消耗
13设置训练数据集使用的设备。gpu or cpu
"""
def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")#训练模型需要的配置
    parser.add_argument('fold', type=str,#交叉验证折数
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',#设置训练器--默认为nnUNetTrainer
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')#针对于新数据集，微调已经训练好的模型
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')#设置训练使用的GPU数
    parser.add_argument("--use_compressed", default=False, action="store_true", required=False,
                        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
                             "data is much more CPU and (potentially) RAM intensive and should only be used if you "
                             "know what you are doing")#是否使用压缩数据的训练方式
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
                                #是否将验证结果：softmax的预测值保存为npz文件，保存为npz文件-->用于找到最佳nnUNet模型
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')#从最近的checkpoint中保存的模型与优化器信息恢复训练
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')#在训练结束后开始验证
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead '
                             'of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! '
                             'WARNING: This will use the same \'validation\' folder as the regular validation '
                             'with no way of distinguishing the two!')#指定模型在进行验证时，使用的检查点为：checkpoint_best
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')#禁用检查点
    parser.add_argument('-device', type=str, default='cuda', required=False,
                    help="Use this to set the device the training should run with. Available options are 'cuda' "
                         "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                         "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")#设置训练数据集时使用的设备
    args = parser.parse_args()


    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    run_training(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
                 args.num_gpus, args.use_compressed, args.npz, args.c, args.val, args.disable_checkpointing, args.val_best,
                 device=device)


if __name__ == '__main__':
    run_training_entry()
