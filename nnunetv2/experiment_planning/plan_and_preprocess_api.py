
import shutil
from typing import List, Type, Optional, Tuple, Union

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles, load_json

from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name, maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

"""
extract_fingerprints-->extract_fingerprints_dataset-
>DatasetFingerprintExtractor->run->{
   
    analyze_case,
}
"""
def extract_fingerprint_dataset(dataset_id: int,
                                fingerprint_extractor_class: Type[
                                    DatasetFingerprintExtractor] = DatasetFingerprintExtractor,
                                num_processes: int = default_num_processes, check_dataset_integrity: bool = False,
                                clean: bool = True, verbose: bool = True):
    """
    Returns the fingerprint as a dictionary (additionally to saving it)
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(dataset_name)

    if check_dataset_integrity:#检查数据集的合理性
        verify_dataset_integrity(join(nnUNet_raw, dataset_name), num_processes)

    fpe = fingerprint_extractor_class(dataset_id, num_processes, verbose=verbose)
    return fpe.run(overwrite_existing=clean)#获取数据集的fingerprint文件


def extract_fingerprints(dataset_ids: List[int], fingerprint_extractor_class_name: str = 'DatasetFingerprintExtractor',
                         num_processes: int = default_num_processes, check_dataset_integrity: bool = False,
                         clean: bool = True, verbose: bool = True):
    """
    clean = False will not actually run this. This is just a switch for use with nnUNetv2_plan_and_preprocess where
    we don't want to rerun fingerprint extraction every time.
    """
    """
    函数有五个参数：dataset_ids，是一个整数列表，表示要提取指纹的数据集的ID；
    fingerprint_extractor_class_name，是一个字符串，表示指纹提取器的类名，
    默认为DatasetFingerprintExtractor；num_processes，是一个整数，
    表示要使用的进程数，默认为default_num_processes；check_dataset_integrity，
    是一个布尔值，表示是否检查数据集的完整性，默认为False；clean，
    是一个布尔值，表示是否清理数据，默认为True；verbose，是一个布尔值，
    表示是否显示详细信息，默认为True。
    根据fingerprint_extractor_class_name从nnunetv2.experiment_planning
    模块中递归查找指纹提取器的类，并将其赋值给fingerprint_extractor_class变量。
    对于dataset_ids中的每个数据集ID，调用extract_fingerprint_dataset函数，
    传入数据集ID、指纹提取器类、进程数、是否检查数据集完整性、
    是否清理数据和是否显示详细信息作为参数。
    """
    fingerprint_extractor_class = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                              fingerprint_extractor_class_name,
                                                              current_module="nnunetv2.experiment_planning")
    for d in dataset_ids:
        extract_fingerprint_dataset(d, fingerprint_extractor_class, num_processes, check_dataset_integrity, clean,
                                    verbose)


def plan_experiment_dataset(dataset_id: int,
                            experiment_planner_class: Type[ExperimentPlanner] = ExperimentPlanner,
                            gpu_memory_target_in_gb: float = 8, preprocess_class_name: str = 'DefaultPreprocessor',
                            overwrite_target_spacing: Optional[Tuple[float, ...]] = None,
                            overwrite_plans_name: Optional[str] = None) -> dict:
    """
    overwrite_target_spacing ONLY applies to 3d_fullres and 3d_cascade fullres!
    """
    """
    plan_experiment_dataset->plan_experiment->determine_fullres_target_spacing
    (统一所有图像的spacing)
    plan_experiment_dataset->ExperimentPlanner->plan_experiment->get_plans_for_configuration
    (确定网络架构):
    {input_patch_size,batch_size这包括池化操作的数量、池化操作的内核大小、
    卷积内核大小以及必须被某些值整除的形状。
    get_plans_for_configuration->get_pool_and_conv_props(获取池化与卷积属性)-->
    用来构建编码器与解码器
    }
    """
    kwargs = {}
    if overwrite_plans_name is not None:
        kwargs['plans_name'] = overwrite_plans_name
    return experiment_planner_class(dataset_id,
                                    gpu_memory_target_in_gb=gpu_memory_target_in_gb,
                                    preprocessor_name=preprocess_class_name,
                                    overwrite_target_spacing=[float(i) for i in overwrite_target_spacing] if
                                    overwrite_target_spacing is not None else overwrite_target_spacing,
                                    suppress_transpose=False,  # might expose this later,
                                    **kwargs
                                    ).plan_experiment()


def plan_experiments(dataset_ids: List[int], experiment_planner_class_name: str = 'ExperimentPlanner',
                     gpu_memory_target_in_gb: float = 8, preprocess_class_name: str = 'DefaultPreprocessor',
                     overwrite_target_spacing: Optional[Tuple[float, ...]] = None,
                     overwrite_plans_name: Optional[str] = None):
    """
    overwrite_target_spacing ONLY applies to 3d_fullres and 3d_cascade fullres!
    """
    experiment_planner = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                     experiment_planner_class_name,
                                                     current_module="nnunetv2.experiment_planning")
    for d in dataset_ids:
        plan_experiment_dataset(d, experiment_planner, gpu_memory_target_in_gb, preprocess_class_name,
                                overwrite_target_spacing, overwrite_plans_name)


#预处理细节在 处理器类中实现？
"""
`preprocess_dataset`函数用于预处理一个特定的数据集。它接受以下参数：
- `dataset_id`：数据集的唯一标识符。
- `plans_identifier`：计划标识符，默认为'nnUNetPlans'。
- `configurations`：配置列表，默认为('2d', '3d_fullres', '3d_lowres')。
这些配置定义了预处理的不同方式。
- `num_processes`：进程数列表，默认为(8, 4, 8)。每个配置对应的预处理过程使用的并行进程数。
- `verbose`：是否打印详细信息，默认为False。
函数首先对`num_processes`进行处理，将其转换为列表类型。如果`num_processes`不是列表类型，
则将其转换为列表。接下来，根据配置列表的长度调整`num_processes`的长度。
如果`num_processes`的长度为1，则将其重复扩展为与配置列表相同的长度。
如果`num_processes`的长度与配置列表的长度不相等，则抛出`RuntimeError`异常。
然后，函数根据数据集的唯一标识符获取数据集的名称，并打印预处理的信息。
接下来，函数加载计划文件，并使用`PlansManager`类进行管理。
然后，对于每个配置和进程数，函数打印配置信息，并检查配置是否存在于计划文件中。
如果配置不存在，则打印相应的信息并跳过该配置。
如果配置存在，则获取相应的预处理器类，并创建预处理器对象。
最后，运行预处理器的`run`方法，传递数据集的唯一标识符、配置、计划标识符和进程数作为参数。
最后一部分代码用于将标签数据复制到`nnUNet_preprocessed`文件夹中的`gt_segmentations`文件夹，
以便进行验证。它首先创建`gt_segmentations`文件夹（如果不存在），
然后加载数据集的`dataset.json`文件，并获取训练图像和目标的文件名。接下来，
将标签文件复制到`gt_segmentations`文件夹中，只复制比已存在文件更新的文件。
"""
def preprocess_dataset(dataset_id: int,
                       plans_identifier: str = 'nnUNetPlans',
                       configurations: Union[Tuple[str], List[str]] = ('2d', '3d_fullres', '3d_lowres'),
                       num_processes: Union[int, Tuple[int, ...], List[int]] = (8, 4, 8),
                       verbose: bool = False) -> None:
    if not isinstance(num_processes, list):

        num_processes = list(num_processes)
    if len(num_processes) == 1:
        num_processes = num_processes * len(configurations)
    if len(num_processes) != len(configurations):
        raise RuntimeError(
            f'The list provided with num_processes must either have len 1 or as many elements as there are '
            f'configurations (see --help). Number of configurations: {len(configurations)}, length '
            f'of num_processes: '
            f'{len(num_processes)}')

    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(f'Preprocessing dataset {dataset_name}')
    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
    plans_manager = PlansManager(plans_file)
    for n, c in zip(num_processes, configurations):
        print(f'Configuration: {c}...')
        if c not in plans_manager.available_configurations:
            print(
                f"INFO: Configuration {c} not found in plans file {plans_identifier + '.json'} of "
                f"dataset {dataset_name}. Skipping.")
            continue
        configuration_manager = plans_manager.get_configuration(c)
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        preprocessor.run(dataset_id, c, plans_identifier, num_processes=n)
    """
    首先生成 plansManager类对象 -->通过get.configureation 函数获取ConfigurationManager类对象
    然后通过ConfigurationManager类中的 preprocessor_class获取预处理器类对象-->运行run函数
    这里的preprocessor对象又继承 DefaultPreprocessor类 run函数是 DefaultPreprocessor类中的方法
    """

    # copy the gt to a folder in the nnUNet_preprocessed so that we can do validation even if the raw data is no
    # longer there (useful for compute cluster where only the preprocessed data is available)
    from distutils.file_util import copy_file
    maybe_mkdir_p(join(nnUNet_preprocessed, dataset_name, 'gt_segmentations'))
    dataset_json = load_json(join(nnUNet_raw, dataset_name, 'dataset.json'))
    dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)
    # only copy files that are newer than the ones already present
    for k in dataset:
        copy_file(dataset[k]['label'],
                  join(nnUNet_preprocessed, dataset_name, 'gt_segmentations', k + dataset_json['file_ending']),
                  update=True)



def preprocess(dataset_ids: List[int],
               plans_identifier: str = 'nnUNetPlans',
               configurations: Union[Tuple[str], List[str]] = ('2d', '3d_fullres', '3d_lowres'),
               num_processes: Union[int, Tuple[int, ...], List[int]] = (8, 4, 8),
               verbose: bool = False):
    for d in dataset_ids:
        preprocess_dataset(d, plans_identifier, configurations, num_processes, verbose)
