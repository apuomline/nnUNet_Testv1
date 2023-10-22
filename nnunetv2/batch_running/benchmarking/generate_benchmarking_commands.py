"""
这段代码是一个Python脚本，它用于在DKFZ基础设施中使用LSF调度器运行一个特定的任务。
它包含了一些变量和配置，用于生成一系列的命令，这些命令将被提交给调度器以在GPU上执行训练任务。

在代码的开头，`if __name__ == '__main__':`语句用于判断是否直接运行该脚本。
这是Python中常见的一种约定，可以确保脚本在作为模块导入时不会自动执行。

在代码中定义了一些变量，包括`gpu_models`、`datasets`、`trainers`、`plans`、`configs`和`num_gpus`。
这些变量用于配置训练任务的不同参数和选项。

`benchmark_configurations`是一个字典，用于指定每个数据集应该使用哪些配置。

`exclude_hosts`、`resources`、`queue`和`preamble`是一些调度器相关的选项和参数。

`train_command`是实际的训练命令，它指定了训练任务的一些参数和选项。

`folds`是一个包含要训练的折叠数的元组。

`use_these_modules`是一个字典，用于指定每个训练器应该使用哪个计划。

`additional_arguments`是一个包含额外参数的字符串。

`output_file`是一个指定输出文件路径的字符串。

代码的主要部分是一个嵌套的循环，用于生成一系列的训练命令。每个命令都是通过组合不同的变量和选项来构建的。
生成的命令被写入到指定的输出文件中。

总之，这段代码是用于生成一系列训练命令的脚本，这些命令将在GPU上执行训练任务。
它使用了一些变量和配置来生成不同的命令，以满足特定的训练需求。请注意，这段代码可能需要根据您的调度器和环境进行适当的调整和修改。
"""
if __name__ == '__main__':
    """
    This code probably only works within the DKFZ infrastructure (using LSF). You will need to adapt it to your scheduler! 
    """
    gpu_models = [#'NVIDIAA100_PCIE_40GB', 'NVIDIAGeForceRTX2080Ti', 'NVIDIATITANRTX', 'TeslaV100_SXM2_32GB',
                  'NVIDIAA100_SXM4_40GB']#, 'TeslaV100_PCIE_32GB']
    datasets = [2, 3, 4, 5]
    trainers = ['nnUNetTrainerBenchmark_5epochs', 'nnUNetTrainerBenchmark_5epochs_noDataLoading']
    plans = ['nnUNetPlans']
    configs = ['2d', '2d_bs3x', '2d_bs6x', '3d_fullres', '3d_fullres_bs3x', '3d_fullres_bs6x']
    num_gpus = 1

    benchmark_configurations = {d: configs for d in datasets}

    exclude_hosts = "-R \"select[hname!='e230-dgxa100-1']'\""
    resources = "-R \"tensorcore\""
    queue = "-q gpu"
    preamble = "-L /bin/bash \"source ~/load_env_torch210.sh && "
    train_command = 'nnUNet_compile=False nnUNet_results=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake_benchmark nnUNetv2_train'

    folds = (0, )

    use_these_modules = {
        tr: plans for tr in trainers
    }

    additional_arguments = f' -num_gpus {num_gpus}'  # ''

    output_file = "/home/isensee/deleteme.txt"
    with open(output_file, 'w') as f:
        for g in gpu_models:
            gpu_requirements = f"-gpu num={num_gpus}:j_exclusive=yes:gmodel={g}"
            for tr in use_these_modules.keys():
                for p in use_these_modules[tr]:
                    for dataset in benchmark_configurations.keys():
                        for config in benchmark_configurations[dataset]:
                            for fl in folds:
                                command = f'bsub {exclude_hosts} {resources} {queue} {gpu_requirements} {preamble} {train_command} {dataset} {config} {fl} -tr {tr} -p {p}'
                                if additional_arguments is not None and len(additional_arguments) > 0:
                                    command += f' {additional_arguments}'
                                f.write(f'{command}\"\n')