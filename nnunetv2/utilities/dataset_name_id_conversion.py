#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import Union

from paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


"""

"""
def find_candidate_datasets(dataset_id: int):
    startswith = "Dataset%03.0d" % dataset_id
    if nnUNet_preprocessed is not None and isdir(nnUNet_preprocessed):
        candidates_preprocessed = subdirs(nnUNet_preprocessed, prefix=startswith, join=False)
    else:
        candidates_preprocessed = []

    if nnUNet_raw is not None and isdir(nnUNet_raw):
        candidates_raw = subdirs(nnUNet_raw, prefix=startswith, join=False)
    else:
        candidates_raw = []

    candidates_trained_models = []
    if nnUNet_results is not None and isdir(nnUNet_results):
        candidates_trained_models += subdirs(nnUNet_results, prefix=startswith, join=False)
        #这里为什用+=????
    all_candidates = candidates_preprocessed + candidates_raw + candidates_trained_models
    unique_candidates = np.unique(all_candidates)
    return unique_candidates


#根据数据集ID号寻找制定的数据集名字
"""
若找到的数据集名字大于1 则数据集ID号出错
"""
def convert_id_to_dataset_name(dataset_id: int):
    unique_candidates = find_candidate_datasets(dataset_id)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one dataset name found for dataset id %d. Please correct that. (I looked in the "
                           "following folders:\n%s\n%s\n%s" % (dataset_id, nnUNet_raw, nnUNet_preprocessed, nnUNet_results))
    if len(unique_candidates) == 0:
        """
        对于当前ID，若没有找到对应的数据集名称-->打印当前环境变量？？为什么要打印当前环境变量
        根据当前数据集ID号分别在以下文件夹中寻找：
            nnUNet_preprocessed
            nnUNet_results
            nnUNet_raw
            若没有找到对应的数据集名称，
            则说明是:
                1ID号可能出错
                2三个文件夹的路径可能出错
        """
        raise RuntimeError(f"Could not find a dataset with the ID {dataset_id}. Make sure the requested dataset ID "
                           f"exists and that nnU-Net knows where raw and preprocessed data are located "
                           f"(see Documentation - Installation). Here are your currently defined folders:\n"
                           f"nnUNet_preprocessed={os.environ.get('nnUNet_preprocessed') if os.environ.get('nnUNet_preprocessed') is not None else 'None'}\n"
                           f"nnUNet_results={os.environ.get('nnUNet_results') if os.environ.get('nnUNet_results') is not None else 'None'}\n"
                           f"nnUNet_raw={os.environ.get('nnUNet_raw') if os.environ.get('nnUNet_raw') is not None else 'None'}\n"
                           f"If something is not right, adapt your environment variables.")
    return unique_candidates[0]


def convert_dataset_name_to_id(dataset_name: str):
    assert dataset_name.startswith("Dataset")
    dataset_id = int(dataset_name[7:10])
    return dataset_id


"""
若数据集名称命名不正确，则需要根据制定的ID号寻找对应的数据集名称
"""
def maybe_convert_to_dataset_name(dataset_name_or_id: Union[int, str]) -> str:
    if isinstance(dataset_name_or_id, str) and dataset_name_or_id.startswith("Dataset"):
        return dataset_name_or_id#数据集命名正确，直接返回dataset_name_or_id
    if isinstance(dataset_name_or_id, str):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError("dataset_name_or_id was a string and did not start with 'Dataset' so we tried to "
                             "convert it to a dataset ID (int). That failed, however. Please give an integer number "
                             "('1', '2', etc) or a correct tast name. Your input: %s" % dataset_name_or_id)
    return convert_id_to_dataset_name(dataset_name_or_id)