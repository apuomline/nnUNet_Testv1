#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
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
import multiprocessing
import re
from multiprocessing import Pool
from typing import Type

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    get_filenames_of_train_images_and_targets


def verify_labels(label_file: str, readerclass: Type[BaseReaderWriter], expected_labels: List[int]) -> bool:
    rw = readerclass()
    seg, properties = rw.read_seg(label_file)
    found_labels = np.sort(pd.unique(seg.ravel()))  # np.unique(seg)
    unexpected_labels = [i for i in found_labels if i not in expected_labels]
    if len(found_labels) == 0 and found_labels[0] == 0:
        print('WARNING: File %s only has label 0 (which should be background). This may be intentional or not, '
              'up to you.' % label_file)
    if len(unexpected_labels) > 0:
        print("Error: Unexpected labels found in file %s.\nExpected: %s\nFound: %s" % (label_file, expected_labels,
                                                                                       found_labels))
        return False
    return True


def check_cases(image_files: List[str], label_file: str, expected_num_channels: int,
                readerclass: Type[BaseReaderWriter]) -> bool:
    rw = readerclass()
    ret = True

    images, properties_image = rw.read_images(image_files)
    segmentation, properties_seg = rw.read_seg(label_file)

    # check for nans
    if np.any(np.isnan(images)):
        print(f'Images contain NaN pixel values. You need to fix that by '
              f'replacing NaN values with something that makes sense for your images!\nImages:\n{image_files}')
        ret = False
    if np.any(np.isnan(segmentation)):
        print(f'Segmentation contains NaN pixel values. You need to fix that.\nSegmentation:\n{label_file}')
        ret = False

    # check shapes
    shape_image = images.shape[1:]
    shape_seg = segmentation.shape[1:]
    if not all([i == j for i, j in zip(shape_image, shape_seg)]):
        print('Error: Shape mismatch between segmentation and corresponding images. \nShape images: %s. '
              '\nShape seg: %s. \nImage files: %s. \nSeg file: %s\n' %
              (shape_image, shape_seg, image_files, label_file))
        ret = False

    # check spacings
    spacing_images = properties_image['spacing']
    spacing_seg = properties_seg['spacing']
    if not np.allclose(spacing_seg, spacing_images):
        print('Error: Spacing mismatch between segmentation and corresponding images. \nSpacing images: %s. '
              '\nSpacing seg: %s. \nImage files: %s. \nSeg file: %s\n' %
              (shape_image, shape_seg, image_files, label_file))
        ret = False

    # check modalities
    if not len(images) == expected_num_channels:
        print('Error: Unexpected number of modalities. \nExpected: %d. \nGot: %d. \nImages: %s\n'
              % (expected_num_channels, len(images), image_files))
        ret = False

    # nibabel checks
    if 'nibabel_stuff' in properties_image.keys():
        # this image was read with NibabelIO
        affine_image = properties_image['nibabel_stuff']['original_affine']
        affine_seg = properties_seg['nibabel_stuff']['original_affine']
        if not np.allclose(affine_image, affine_seg):
            print('WARNING: Affine is not the same for image and seg! \nAffine image: %s \nAffine seg: %s\n'
                  'Image files: %s. \nSeg file: %s.\nThis can be a problem but doesn\'t have to be. Please run '
                  'nnUNet_plot_dataset_pngs to verify if everything is OK!\n'
                  % (affine_image, affine_seg, image_files, label_file))

    # sitk checks
    if 'sitk_stuff' in properties_image.keys():
        # this image was read with SimpleITKIO
        # spacing has already been checked, only check direction and origin
        origin_image = properties_image['sitk_stuff']['origin']
        origin_seg = properties_seg['sitk_stuff']['origin']
        if not np.allclose(origin_image, origin_seg):
            print('Warning: Origin mismatch between segmentation and corresponding images. \nOrigin images: %s. '
                  '\nOrigin seg: %s. \nImage files: %s. \nSeg file: %s\n' %
                  (origin_image, origin_seg, image_files, label_file))
        direction_image = properties_image['sitk_stuff']['direction']
        direction_seg = properties_seg['sitk_stuff']['direction']
        if not np.allclose(direction_image, direction_seg):
            print('Warning: Direction mismatch between segmentation and corresponding images. \nDirection images: %s. '
                  '\nDirection seg: %s. \nImage files: %s. \nSeg file: %s\n' %
                  (direction_image, direction_seg, image_files, label_file))

    return ret


#验证在nnUNet_raw_data_base文件夹中存放的数据集，其格式是否满足要求
def verify_dataset_integrity(folder: str, num_processes: int = 8) -> None:
    """
    folder needs the imagesTr, imagesTs and labelsTr subfolders. There also needs to be a dataset.json
    checks if the expected number of training cases and labels are present
    for each case, if possible, checks whether the pixel grids are aligned
    checks whether the labels really only contain values they should
    :param folder:
    :return:
    """
    #查看raw_data_base文件夹中是否含有dataset.json文件
    assert isfile(join(folder, "dataset.json")), "There needs to be a dataset.json file in folder, folder=%s" % folder
    dataset_json = load_json(join(folder, "dataset.json"))#将dataset.json文件加载进来

    if not 'dataset' in dataset_json.keys():#如果dataset.json文件中没有键dataset
        assert isdir(join(folder, "imagesTr")), "There needs to be a imagesTr subfolder in folder, folder=%s" % folder
        assert isdir(join(folder, "labelsTr")), "There needs to be a labelsTr subfolder in folder, folder=%s" % folder
        #检查数据集文件夹下是否含有imagesTr 与labelsTr文件夹
    # make sure all required keys are there
    dataset_keys = list(dataset_json.keys())#获取json文件中所有键值
    required_keys = ['labels', "channel_names", "numTraining", "file_ending"]
    assert all([i in dataset_keys for i in required_keys]), 'not all required keys are present in dataset.json.' \
                                                            '\n\nRequired: \n%s\n\nPresent: \n%s\n\nMissing: ' \
                                                            '\n%s\n\nUnused by nnU-Net:\n%s' % \
                                                            (str(required_keys),
                                                             str(dataset_keys),
                                                             str([i for i in required_keys if i not in dataset_keys]),
                                                             str([i for i in dataset_keys if i not in required_keys]))
    #断言json文件中是否包含以上必须有的键值
    """
    获取训练的样本数，模态数(单模态或者多模态)，文件扩展名
    """
    expected_num_training = dataset_json['numTraining']
    num_modalities = len(dataset_json['channel_names'].keys()
                         if 'channel_names' in dataset_json.keys()
                         else dataset_json['modality'].keys())
    file_ending = dataset_json['file_ending']

    dataset = get_filenames_of_train_images_and_targets(folder, dataset_json)

    #这里的dataset获取到的是数据集中的所有样本名称--存放路径 以及与之对应的标签--存放路径
    # check if the right number of training cases is present
    assert len(dataset) == expected_num_training, 'Did not find the expected number of training cases ' \
                                                               '(%d). Found %d instead.\nExamples: %s' % \
                                                               (expected_num_training, len(dataset),
                                                                list(dataset.keys())[:5])
    #从数据集文件夹中获取到训练样本名称与对应的标签名称 断言样本数是否data.json文件中的样本数一致

    # check if corresponding labels are present
    if 'dataset' in dataset_json.keys():
        """
        确保样本与对应的标签都存在
        """
        # just check if everything is there
        ok = True
        missing_images = []
        missing_labels = []
        for k in dataset:
            for i in dataset[k]['images']:
                if not isfile(i):
                    missing_images.append(i)
                    ok = False
            if not isfile(dataset[k]['label']):
                missing_labels.append(dataset[k]['label'])
                ok = False
        if not ok:
            raise FileNotFoundError(f"Some expeted files were missing. Make sure you are properly referencing them "
                                    f"in the dataset.json. Or use imagesTr & labelsTr folders!\nMissing images:"
                                    f"\n{missing_images}\n\nMissing labels:\n{missing_labels}")
    else:
        # old code that uses imagestr and labelstr folders
        labelfiles = subfiles(join(folder, 'labelsTr'), suffix=file_ending, join=False)
        label_identifiers = [i[:-len(file_ending)] for i in labelfiles]
        labels_present = [i in label_identifiers for i in dataset.keys()]
        missing = [i for j, i in enumerate(dataset.keys()) if not labels_present[j]]
        assert all(labels_present), 'not all training cases have a label file in labelsTr. Fix that. Missing: %s' % missing

    """
    它使用dataset.json文件中指定的标签创建一个LabelManager对象。
    它检查标签是否按连续顺序排列（0、1、2、...）。具体细节还需要理解
    如果不是，则引发AssertionError并显示错误消息。
    它根据数据集格式和第一个图像文件确定读取器/写入器类。
    它使用并行处理来验证分割图像的标签。
    如果任何分割图像包含意外标签，则引发RuntimeError并显示错误消息。
        (这里验证标签到底是为了验证什么？)
    它使用并行处理来检查图像和标签的形状和间距。
    如果任何图像存在错误，则引发RuntimeError并显示错误消息。
    """
    labelfiles = [v['label'] for v in dataset.values()]
    image_files = [v['images'] for v in dataset.values()]

    # no plans exist yet, so we can't use PlansManager and gotta roll with the default. It's unlikely to cause
    # problems anyway
    label_manager = LabelManager(dataset_json['labels'], regions_class_order=dataset_json.get('regions_class_order'))
    expected_labels = label_manager.all_labels
    if label_manager.has_ignore_label:
        expected_labels.append(label_manager.ignore_label)
    labels_valid_consecutive = np.ediff1d(expected_labels) == 1
    assert all(
        labels_valid_consecutive), f'Labels must be in consecutive order (0, 1, 2, ...). The labels {np.array(expected_labels)[1:][~labels_valid_consecutive]} do not satisfy this restriction'

    # determine reader/writer class
    reader_writer_class = determine_reader_writer_from_dataset_json(dataset_json, dataset[dataset.keys().__iter__().__next__()]['images'][0])

    # check whether only the desired labels are present
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        result = p.starmap(
            verify_labels,
            zip([join(folder, 'labelsTr', i) for i in labelfiles], [reader_writer_class] * len(labelfiles),
                [expected_labels] * len(labelfiles))
        )
        if not all(result):
            raise RuntimeError(
                'Some segmentation images contained unexpected labels. Please check text output above to see which one(s).')


        # check whether shapes and spacings match between images and labels
        result = p.starmap(
            check_cases,
            zip(image_files, labelfiles, [num_modalities] * expected_num_training,
                [reader_writer_class] * expected_num_training)
        )
        if not all(result):
            raise RuntimeError(
                'Some images have errors. Please check text output above to see which one(s) and what\'s going on.')

    # check for nans
    # check all same orientation nibabel
    print('\n####################')
    print('verify_dataset_integrity Done. \nIf you didn\'t see any error messages then your dataset is most likely OK!')
    print('####################\n')


if __name__ == "__main__":
    # investigate geometry issues
    example_folder = join(nnUNet_raw, 'Dataset250_COMPUTING_it0')
    num_processes = 6
    verify_dataset_integrity(example_folder, num_processes)
