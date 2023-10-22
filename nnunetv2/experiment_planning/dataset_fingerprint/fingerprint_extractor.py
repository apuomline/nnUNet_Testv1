import multiprocessing
import os
from time import sleep
from typing import List, Type, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from tqdm import tqdm

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


class DatasetFingerprintExtractor(object):
    def __init__(self, dataset_name_or_id: Union[str, int], num_processes: int = 8, verbose: bool = False):
        """
        extracts the dataset fingerprint used for experiment planning. The dataset fingerprint will be saved as a
        json file in the input_folder

        Philosophy here is to do only what we really need. Don't store stuff that we can easily read from somewhere
        else. Don't compute stuff we don't need (except for intensity_statistics_per_channel)
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        self.verbose = verbose

        self.dataset_name = dataset_name
        self.input_folder = join(nnUNet_raw, dataset_name)
        self.num_processes = num_processes
        self.dataset_json = load_json(join(self.input_folder, 'dataset.json'))
        self.dataset = get_filenames_of_train_images_and_targets(self.input_folder, self.dataset_json)

        # We don't want to use all foreground voxels because that can accumulate a lot of data (out of memory). It is
        # also not critically important to get all pixels as long as there are enough. Let's use 10e7 voxels in total
        # (for the entire dataset)
        self.num_foreground_voxels_for_intensitystats = 10e7

    @staticmethod
    def collect_foreground_intensities(segmentation: np.ndarray, images: np.ndarray, seed: int = 1234,
                                       num_samples: int = 10000):
        """
        images=image with multiple channels = shape (c, x, y(, z))
        若图像为2维--该如何应对？对应的shape：(c,x,y)？？ 断言写的是4？--
        这里不包含2维图像吗？
        """
        """
        断言通道数，图像与分割图像不能为空
          numpy中的choice函数#从(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
                #replace:True表示可以取相同数字，False表示不可以取相同数字
                #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
        """
        assert len(images.shape) == 4
        assert len(segmentation.shape) == 4

        assert not np.any(np.isnan(segmentation)), "Segmentation contains NaN values. grrrr.... :-("
        assert not np.any(np.isnan(images)), "Images contains NaN values. grrrr.... :-("

        rs = np.random.RandomState(seed)#设置随机种子

        intensities_per_channel = []# 对于每一个通道 随机选取n个前景像素--后续用于增强?
        # we don't use the intensity_statistics_per_channel at all, it's just something that might be nice to have
        intensity_statistics_per_channel = []#对于每一个通道 统计前景像素 信息

        # segmentation is 4d: 1,x,y,z. We need to remove the empty dimension for the following code to work
        foreground_mask = segmentation[0] > 0 #前景掩码
        """
        
        对于图像中的每一个通道：
            随机选取部分前景像素--组成的列表--后续应该用于增强
            计算前景区域的均值，中位数，最值，99.5% 0.5% 
        """
        for i in range(len(images)):
            foreground_pixels = images[i][foreground_mask]
            num_fg = len(foreground_pixels)
            # sample with replacement so that we don't get issues with cases that have less than num_samples
            # foreground_pixels. We could also just sample less in those cases but that would than cause these
            # training cases to be underrepresented
            intensities_per_channel.append(
                rs.choice(foreground_pixels, num_samples, replace=True) if num_fg > 0 else [])
            
            intensity_statistics_per_channel.append({
                'mean': np.mean(foreground_pixels) if num_fg > 0 else np.nan,
                'median': np.median(foreground_pixels) if num_fg > 0 else np.nan,
                'min': np.min(foreground_pixels) if num_fg > 0 else np.nan,
                'max': np.max(foreground_pixels) if num_fg > 0 else np.nan,
                'percentile_99_5': np.percentile(foreground_pixels, 99.5) if num_fg > 0 else np.nan,
                'percentile_00_5': np.percentile(foreground_pixels, 0.5) if num_fg > 0 else np.nan,

            })

        return intensities_per_channel, intensity_statistics_per_channel
#尽全力啊 努力提升自己 加油！！！
    @staticmethod
    def analyze_case(image_files: List[str], segmentation_file: str, reader_writer_class: Type[BaseReaderWriter],
                     num_samples: int = 10000):
        """
        这段代码定义了一个名为`analyze_case`的函数，用于分析裁剪后图像和labels的信息。

        函数接受以下参数：
        - `image_files`：图像文件的列表。
        - `segmentation_file`：分割结果文件的路径。
        - `reader_writer_class`：用于读取和写入图像和labels的类。
        - `num_samples`：用于计算前景强度统计的样本数，默认为10000。

        函数首先创建一个`reader_writer_class`的实例`rw`，然后使用它来读取图像和labels。

        然后，它使用`crop_to_nonzero`函数对图像和分割结果进行裁剪，以去除图像和分割结果中的零值区域。

        接下来，它使用`DatasetFingerprintExtractor.collect_foreground_intensities`函数来收集前景区域的强度，并计算前景强度的统计信息。

        接着，它获取图像的间距和裁剪前后的形状，并计算裁剪后的图像相对大小。

        最后，函数返回裁剪后的图像形状、间距、前景强度和统计信息、裁剪后的图像相对大小。
        这里的分割图像到底指的是什么？--->预处理的话不可能上来就分割！
        """
        rw = reader_writer_class()
        images, properties_images = rw.read_images(image_files)
        segmentation, properties_seg = rw.read_seg(segmentation_file)

        # we no longer crop and save the cropped images before this is run. Instead we run the cropping on the fly.
        # Downside is that we need to do this twice (once here and once during preprocessing). Upside is that we don't
        # need to save the cropped data anymore. Given that cropping is not too expensive it makes sense to do it this
        # way. This is only possible because we are now using our new input/output interface.
        data_cropped, seg_cropped, bbox = crop_to_nonzero(images, segmentation)

        foreground_intensities_per_channel, foreground_intensity_stats_per_channel = \
            DatasetFingerprintExtractor.collect_foreground_intensities(seg_cropped, data_cropped,
                                                                       num_samples=num_samples)

        spacing = properties_images['spacing']

        shape_before_crop = images.shape[1:]
        shape_after_crop = data_cropped.shape[1:]
        relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(shape_before_crop)
        #裁剪后的图像相对大小
        return shape_after_crop, spacing, foreground_intensities_per_channel, foreground_intensity_stats_per_channel, \
               relative_size_after_cropping

    def run(self, overwrite_existing: bool = False) -> dict:
        """
    这段代码是一个类方法`run`，它接受一个布尔值参数`overwrite_existing`，并返回一个字典。
    首先，代码创建一个输出文件夹`preprocessed_output_folder`，并确保该文件夹存在。然后，它确定存储属性文件`properties_file`的路径。
    接下来，代码检查属性文件是否存在或是否需要覆盖现有文件。如果属性文件不存在或需要覆盖现有文件，则执行以下操作：
    - 根据数据集的信息确定读写器类`reader_writer_class`。
    - 确定每个训练案例需要采样的前景体素数量`num_foreground_samples_per_case`。
    - 使用多进程池`multiprocessing.Pool`并行处理数据集中的每个案例。对于每个案例，
    调用`DatasetFingerprintExtractor.analyze_case`方法进行分析，将结果存储在列表`r`中。
    - 使用进度条`tqdm`来跟踪处理的案例数量。
    - 等待所有案例的处理完成。
    完成后，代码从结果列表`r`中提取并组织一些统计信息，包括裁剪后的形状、间距、前景像素强度统计信息和裁剪后的图像相对大小。
    然后，代码计算每个通道的前景像素强度的统计信息，并将其存储在字典`intensity_statistics_per_channel`中。
    最后，代码创建一个包含所有统计信息的字典`fingerprint`，并将其保存到属性文件中。
    如果属性文件已存在且不需要覆盖现有文件，则直接加载属性文件并返回其中的内容。
    希望这次解释能更清楚地理解这段代码的功能。如果还有其他问题，请随时提问。
        """
        # we do not save the properties file in self.input_folder because that folder might be read-only. We can only
        # reliably write in nnUNet_preprocessed and nnUNet_results, so nnUNet_preprocessed it is
        preprocessed_output_folder = join(nnUNet_preprocessed, self.dataset_name)
        maybe_mkdir_p(preprocessed_output_folder)
        properties_file = join(preprocessed_output_folder, 'dataset_fingerprint.json')

        if not isfile(properties_file) or overwrite_existing:
            reader_writer_class = determine_reader_writer_from_dataset_json(self.dataset_json,
                                                                            # yikes. Rip the following line
                                                                            self.dataset[self.dataset.keys().__iter__().__next__()]['images'][0])

            # determine how many foreground voxels we need to sample per training case
            num_foreground_samples_per_case = int(self.num_foreground_voxels_for_intensitystats //
                                                  len(self.dataset))

            r = []
            with multiprocessing.get_context("spawn").Pool(self.num_processes) as p:
                for k in self.dataset.keys():
                    r.append(p.starmap_async(DatasetFingerprintExtractor.analyze_case,
                                             ((self.dataset[k]['images'], self.dataset[k]['label'], reader_writer_class,
                                               num_foreground_samples_per_case),)))
                remaining = list(range(len(self.dataset)))
                # p is pretty nifti. If we kill workers they just respawn but don't do any work.
                # So we need to store the original pool of workers.
                workers = [j for j in p._pool]
                with tqdm(desc=None, total=len(self.dataset), disable=self.verbose) as pbar:
                    while len(remaining) > 0:
                        all_alive = all([j.is_alive() for j in workers])
                        if not all_alive:
                            raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                               'OK jokes aside.\n'
                                               'One of your background processes is missing. This could be because of '
                                               'an error (look for an error message) or because it was killed '
                                               'by your OS due to running out of RAM. If you don\'t see '
                                               'an error message, out of RAM is likely the problem. In that case '
                                               'reducing the number of workers might help')
                        done = [i for i in remaining if r[i].ready()]
                        for _ in done:
                            pbar.update()
                        remaining = [i for i in remaining if i not in done]
                        sleep(0.1)

            # results = ptqdm(DatasetFingerprintExtractor.analyze_case,
            #                 (training_images_per_case, training_labels_per_case),
            #                 processes=self.num_processes, zipped=True, reader_writer_class=reader_writer_class,
            #                 num_samples=num_foreground_samples_per_case, disable=self.verbose)
            results = [i.get()[0] for i in r]

            shapes_after_crop = [r[0] for r in results]
            spacings = [r[1] for r in results]
            foreground_intensities_per_channel = [np.concatenate([r[2][i] for r in results]) for i in
                                                  range(len(results[0][2]))]
            # we drop this so that the json file is somewhat human readable
            # foreground_intensity_stats_by_case_and_modality = [r[3] for r in results]
            median_relative_size_after_cropping = np.median([r[4] for r in results], 0)

            num_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())
            intensity_statistics_per_channel = {}
            for i in range(num_channels):
                intensity_statistics_per_channel[i] = {
                    'mean': float(np.mean(foreground_intensities_per_channel[i])),
                    'median': float(np.median(foreground_intensities_per_channel[i])),
                    'std': float(np.std(foreground_intensities_per_channel[i])),
                    'min': float(np.min(foreground_intensities_per_channel[i])),
                    'max': float(np.max(foreground_intensities_per_channel[i])),
                    'percentile_99_5': float(np.percentile(foreground_intensities_per_channel[i], 99.5)),
                    'percentile_00_5': float(np.percentile(foreground_intensities_per_channel[i], 0.5)),
                }

            fingerprint = {
                    "spacings": spacings,
                    "shapes_after_crop": shapes_after_crop,
                    'foreground_intensity_properties_per_channel': intensity_statistics_per_channel,
                    "median_relative_size_after_cropping": median_relative_size_after_cropping
                }

            try:
                save_json(fingerprint, properties_file)
            except Exception as e:
                if isfile(properties_file):
                    os.remove(properties_file)
                raise e
        else:
            fingerprint = load_json(properties_file)
        return fingerprint


if __name__ == '__main__':
    dfe = DatasetFingerprintExtractor(2, 8)
    dfe.run(overwrite_existing=False)
