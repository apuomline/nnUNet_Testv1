from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy import number


class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None
    """
    use_mask_for_norm :是否使用掩码进行归一化处理
    intensityproperties: 强度属性的字典，
     target_dtype： 目标数据类型
    针对于不同的医疗图像使用不同的归一化处理
    """
    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        assert isinstance(intensityproperties, dict)
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass


#类中的分割图像从哪里来呢？ 从哪里获取到的？
class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True
    """
    这段代码实现了Z-Score标准化（Z-Score Normalization）的图像归一化方法。
    Z-Score标准化是一种常用的统计方法，用于将数据转换为具有零均值和单位方差的分布。
    该类继承自ImageNormalization类，并重写了其中的run方法。
    run方法接受一个图像数组image和一个分割数组seg（可选），并返回归一化后的图像数组。
    首先，将图像数组的数据类型转换为目标数据类型（target_dtype）。
    如果use_mask_for_norm参数不为空且为True，则表示要使用分割数组进行归一化。
    在这种情况下，分割数组中的负值表示图像的“外部”区域，例如BraTS数据集中的脑部周围的零值。
    我们只希望在脑部区域进行归一化，因此需要对图像进行掩码操作。使用分割数组中大于等于0的像素作为掩码，
    计算掩码区域内像素的均值和标准差，
    然后对掩码区域内的像素进行Z-Score标准化。最后，将归一化后的像素值赋值给图像数组的对应位置。
    如果use_mask_for_norm参数为空或为False，则表示不使用分割数组进行归一化。在这种情况下，直接计算整个图像数组的均值和标准差，并对整个图像数组进行Z-Score标
    """
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (max(std, 1e-8))
        return image


class CTNormalization(ImageNormalization):
    """
    这段代码实现了CT归一化（CTNormalization）的图像归一化方法。
    CT归一化是针对CT扫描图像的一种特殊的归一化方法，旨在将CT图像的像素值映射到特定的范围内。
    该类同样继承自ImageNormalization类，并重写了其中的run方法。
    run方法接受一个图像数组image和一个分割数组seg（可选），并返回归一化后的图像数组。
    首先，将图像数组的数据类型转换为目标数据类型（target_dtype）。
    然后，断言确保intensityproperties参数不为空，
    因为CT归一化需要使用intensityproperties参数来定义归一化范围。
    intensityproperties是一个字典，包含了CT图像的均值、标准差以及百分位数的信息。
    接下来，将图像数组中的像素值限制在intensityproperties中定义的范围内，通过np.clip函数实现。
    这样做是为了去除图像中的异常值或噪声，并确保像素值在合理的范围内。
    最后，将图像数组中的像素值减去均值，再除以标准差，实现CT归一化。
    同时，为了避免除以零的错误，使用max(std_intensity, 1e-8)来确保标准差不为零。
    最终，返回归一化后的图像数组。
    """
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        image = image.astype(self.target_dtype)
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']
        image = np.clip(image, lower_bound, upper_bound)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        return image


class NoNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image.astype(self.target_dtype)


class RescaleTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype)
        image = image - image.min()
        image = image / np.clip(image.max(), a_min=1e-8, a_max=None)
        return image


class RGBTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert image.min() >= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                 "Your images do not seem to be RGB images"
        assert image.max() <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                   ". Your images do not seem to be RGB images"
        image = image.astype(self.target_dtype)
        image = image / 255.
        return image

