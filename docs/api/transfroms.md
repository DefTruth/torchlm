# Supported Transforms Set

## native torchlm's transforms

```python
__all__ = [
    "LandmarksCompose",
    "LandmarksNormalize",
    "LandmarksUnNormalize",
    "LandmarksToTensor",
    "LandmarksToNumpy",
    "LandmarksResize",
    "LandmarksClip",
    "LandmarksAlign",
    "LandmarksRandomAlign",
    "LandmarksRandomCenterCrop",
    "LandmarksRandomHorizontalFlip",
    "LandmarksHorizontalFlip",
    "LandmarksRandomScale",
    "LandmarksRandomTranslate",
    "LandmarksRandomRotate",
    "LandmarksRandomShear",
    "LandmarksRandomHSV",
    "LandmarksRandomMask",
    "LandmarksRandomBlur",
    "LandmarksRandomBrightness",
    "LandmarksRandomPatches",
    "LandmarksRandomBackground",
    "LandmarksRandomPatchesWithAlpha",
    "LandmarksRandomBackgroundWithAlpha",
    "LandmarksRandomMaskWithAlpha",
    "BindAlbumentationsTransform",
    "BindTorchVisionTransform",
    "BindArrayCallable",
    "BindTensorCallable",
    "BindEnum",
    "bind",
    "set_transforms_logging",
    "set_transforms_debug"
]
```  

## transforms from torchvision

```python
# torchvision >= 0.9.0
_Supported_Image_Only_Transform_Set: Tuple = (
    torchvision.transforms.Normalize,
    torchvision.transforms.ColorJitter,
    torchvision.transforms.Grayscale,
    torchvision.transforms.RandomGrayscale,
    torchvision.transforms.RandomErasing,
    torchvision.transforms.GaussianBlur,
    torchvision.transforms.RandomInvert,
    torchvision.transforms.RandomPosterize,
    torchvision.transforms.RandomSolarize,
    torchvision.transforms.RandomAdjustSharpness,
    torchvision.transforms.RandomAutocontrast,
    torchvision.transforms.RandomEqualize
)
```

## transforms from albumentations

```python
# albumentations >= v 1.1.0
_Supported_Image_Only_Transform_Set: Tuple = (
    albumentations.Blur,
    albumentations.CLAHE,
    albumentations.ChannelDropout,
    albumentations.ChannelShuffle,
    albumentations.ColorJitter,
    albumentations.Downscale,
    albumentations.Emboss,
    albumentations.Equalize,
    albumentations.FDA,
    albumentations.FancyPCA,
    albumentations.FromFloat,
    albumentations.GaussNoise,
    albumentations.GaussianBlur,
    albumentations.GlassBlur,
    albumentations.HistogramMatching,
    albumentations.HueSaturationValue,
    albumentations.ISONoise,
    albumentations.ImageCompression,
    albumentations.InvertImg,
    albumentations.MedianBlur,
    albumentations.MotionBlur,
    albumentations.Normalize,
    albumentations.PixelDistributionAdaptation,
    albumentations.Posterize,
    albumentations.RGBShift,
    albumentations.RandomBrightnessContrast,
    albumentations.RandomFog,
    albumentations.RandomGamma,
    albumentations.RandomRain,
    albumentations.RandomShadow,
    albumentations.RandomSnow,
    albumentations.RandomSunFlare,
    albumentations.RandomToneCurve,
    albumentations.Sharpen,
    albumentations.Solarize,
    albumentations.Superpixels,
    albumentations.TemplateTransform,
    albumentations.ToFloat,
    albumentations.ToGray
)

_Supported_Dual_Transform_Set: Tuple = (
    albumentations.Affine,
    albumentations.CenterCrop,
    albumentations.CoarseDropout,
    albumentations.Crop,
    albumentations.CropAndPad,
    albumentations.CropNonEmptyMaskIfExists,
    albumentations.Flip,
    albumentations.HorizontalFlip,
    albumentations.Lambda,
    albumentations.LongestMaxSize,
    albumentations.NoOp,
    albumentations.PadIfNeeded,
    albumentations.Perspective,
    albumentations.PiecewiseAffine,
    albumentations.RandomCrop,
    albumentations.RandomCropNearBBox,
    albumentations.RandomGridShuffle,
    albumentations.RandomResizedCrop,
    albumentations.RandomRotate90,
    albumentations.RandomScale,
    albumentations.RandomSizedCrop,
    albumentations.Resize,
    albumentations.Rotate,
    albumentations.SafeRotate,
    albumentations.ShiftScaleRotate,
    albumentations.SmallestMaxSize,
    albumentations.Transpose,
    albumentations.VerticalFlip
)
```

## torchlm原生数据增强方法API文档

* LandmarksNormalize: 对img进行归一化处理，x = (x - mean) / std，对于landmarks不做任何处理。初始化参数说明，mean: float, 均值，默认为127.5；std: float, 标准差，默认128.0；

```python
class LandmarksNormalize(LandmarksTransform):
   def __init__(
            self,
            mean: float = 127.5,
            std: float = 128.
   ):
```

* LandmarksUnNormalize：对img进行反归一化处理，x = x * std + std，对于landmarks不做任何处理。初始化参数说明，mean: float, 均值，默认为127.5；std: float, 标准差，默认128.0；

```python
class LandmarksUnNormalize(LandmarksTransform):
    def __init__(
            self,
            mean: float = 127.5,
            std: float = 128.
    ):
```

* LandmarksToTensor：将img和landmarks从numpy数组转换成Tensor，无初始化参数

```python
class LandmarksToTensor(LandmarksTransform):
    def __init__(self):
```

* LandmarksToNumpy: 将img和landmarks从Tensor转换成numpy数组，无初始化参数

```python
class LandmarksToNumpy(LandmarksTransform):
    def __init__(self):
```

* LandmarksResize: 对img进行resize，并且调整landmarks坐标。初始化参数说明，size: Union[Tuple[int, int], int]，目标图片resize后大小，顺序为(w,h)；keep_aspect: bool，是否保持尺度不变，默认False；如果设置为True，则保持输入img宽高比率不变，并对空白区域进行零值padding。

```python
class LandmarksResize(LandmarksTransform):
    def __init__(
            self,
            size: Union[Tuple[int, int], int],
            keep_aspect: bool = False
    ):
```

* LandmarksClip：以landmarks外接矩形框为中心，在img上进行裁剪操作。比如我们通常要去掉一些冗余的背景区域，只在landmarks外接矩形的某个延展区域内进行估计点位。初始化参数说明，width_pad: float，默认0.2，相对于landmarks的宽的扩张比率，扩张后的框宽为landmarks外接矩形的(width_pad+1.+width_pad)倍；height_pad: float，默认0.2，相对于landmarks的高的扩张比率，扩张后的框高为landmarks外接矩形的(height_pad+1.+height_pad)倍；target_size：Union[Tuple[int, int], int]，默认为None，这个表示clip之后，是否要再resize到某个指定的尺寸；如果要对clip后得到的img重新resize到指定的target_size，则kwargs，表示传给LandmarksResize的参数，否则该参数无任何作用。

```python
class LandmarksClip(LandmarksTransform):
    def __init__(
            self,
            width_pad: float = 0.2,
            height_pad: float = 0.2,
            target_size: Union[Tuple[int, int], int] = None,
            **kwargs
    ):
```

* LandmarksRandomCenterCrop: 随机中心裁剪，将会以img的中心点(cx,cy)为中心，按照指定的宽高可选的比率范围，随机选择宽高比率进行裁剪。该改变换会根据图片实际裁剪的比率变动landmarks的坐标，并且会自动检查点位坐标是否越界的“安全性”，确保crop的区域一定会包含所有的点位。一般来说，随机中心裁剪对**模拟人脸检测器带来的抖动比较有用**，避免对某种固定尺寸比率的人脸矩形框过拟合。实际应用的时候，通常无法保证上游的人脸检测器是不会更改的，所以这种人脸框的抖动几乎是必然存在的。随机中心裁剪就可以模拟这种情形。初始化参数说明，width_range: Tuple[float, float]，宽的比率选择范围，默认为，(0.8, 1.0)，假设随机选择了一个比率w_r，原来图像的宽为img_w，则最后crop的宽是(w_r * img_w)，并且这个crop不是以(cx,cy)为中心对称的，而是会在确保包含所有点位的基础上，随机左右偏移；height_range: Tuple[float, float]，宽的比率选择范围，默认为，(0.8, 1.0)，假设随机选择了一个比率h_r，原来图像的宽为img_h，则最后crop的宽是(h_r * img_h)，并且这个crop不是以(cx,cy)为中心对称的，而是会在确保包含所有点位的基础上，随机上下偏移；prob: float，默认0.5，发生的概率。

```python
class LandmarksRandomCenterCrop(LandmarksTransform):
    def __init__(
            self,
            width_range: Tuple[float, float] = (0.8, 1.0),
            height_range: Tuple[float, float] = (0.8, 1.0),
            prob: float = 0.5
    ):
```
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomCenterCrop(width_range=(0.5, 0.1), height_range=(0.5, 0.1), prob=1.),
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_center_crop_101.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_center_crop_125.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_center_crop_205.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_center_crop_291.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_center_crop_64.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_center_crop_497.jpg' height="100px" width="100px">
</div>  

* LandmarksHorizontalFlip: 水平镜像翻转，无初始化参数。注意，这个方法会改变点位的索引顺序，所以一般不要使用，除非你的任务是允许进行镜像的，否则你的loss可能一直无法收敛哦。这是个坑来的（亲自踩过的），这里还是实现了这个方法，留给有缘人踩一踩。

```python
class LandmarksHorizontalFlip(LandmarksTransform):
    """WARNING: HorizontalFlip augmentation mirrors the input image. When you apply
     that augmentation to keypoints that mark the side of body parts (left or right),
     those keypoints will point to the wrong side (since left on the mirrored image
     becomes right). So when you are creating an augmentation pipeline look carefully
     which augmentations could be applied to the input data. Also see:
     https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    """
    def __init__(self):
```

* LandmarksRandomHorizontalFlip: 随机版本的水平镜像翻转，初始化参数说明，prob: float，默认0.5，发生的概率。这个是随机版本的坑，更坑，是留给更有缘人去踩的。如果你的任务本质上不允许进行镜像，而你又用了这个数据增强，那么恭喜你，有缘人，你更难发现这个坑了，因为这个loss可能时而下降时而上升，也可能是收敛到某个看似正常的地方却一直下不去了（当镜像发生的概率很小时）。**用咱们修真界的话来讲就是，这就是一颗传说中的“天品毒丹”啊，非有缘人而不可得，实乃调参炼药坑己坑人，必备良药啊，请各位有缘人务必收好**。

```python
class LandmarksRandomHorizontalFlip(LandmarksTransform):
    """WARNING: HorizontalFlip augmentation mirrors the input image. When you apply
     that augmentation to keypoints that mark the side of body parts (left or right),
     those keypoints will point to the wrong side (since left on the mirrored image
     becomes right). So when you are creating an augmentation pipeline look carefully
     which augmentations could be applied to the input data. Also see:
     https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    """
    def __init__(
            self,
            prob: float = 0.5
    ):
```

* LandmarksAlign：人脸对齐。这个主要是用在人脸关键点检测的任务上，并且不一定是数据增强，主要为为了便于获得一张对齐后的人脸和对齐后的关键点。初始化参数说明，eyes_index: Union[Tuple[int, int], List[int]]，必须要指定的左右眼中心点在landmarks中的索引。

```python
class LandmarksAlign(LandmarksTransform):
    def __init__(
            self,
            eyes_index: Union[Tuple[int, int], List[int]] = None
    ):
```
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomRotate(80, prob=1.),  # 先增加旋转
        torchlm.LandmarksRandomAlign(eyes_index=(96, 97), prob=1.),  # 再进行对齐看效果
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_align_494.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_align_720.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_align_886.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_align_901.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_align_908.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_align_920.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomAlign：随机版本的人脸对齐。这个主要是用在人脸关键点检测的任务上，可用作数据增强。可能出于某种奇怪的原因，你想随机地将一部分人脸根据关键点进行对齐后，作为训练数据，那么就可以使用这个数据增强方法。初始化参数说明，eyes_index: Union[Tuple[int, int], List[int]]，必须要指定的左右眼中心点在landmarks中的索引；prob: float，默认0.5，发生的概率。

```python
class LandmarksRandomAlign(LandmarksTransform):
    def __init__(
            self,
            eyes_index: Union[Tuple[int, int], List[int]] = None,
            prob: float = 0.5
    ):
```

* LandmarksRandomScale：随机尺度变换。从指定的scale可选值域范围内挑选比率，对输入的图片进行尺度变换，拉伸变形，并同时调整landmarks的坐标。初始化参数说明，scale: Union[Tuple[float, float], float]，scale变化的范围比率，相对于原图片的宽高；prob: float，默认0.5，发生的概率；diff: bool，是否让宽高的比率是随机不一致的，默认True，若为False，对图片按照宽高选择到的相同比率进行拉伸。

```python
class LandmarksRandomScale(LandmarksTransform):
    def __init__(
            self,
            scale: Union[Tuple[float, float], float] = 0.4,
            prob: float = 0.5,
            diff: bool = True
    ):
```
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomScale(scale=(-0.5, 1.5), prob=1.),
        torchlm.LandmarksResize((256, 256), keep_aspect=True)
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_scale_215.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_scale_220.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_scale_255.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_scale_362.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_scale_415.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_scale_783.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomShear：随机剪切变化。对图片进行剪切变化，并同时调整landmarks坐标。初始化参数说明，shear_factor: Union[Tuple[float, float], List[float], float]，随机剪切的比率范围，通常是(-1.,1)之间；prob: float，默认0.5，发生的概率；

```python
class LandmarksRandomShear(LandmarksTransform):
    def __init__(
            self,
            shear_factor: Union[Tuple[float, float], List[float], float] = 0.2,
            prob: float = 0.5
    ):
```
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomShear(prob=1.),
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_shear_2.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_shear_114.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_shear_163.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_shear_284.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_shear_438.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_shear_460.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomHSV: 随机HSV颜色空间变换。对图片进行颜色HSV的颜色空间的随机像素调整，这是一个image-only变换，不改变landmarks的坐标位置。初始化参数说明，hue: Union[Tuple[int, int], int]，色度色温变画范围，建议(-20,+20)之间；saturation: Union[Tuple[int, int], int]，饱和度变化范围，建议(-20,+20)之间；brightness: Union[Tuple[int, int], int]，亮度变化范围，建议(-20,+20)之间；prob: float，默认0.5，发生的概率；

```python
class LandmarksRandomHSV(LandmarksTransform):
    def __init__(
            self,
            hue: Union[Tuple[int, int], int] = 20,
            saturation: Union[Tuple[int, int], int] = 20,
            brightness: Union[Tuple[int, int], int] = 20,
            prob: float = 0.5
    ):
```
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomHSV(prob=1.),
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_hsv_301.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_hsv_357.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_hsv_500.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_hsv_516.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_hsv_739.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_hsv_861.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomTranslate：随机平移变换。对输入的图像，从给定的平移比率范围内随机选择一个比率进行平移变化，并且对应改变landmarks的坐标位置。平移变换有可能会使得部分landmarks越界，这也在内部做了数据的"安全性"判断，会自动撤销不安全的变换，确保landmarks在变换前后数量的一致性。初始化参数说明，translate: Union[Tuple[float, float], float]，平移的比率范围，建议是(-0.2,0.2)；prob: float，默认0.5，发生的概率；diff: bool，是否在宽高方向使用不同的平移比率，默认为False；

```python
class LandmarksRandomTranslate(LandmarksTransform):
    def __init__(
            self,
            translate: Union[Tuple[float, float], float] = 0.2,
            prob: float = 0.5,
            diff: bool = False
    ):
```
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomTranslate(prob=1.),
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_translate_7.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_translate_79.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_translate_119.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_translate_128.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_translate_130.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_translate_204.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomRotate: 随机旋转变换。对输入的图像，从给定的角度范围内随机选择一个角度进行旋转变化，并且对应改变landmarks的坐标位置。旋转变换有可能会使得部分landmarks越界，这也在内部做了数据的"安全性"判断，会自动撤销不安全的变换，确保landmarks在变换前后数量的一致性。初始化参数说明，angle: Union[Tuple[int, int], List[int], int]，角度变化的范围，建议(-90,+90)；prob: float，默认0.5，发生的概率；bins: Optional[int]，可以指定将angle范围内的角度，等距离地划分为多少bin；

```python
class LandmarksRandomRotate(LandmarksTransform):
    def __init__(
            self,
            angle: Union[Tuple[int, int], List[int], int] = 10,
            prob: float = 0.5,
            bins: Optional[int] = None
    ):
```
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomRotate(angle=80, prob=1.),
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_rotate_236.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_rotate_238.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_rotate_641.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_rotate_709.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_rotate_724.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_rotate_831.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomBlur: 随机高斯模糊。对输入的图像，从给定的kernel范围内随机选择一个kernel，进行高斯模糊，该变换是image-only变换，不改变landmarks的坐标位置。这种变换可用用来模拟模糊的场景，能够提高模型的泛化能力。可以想象，如果你的关键点检测器都是在很清晰的图片上训练的，那么实际应用的时候，遇到模糊的场景可能就完了，况且，模糊是一种很常见的场景。初始化参数说明，kernel_range: Tuple[int, int]，高斯kernel的选择范围，值越大越模糊，默认是(3, 11)；prob: float，默认0.5，发生的概率；sigma_range: Tuple[int, int] ，方差的选择范围，默认 (0, 4)，用默认的就行。

```python
class LandmarksRandomBlur(LandmarksTransform):
    def __init__(
            self,
            kernel_range: Tuple[int, int] = (3, 11),
            prob: float = 0.5,
            sigma_range: Tuple[int, int] = (0, 4)
    ):
```
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksResize((256, 256)),
        torchlm.LandmarksRandomBlur(kernel_range=(5, 35), prob=1.)
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_blur_126.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_blur_289.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_blur_300.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_blur_317.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_blur_453.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_blur_456.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomBrightness: 随机亮度和对比度变化。对输入的图像，从给定的亮度和对比度变化范围内随机选择值，进行亮度和对比度变化，该变换是image-only变换，不改变landmarks的坐标位置。初始化参数说明，brightness: Tuple[float, float]，亮度变化范围，默认 (-30., 30.)；contrast: Tuple[float, float]，对比度变化范围，默认(0.5, 1.5)；prob: float，默认0.5，发生的概率；

```python
class LandmarksRandomBrightness(LandmarksTransform):
    def __init__(
            self,
            brightness: Tuple[float, float] = (-30., 30.),
            contrast: Tuple[float, float] = (0.5, 1.5),
            prob: float = 0.5
    ):
```
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomBrightness(prob=1.),
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_brightness_247.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_brightness_247.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_brightness_247.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_brightness_247.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_brightness_247.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_brightness_247.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomMask: 随机遮挡变换。从给定的mask面积比率范围内随机选择一个面积比率，在原图上随机选一块该面积大小，矩形形状随机的方块进行mask，并且mask的值也是随机的。该变换是image-only变换，不改变landmarks的坐标位置。初始化参数说明，mask_ratio: float，遮挡的比率最大值，默认0.1，指被遮挡区域占输入图片的面积的比率；prob: float，默认0.5，发生的概率；trans_ratio: float，默认0.5，控制mask区域的随机形状，用默认值行。

```python
class LandmarksRandomMask(LandmarksTransform):
    def __init__(
            self,
            mask_ratio: float = 0.1,
            prob: float = 0.5,
            trans_ratio: float = 0.5
    ):
```  
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomMask(prob=1.),
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_mask_alpha_False_303.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_mask_alpha_False_320.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_mask_alpha_False_474.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_mask_alpha_False_506.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_mask_alpha_False_550.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_mask_alpha_False_750.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomMaskMixUp：带Alpha融合的随机遮挡变化。torchlm特有的。 随机遮挡变换。从给定的mask面积比率范围内随机选择一个面积比率，在原图上随机选一块该面积大小，矩形形状随机的方块，按照随机挑选的alpha比率进行alpha融合，制造类似阴影的效果，mask的值也是随机的。该变换是image-only变换，不改变landmarks的坐标位置。初始化参数说明，mask_ratio: float，遮挡的比率最大值，默认0.1，指被遮挡区域占输入图片的面积的比率；prob: float，默认0.5，发生的概率；trans_ratio: float，默认0.5，控制mask区域的随机形状，用默认值行； alpha: float，mask可选最大alpha值，默认0.9；

```python
class LandmarksRandomMaskMixUp(LandmarksTransform):

    def __init__(
            self,
            mask_ratio: float = 0.25,
            prob: float = 0.5,
            trans_ratio: float = 0.5,
            alpha: float = 0.9
    ):
```  
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomMaskMixUp(prob=1.),
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_mask_alpha_True_200.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_mask_alpha_True_277.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_mask_alpha_True_390.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_mask_alpha_True_580.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_mask_alpha_True_642.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_mask_alpha_True_734.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomPatches：随机打补丁变换。这是torchlm特有的，该版本类似随机mask，其实是一个假的MixUp，这里没有使用alpha进行融合；该类型会把随机选择到的patch图像块，随机地填充到原图的一块区域，该类型需要指定patches图片集。当然，除了可以手动指定patches图片集合的路径外，也可以使用默认的，torchlm已经内置了100张随机的背景图片，大家可以不用额外准备这个图片集。初始化参数说明，patch_dirs: List[str]，指定使用的patches图片集，若不指定则会使用torchlm自带的；patch_ratio: float，最大的打补丁面积，默认0.15；prob: float，默认0.5，发生的概率；trans_ratio: float，默认0.5，控制打补丁区域的随机形状，用默认值行。

```python
class LandmarksRandomPatches(LandmarksTransform):

    def __init__(
            self,
            patch_dirs: List[str] = None,
            patch_ratio: float = 0.15,
            prob: float = 0.5,
            trans_ratio: float = 0.5
    ):
```  
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomPatches(prob=1.),
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_patches_mixup_alpha_False_26.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_patches_mixup_alpha_False_75.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_patches_mixup_alpha_False_498.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_patches_mixup_alpha_False_540.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_patches_mixup_alpha_False_571.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_patches_mixup_alpha_False_593.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomPatchesMixUp：带Alpha融合的随机打补丁变换。这是torchlm特有的，针对关键点检测的改造的一个MixUp版本，在分类和目标检测领域大名鼎鼎的MixUp，当然可以考虑修改一下用到关键点检测当中。该类型会把随机选择到的patch图像块，根据随机选择的alpha比率，随机地填充到原图的一块区域，该类型需要指定patches图片集。当然，除了可以手动指定patches图片集合的路径外，也可以使用默认的，torchlm已经内置了100张随机的背景图片，大家可以不用额外准备这个图片集。初始化参数说明，patch_dirs: List[str]，指定使用的patches图片集，若不指定则会使用torchlm自带的；patch_ratio: float，最大的打补丁面积，默认0.15；prob: float，默认0.5，发生的概率；trans_ratio: float，默认0.5，控制打补丁区域的随机形状，用默认值行；alpha: float，patch可选最大alpha值，默认0.9；

```python
class LandmarksRandomPatchesMixUp(LandmarksTransform):
    def __init__(
            self,
            patch_dirs: List[str] = None,
            patch_ratio: float = 0.2,
            prob: float = 0.5,
            trans_ratio: float = 0.5,
            alpha: float = 0.9
    ):
```  
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomPatchesMixUp(alpha=0.5, prob=1.),
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_patches_mixup_alpha_True_130.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_patches_mixup_alpha_True_136.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_patches_mixup_alpha_True_343.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_patches_mixup_alpha_True_432.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_patches_mixup_alpha_True_477.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_patches_mixup_alpha_True_489.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomBackground：随机背景替换变换。torchlm特有的。随机从指定的背景图片集合中选择一张背景图片，从中随机裁剪一块，按照填充的方式，将输入的图片填充到选定的背景上，该变换会改变landmarks的坐标位置。初始化参数说明，background_dirs: List[str]，指定使用的backgrounds图片集，若不指定则会使用torchlm自带的；prob: float，默认0.5，发生的概率；

```python
class LandmarksRandomBackground(LandmarksTransform):
    def __init__(
            self,
            background_dirs: List[str] = None,
            prob: float = 0.5
    ):
```  
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomBackground(prob=1.),
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_backgrounds_mixup_alpha_False_416.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_backgrounds_mixup_alpha_False_452.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_backgrounds_mixup_alpha_False_608.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_backgrounds_mixup_alpha_False_761.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_backgrounds_mixup_alpha_False_771.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_backgrounds_mixup_alpha_False_873.jpg' height="100px" width="100px">
</div>  

* LandmarksRandomBackgroundMixUp：带Alpha融合的随机背景替换变换。torchlm特有的。是我实现的MixUp在关键点检测的一个特例。随机从指定的背景图片集合中选择一张背景图片，从中随机裁剪一块，按照随机选择一个alpha比率融合到输入的图片上，该变换是image-only变换，不会改变Landmarks的坐标位置。初始化参数说明，background_dirs: List[str]，指定使用的backgrounds图片集，若不指定则会使用torchlm自带的；prob: float，默认0.5，发生的概率；alpha: float，background可选最大alpha值，默认0.3，注意这个值不能太大；

```python
class LandmarksRandomBackgroundMixUp(LandmarksTransform):
    def __init__(
            self,
            background_dirs: List[str] = None,
            alpha: float = 0.3,
            prob: float = 0.5
    ):
```  
```python
transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomBackgroundMixUp(alpha=0.5, prob=1.),
        torchlm.LandmarksResize((256, 256))
    ])
```
<div align='center'>
  <img src='../res/2_wflw_orginal.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_backgrounds_mixup_alpha_True_42.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_backgrounds_mixup_alpha_True_129.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_backgrounds_mixup_alpha_True_988.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_backgrounds_mixup_alpha_True_169.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_backgrounds_mixup_alpha_True_220.jpg' height="100px" width="100px">
  <img src='../res/2_wflw_backgrounds_mixup_alpha_True_233.jpg' height="100px" width="100px">
</div>  

* BindAlbumentationsTransform：绑定albumentations数据增强方法的wrapper类，不建议直接使用。初始化参数说明，transform，指定一个albumentations的transform；prob，发生的概率，和torchlm.bind中的prob参数含义相同。

```python
class BindAlbumentationsTransform(LandmarksTransform):
    def __init__(
            self,
            transform: Albumentations_Transform_Type,
            prob: float = 1.0
    ):
```  

* BindTorchVisionTransform：绑定torchvision数据增强方法的wrapper类，不建议直接使用。初始化参数说明，transform，指定一个torchvision的transform；prob，发生的概率，和torchlm.bind中的prob参数含义相同。
  
```python
class BindTorchVisionTransform(LandmarksTransform):
    def __init__(
            self,
            transform: TorchVision_Transform_Type,
            prob: float = 1.0
    ):
```  

* BindArrayCallable：绑定自定义numpy输入输出数据增强方法的wrapper类，不建议直接使用。初始化参数说明，call_func，指定一个numpy输入输出的自定义数据增强函数；prob，发生的概率，和torchlm.bind中的prob参数含义相同。
  
```python
class BindArrayCallable(LandmarksTransform):
    def __init__(
            self,
            call_func: Callable_Array_Func_Type,
            prob: float = 1.0
    ):
```  

* BindTensorCallable：绑定自定义Tensor输入输出数据增强方法的wrapper类，不建议直接使用。初始化参数说明，call_func，指定一个Tensor输入输出的自定义数据增强函数；prob，发生的概率，和torchlm.bind中的prob参数含义相同。
  
```python
class BindTensorCallable(LandmarksTransform):
    def __init__(
            self,
            call_func: Callable_Tensor_Func_Type,
            prob: float = 1.0
    ):
```  

* LandmarksCompose：关键点数据增强的pipeline。初始化参数说明，transforms，指定一个数据增强类型列表。

```python
class LandmarksCompose(object):

    def __init__(
            self,
            transforms: List[LandmarksTransform]
    ):
```  

