## Supported Transforms Set

### native torchlm's transforms

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

### transforms from torchvision

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

### transforms from albumentations

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