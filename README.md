![torchlm-logo](docs/res/logo.png)    
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-9cf.svg)](https://github.com/DefTruth/torchlm/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
[![Slack](https://img.shields.io/badge/slack-chat-ffa.svg?logo=slack)](https://join.slack.com/t/torchlm/shared_invite/zt-mqwc7235-940aAh8IaKYeWclrJx10SA)
[![PyPI version](https://img.shields.io/pypi/v/torchlm?color=aff)](https://badge.fury.io/py/torchlm)
[![Python Version](https://img.shields.io/pypi/pyversions/torchlm?color=dfd)](https://pypi.org/project/torchlm/)
[![OS](https://img.shields.io/badge/macos|linux|windows-pass-skyblue.svg)](https://pypi.org/project/torchlm/)
[![License](https://img.shields.io/badge/license-MIT-lightblue.svg)](https://pypi.org/project/torchlm/)


## 🤗 Introduction
**torchlm** is a PyTorch landmarks-only library with **100+ data augmentations**, **training** and **inference**. **torchlm** is only focus on any landmarks detection, such as face landmarks, hand keypoints and body keypoints, etc. It provides **30+** native data augmentations and compatible with **80+** torchvision and albumations's transforms, no matter the input is a np.ndarray or a torch Tensor, **torchlm** will **automatically** be compatible with different data types through a **autodtype** wrapper. Further, in the future **torchlm** will add modules for **training** and **inference**.

# 🆕 What's New

* [2022/02/13]: Add **30+** native data augmentations and **bind** 80+ torchvision and albumations's transforms.

## 🛠️ Usage

### Requirements
* opencv-python-headless>=4.5.2
* numpy>=1.14.4
* torch>=1.6.0
* torchvision>=0.9.0
* albumentations>=1.1.0

### Installation
you can install **torchlm** directly from pip.
```shell
pip3 install torchlm
# install from specific pypi mirrors use '-i'
pip3 install torchlm -i https://pypi.org/simple/
```

### Data Augmentation
**torchlm** provides 30+ native data augmentations for landmarks and is compatible with 80+ transforms from torchvision and albumations, no matter the input is a np.ndarray or a torch Tensor, torchlm will automatically be compatible with different data types through a autodtype wrapper. 
* use native torchlm's transforms 
```python
import torchlm
transform = torchlm.LandmarksCompose([
        # use native torchlm transforms
        torchlm.LandmarksRandomHorizontalFlip(prob=0.5),
        torchlm.LandmarksRandomScale(prob=0.5),
        torchlm.LandmarksRandomTranslate(prob=0.5),
        torchlm.LandmarksRandomShear(prob=0.5),
        torchlm.LandmarksRandomMask(prob=0.5),
        torchlm.LandmarksRandomBlur(kernel_range=(5, 25), prob=0.5),
        torchlm.LandmarksRandomBrightness(prob=0.),
        torchlm.LandmarksRandomRotate(40, prob=0.5, bins=8),
        torchlm.LandmarksRandomCenterCrop((0.5, 1.0), (0.5, 1.0), prob=0.5),
        torchlm.LandmarksResize((256, 256)),
        torchlm.LandmarksNormalize(),
        torchlm.LandmarksToTensor(),
        torchlm.LandmarksToNumpy(),
        torchlm.LandmarksUnNormalize()
    ])
```  
* **bind** torchvision and albumations's transform
```python
import torchvision
import albumentations
import torchlm
transform = torchlm.LandmarksCompose([
        # use native torchlm transforms
        torchlm.LandmarksRandomHorizontalFlip(prob=0.5),
        torchlm.LandmarksRandomScale(prob=0.5),
        # ...
        # bind torchvision image only transforms
        torchlm.bind(torchvision.transforms.GaussianBlur(kernel_size=(5, 25))),
        torchlm.bind(torchvision.transforms.RandomAutocontrast(p=0.5)),
        torchlm.bind(torchvision.transforms.RandomAdjustSharpness(sharpness_factor=3, p=0.5)),
        # bind albumentations image only transforms
        torchlm.bind(albumentations.ColorJitter(p=0.5)),
        torchlm.bind(albumentations.GlassBlur(p=0.5)),
        torchlm.bind(albumentations.RandomShadow(p=0.5)),
        # bind albumentations dual transforms
        torchlm.bind(albumentations.RandomCrop(height=200, width=200, p=0.5)),
        torchlm.bind(albumentations.RandomScale(p=0.5)),
        torchlm.bind(albumentations.Rotate(p=0.5)),
        torchlm.LandmarksResize((256, 256)),
        torchlm.LandmarksNormalize(),
        torchlm.LandmarksToTensor(),
        torchlm.LandmarksToNumpy(),
        torchlm.LandmarksUnNormalize()
    ])
```
* **bind** custom callable array or Tensor functions  

```python
# First, defined your custom functions
def callable_array_noop(
        img: np.ndarray,
        landmarks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # Do some transform here ...
    return img.astype(np.uint32), landmarks.astype(np.float32)


def callable_tensor_noop(
        img: Tensor,
        landmarks: Tensor
) -> Tuple[Tensor, Tensor]:
    # Do some transform here ...
    return img, landmarks
```

```python
# Then, bind your functions and put it into transforms pipeline.
transform = torchlm.LandmarksCompose([
        # use native torchlm transforms
        torchlm.LandmarksRandomHorizontalFlip(prob=0.5),
        torchlm.LandmarksRandomScale(prob=0.5),
        # ...
        # bind torchvision image only transforms
        torchlm.bind(torchvision.transforms.GaussianBlur(kernel_size=(5, 25))),
        torchlm.bind(torchvision.transforms.RandomAutocontrast(p=0.5)),
        torchlm.bind(torchvision.transforms.RandomAdjustSharpness(sharpness_factor=3, p=0.5)),
        # bind albumentations image only transforms
        torchlm.bind(albumentations.ColorJitter(p=0.5)),
        torchlm.bind(albumentations.GlassBlur(p=0.5)),
        torchlm.bind(albumentations.RandomShadow(p=0.5)),
        # bind albumentations dual transforms
        torchlm.bind(albumentations.RandomCrop(height=200, width=200, p=0.5)),
        torchlm.bind(albumentations.RandomScale(p=0.5)),
        torchlm.bind(albumentations.Rotate(p=0.5)),
        # bind custom callable array functions
        torchlm.bind(callable_array_noop, bind_type=torchlm.BindEnum.Callable_Array),
        # bind custom callable Tensor functions
        torchlm.bind(callable_tensor_noop, bind_type=torchlm.BindEnum.Callable_Tensor),
        torchlm.LandmarksResize((256, 256)),
        torchlm.LandmarksNormalize(),
        torchlm.LandmarksToTensor(),
        torchlm.LandmarksToNumpy(),
        torchlm.LandmarksUnNormalize()
    ])
```
<div align='center'>
  <img src='docs/res/124.jpg' height="90px" width="90px">
  <img src='docs/res/158.jpg' height="90px" width="90px">
  <img src='docs/res/386.jpg' height="90px" width="90px">
  <img src='docs/res/478.jpg' height="90px" width="90px">
  <img src='docs/res/537.jpg' height="90px" width="90px">
  <img src='docs/res/605.jpg' height="90px" width="90px">
  <img src='docs/res/802.jpg' height="90px" width="90px">
</div>  


* setup logging mode as `True` globally might help you figure out the runtime details
```python
import torchlm
# some global setting
torchlm.set_transforms_debug(True)
torchlm.set_transforms_logging(True)
torchlm.set_autodtype_logging(True)
```
Some details logs will show you at each runtime, just like the follows
```shell
LandmarksRandomHorizontalFlip() AutoDtype Info: AutoDtypeEnum.Array_InOut
LandmarksRandomHorizontalFlip() Execution Flag: True
LandmarksRandomScale() AutoDtype Info: AutoDtypeEnum.Array_InOut
LandmarksRandomScale() Execution Flag: False
...
BindTorchVisionTransform(GaussianBlur())() AutoDtype Info: AutoDtypeEnum.Tensor_InOut
BindTorchVisionTransform(GaussianBlur())() Execution Flag: True
...
BindAlbumentationsTransform(ColorJitter())() AutoDtype Info: AutoDtypeEnum.Array_InOut
BindAlbumentationsTransform(ColorJitter())() Execution Flag: True
...
BindArrayCallable(callable_array_noop())() AutoDtype Info: AutoDtypeEnum.Array_InOut
BindArrayCallable(callable_array_noop())() Execution Flag: True
BindTensorCallable(callable_tensor_noop())() AutoDtype Info: AutoDtypeEnum.Tensor_InOut
BindTensorCallable(callable_tensor_noop())() Execution Flag: True
...
LandmarksUnNormalize() AutoDtype Info: AutoDtypeEnum.Array_InOut
LandmarksUnNormalize() Execution Flag: True
```
* Execution Flag: True means current transform was executed successful, False means it was not executed because of the random probability or some Runtime Exceptions(torchlm will should the error infos if debug mode is True).
* AutoDtype Info: 
  * Array_InOut means current transform need a np.ndnarray as input and then output a np.ndarray.
  * Tensor_InOut means current transform need a torch Tensor as input and then output torch Tensor. 
  * Array_In means current transform needs a np.ndarray input and then output a torch Tensor. 
  * Tensor_In means current transform needs a torch Tensor input and then output a np.ndarray. 
    
  But, is ok if your pass a Tensor to a np.ndarray like transform, **torchlm** will automatically be compatible with different data types and then wrap back to the original type through a autodtype wrapper.


* Supported Transforms Sets, see [transforms.md](docs/api/transfroms.md). A detail example can be found at [test/transforms.py](test/transforms.py).

### Training(TODO)
* [ ] YOLOX
* [ ] YOLOv5
* [ ] NanoDet
* [ ] PIPNet
* [ ] ResNet
* [ ] MobileNet
* [ ] ShuffleNet
* [ ] ...

### Inference(TODO)
* [ ] ONNXRuntime
* [ ] MNN
* [ ] NCNN
* [ ] TNN
* [ ] ... 

## 📖 License.
The code of torchlm is released under the MIT License.

## 🎓 Acknowledgement  
The implementation of torchlm's transforms borrow the code from [Paperspace](ttps://github.com/Paperspace/DataAugmentationForObjectDetection/blob/master/data_aug/bbox_util.py).  
