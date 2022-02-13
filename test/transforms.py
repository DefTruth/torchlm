import cv2
import numpy as np
import torchvision
import albumentations
import matplotlib.pyplot as plt

import torchlm

def test_torchlm_transforms():
    img_path = "./1.jpg"
    anno_path = "./1.txt"
    img = cv2.imread(img_path)[:, :, ::-1].copy()
    print(img.shape)
    with open(anno_path, 'r') as fr:
        lm_info = fr.readlines()[0].strip('\n').split(' ')[1:]

    landmarks = [float(x) for x in lm_info[4:]]
    landmarks = np.array(landmarks).reshape(5, 2)  # (5,2)
    print(landmarks.shape)

    # some global setting
    torchlm.set_transforms_debug(True)
    torchlm.set_transforms_logging(True)
    torchlm.set_autodtype_logging(True)

    transform = torchlm.LandmarksCompose([
        # native torchlm transform
        torchlm.LandmarksRandomHorizontalFlip(prob=0.5),
        torchlm.LandmarksRandomScale(prob=0.5),
        torchlm.LandmarksRandomTranslate(prob=0.5),
        torchlm.LandmarksRandomShear(prob=0.5),
        torchlm.LandmarksRandomMask(prob=0.5),
        torchlm.LandmarksRandomBlur(kernel_range=(5, 25),prob=0.5),
        torchlm.LandmarksRandomBrightness(prob=0.5),
        torchlm.LandmarksRandomRotate(40, prob=0.5, bins=8),
        torchlm.LandmarksRandomCenterCrop((0.5, 1.0), (0.5, 1.0), prob=0.5),
        # bind torchvision transform
        # bind albumentations transform
        torchlm.LandmarksResize((256, 256)),
        torchlm.LandmarksNormalize(),
        torchlm.LandmarksToTensor(),
        torchlm.LandmarksToNumpy(),
        torchlm.LandmarksUnNormalize()
    ])

    trans_img, trans_landmarks = transform(img, landmarks)
    print(trans_img.shape)
    new_img = torchlm.draw_landmarks(trans_img, trans_landmarks)
    plt.imshow(new_img)
    plt.show()
    print("landmarks after transform: ", trans_landmarks.flatten().tolist())


if __name__ == "__main__":
    test_torchlm_transforms()








