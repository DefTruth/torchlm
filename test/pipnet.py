import cv2
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet


def test_pipnet_runtime():
    device = "cpu"
    img_path = "./assets/pipnet0.jpg"
    save_path = "./logs/pipnet0.jpg"
    checkpoint = "./pretrained/pipnet/pipnet_resnet18_10x98x32x256_wflw.pth"
    image = cv2.imread(img_path)

    torchlm.runtime.bind(faceboxesv2())
    torchlm.runtime.bind(
        pipnet(
            backbone="resnet18",
            pretrained=True,
            num_nb=10,
            num_lms=98,
            net_stride=32,
            input_size=256,
            meanface_type="wflw",
            backbone_pretrained=True,
            map_location=device,
            checkpoint=checkpoint
        )
    )
    landmarks, bboxes = torchlm.runtime.forward(image)
    image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
    image = torchlm.utils.draw_landmarks(image, landmarks=landmarks)

    cv2.imwrite(save_path, image)


def test_pipnet_training():
    pass

def test_pipnet_evaluating():
    pass

def test_pipnet_export():
    pass


if __name__ == "__main__":
    test_pipnet_runtime()
    test_pipnet_training()
    test_pipnet_evaluating()
    test_pipnet_runtime()
