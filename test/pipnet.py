import cv2
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet


def test_pipnet():
    device = "cpu"
    img_path = "./assets/pipnet0.jpg"
    save_path = "./logs/pipnet0.jpg"
    checkpoint = "./pretrained/pipnet/weights/pipnet_resnet18_10x98x32x256_wflw.pth"
    image = cv2.imread(img_path)

    torchlm.runtime.set_faces(faceboxesv2())
    torchlm.runtime.set_landmarks(
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

    landmarks, detections = torchlm.runtime.forward(image)

    for i in range(detections.shape[0]):
        x1 = int(detections[i][0])
        y1 = int(detections[i][1])
        x2 = int(detections[i][2])
        y2 = int(detections[i][3])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for j in range(landmarks[i].shape[0]):
            lx = int(landmarks[i, j, 0])
            ly = int(landmarks[i, j, 1])
            cv2.circle(image, (lx, ly), 1, (0, 255, 0), 2)

    cv2.imwrite(save_path, image)
    print(f"Detect done! Saved {save_path}")


if __name__ == "__main__":
    test_pipnet()
