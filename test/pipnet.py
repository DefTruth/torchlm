import cv2
import torch
import torchlm


def test_pipnet():
    img_path = "./assets/pipnet0.jpg"
    save_path = "./logs/pipnet0.jpg"
    weight_file = "./pretrained/pipnet/weights/pipnet_resnet18_10x98x32x256_wflw.pth"
    detector = torchlm.tools.FaceBoxesV2()
    pipnet = torchlm.models.pipnet_resnet18_10x98x32x256_wflw(
        pretrained=False, backbone_pretrained=True)
    device = "cpu"
    state_dict = torch.load(weight_file, map_location=device)

    pipnet = pipnet.to(device)
    pipnet.load_state_dict(state_dict)

    my_thresh = 0.6
    det_box_scale = 1.2

    pipnet.eval()

    image = cv2.imread(img_path)
    image_height, image_width, _ = image.shape
    detections = detector.detect(image, my_thresh, 1)
    for i in range(detections.shape[0]):
        det_xmin = int(detections[i][0])
        det_ymin = int(detections[i][1])
        det_xmax = int(detections[i][2])
        det_ymax = int(detections[i][3])

        det_width = det_xmax - det_xmin + 1
        det_height = det_ymax - det_ymin + 1

        det_xmin -= int(det_width * (det_box_scale - 1) / 2)
        # remove a part of top area for alignment, see paper for details
        det_ymin += int(det_height * (det_box_scale - 1) / 2)
        det_xmax += int(det_width * (det_box_scale - 1) / 2)
        det_ymax += int(det_height * (det_box_scale - 1) / 2)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        det_xmax = min(det_xmax, image_width - 1)
        det_ymax = min(det_ymax, image_height - 1)
        cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
        det_crop_rgb = image[det_ymin:det_ymax, det_xmin:det_xmax, :][:, :, ::-1]  # RGB
        lms_pred_merge = pipnet.detect(img=det_crop_rgb)

        print(lms_pred_merge.shape)
        for j in range(lms_pred_merge.shape[0]):
            x_pred = lms_pred_merge[j, 0]
            y_pred = lms_pred_merge[j, 1]
            cv2.circle(image, (int(x_pred) + det_xmin, int(y_pred) + det_ymin),
                       1, (0, 0, 255), 2)
    cv2.imwrite(save_path, image)
    print(f"Detect done! Saved {save_path}")


if __name__ == "__main__":
    test_pipnet()
