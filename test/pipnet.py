import cv2
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from torchlm.runtime import pipnet_ort


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
    model = pipnet(
        backbone="resnet18",
        pretrained=True,
        num_nb=10,
        num_lms=98,
        net_stride=32,
        input_size=256,
        meanface_type="wflw",
        backbone_pretrained=True,
        map_location="cpu",
        checkpoint=None
    )
    # # generate your custom meanface
    # custom_meanface, custom_meanface_string = \
    #     torchlm.data.annotools.generate_meanface(
    #         annotation_path="../data/WFLW/convertd/train.txt"
    #     )
    # # setting up your custom meanface
    # model.set_custom_meanface(
    #     custom_meanface_file_or_string=custom_meanface_string
    # )

    model.apply_freezing(backbone=True)

    model.apply_training(
        annotation_path="../data/WFLW/convertd/train.txt",
        num_epochs=10,
        learning_rate=0.0001,
        save_dir="./save/pipnet",
        save_prefix="pipnet-wflw-resnet18",
        save_interval=1,
        device="cpu",
        coordinates_already_normalized=True,
        batch_size=16,
        num_workers=4,
        shuffle=True
    )


def test_pipnet_evaluating():
    model = pipnet(
        backbone="resnet18",
        pretrained=True,
        num_nb=10,
        num_lms=98,
        net_stride=32,
        input_size=256,
        meanface_type="wflw",
        backbone_pretrained=True,
        map_location="cpu",
        checkpoint=None
    )
    model.apply_freezing(backbone=True, heads=True, extra=True)

    NME, FR, AUC = model.apply_evaluating(
        annotation_path="../data/WFLW/convertd/test.txt",
        norm_indices=[60, 72],
        coordinates_already_normalized=True,
        eval_normalized_coordinates=False
    )
    print(f"NME: {NME}, FR: {FR}, AUC: {AUC}")


def test_pipnet_exporting():
    model = pipnet(
        backbone="resnet18",
        pretrained=True,
        num_nb=10,
        num_lms=98,
        net_stride=32,
        input_size=256,
        meanface_type="wflw",
        backbone_pretrained=True,
        map_location="cpu",
        checkpoint=None
    )
    model.apply_exporting(
        onnx_path="./save/pipnet/pipnet_resnet18.onnx",
        opset=12, simplify=True, output_names=None
    )

    model_f = faceboxesv2()
    model_f.apply_exporting(
        onnx_path="./save/faceboxesv2/faceboxesv2-640x640.onnx",
        opset=12, simplify=True, output_names=None,
        input_size=640
    )


def test_pipnet_meanface():
    # generate your custom meanface
    custom_meanface, custom_meanface_string = \
        torchlm.data.annotools.generate_meanface(
            annotation_path="../data/WFLW/convertd/train.txt",
            coordinates_already_normalized=True
        )
    canvas = torchlm.data.annotools.draw_meanface(
        meanface=custom_meanface,
        coordinates_already_normalized=True
    )
    cv2.imwrite("./logs/wflw_meanface.jpg", canvas)


def test_pipnet_runtime_ort():
    img_path = "./assets/pipnet0.jpg"
    save_path = "./logs/pipnet0_ort.jpg"
    image = cv2.imread(img_path)

    torchlm.runtime.bind(faceboxesv2())
    torchlm.runtime.bind(
        pipnet_ort(
            onnx_path="./save/pipnet/pipnet_resnet18.onnx",
            num_nb=10,
            num_lms=98,
            net_stride=32,
            input_size=256,
            meanface_type="wflw"
        )
    )
    landmarks, bboxes = torchlm.runtime.forward(image)
    image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
    image = torchlm.utils.draw_landmarks(image, landmarks=landmarks)

    cv2.imwrite(save_path, image)
    print(f"Saved {save_path} !")


if __name__ == "__main__":
    test_pipnet_runtime()
    test_pipnet_training()
    test_pipnet_evaluating()
    test_pipnet_exporting()
    test_pipnet_meanface()
    test_pipnet_runtime_ort()
