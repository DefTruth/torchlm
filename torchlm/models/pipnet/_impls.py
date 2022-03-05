import os
import cv2
import torch
import warnings
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch import optim
from torch.utils.data import DataLoader
from typing import Tuple, Union, Optional, Any, List

from ..core import ABCBaseModel
from ..utils import metrics, transforms
from ._cfgs import _DEFAULT_MEANFACE_STRINGS
from ._utils import _get_meanface, _normalize
from ._data import _PIPTrainDataset, _PIPEvalDataset

_PIPNet_Output_Type = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
_PIPNet_Loss_Output_Type = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


class _PIPNetImpl(ABCBaseModel):

    def __init__(
            self,
            num_nb: int = 10,
            num_lms: int = 68,
            input_size: int = 256,
            net_stride: int = 32,
            meanface_type: Optional[str] = None
    ):
        super(_PIPNetImpl, self).__init__()
        assert net_stride in (32, 64, 128)
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        # setup default meanface
        self.meanface_status = False
        self.meanface_type = meanface_type
        self.meanface_indices: List[int] = []
        self.reverse_index1: List[int] = []
        self.reverse_index2: List[int] = []
        self.max_len: int = -1
        self._set_default_meanface()

    def set_custom_meanface(
            self,
            custom_meanface_file_or_string: str
    ) -> bool:
        """
        :param custom_meanface_file_or_string: a long string or a file contains normalized
        or un-normalized meanface coords, the format is "x0,y0,x1,y1,x2,y2,...,xn-1,yn-1".
        :return: status, True if successful.
        """
        try:
            custom_meanface_type = "custom"
            if os.path.isfile(custom_meanface_file_or_string):
                with open(custom_meanface_file_or_string) as f:
                    custom_meanface_string = f.readlines()[0]
            else:
                custom_meanface_string = custom_meanface_file_or_string

            custom_meanface_indices, custom_reverse_index1, \
            custom_reverse_index2, custom_max_len, custom_meanface_lms = _get_meanface(
                meanface_string=custom_meanface_string, num_nb=self.num_nb)

            # check landmarks number
            if custom_meanface_lms != self.num_lms:
                warnings.warn(
                    f"custom_meanface_lms != self.num_lms, "
                    f"{custom_meanface_lms} != {self.num_lms}"
                    f"So, we will skip this setup for PIPNet meanface."
                    f"Please check and setup meanface carefully before"
                    f"running PIPNet ..."
                )
                self.meanface_status = False
            else:
                # replace if successful
                self.meanface_type = custom_meanface_type
                self.meanface_indices = custom_meanface_indices
                self.reverse_index1 = custom_reverse_index1
                self.reverse_index2 = custom_reverse_index2
                self.max_len = custom_max_len
                self.meanface_status = True
        except:
            self.meanface_status = False

        return self.meanface_status

    def _set_default_meanface(self):
        if self.meanface_type is not None:
            if self.meanface_type.upper() not in _DEFAULT_MEANFACE_STRINGS:
                warnings.warn(
                    f"Can not found default dataset: {self.meanface_type.upper()}!"
                    f"So, we will skip this setup for PIPNet meanface."
                    f"Please check and setup meanface carefully before"
                    f"running PIPNet ..."
                )
                self.meanface_status = False
            else:
                meanface_string = _DEFAULT_MEANFACE_STRINGS[self.meanface_type.upper()]
                meanface_indices, reverse_index1, reverse_index2, max_len, meanface_lms = \
                    _get_meanface(meanface_string=meanface_string, num_nb=self.num_nb)
                # check landmarks number
                if meanface_lms != self.num_lms:
                    warnings.warn(
                        f"meanface_lms != self.num_lms, {meanface_lms} != {self.num_lms}"
                        f"So, we will skip this setup for PIPNet meanface."
                        f"Please check and setup meanface carefully before"
                        f"running PIPNet ..."
                    )
                    self.meanface_status = False
                else:
                    self.meanface_indices = meanface_indices
                    self.reverse_index1 = reverse_index1
                    self.reverse_index2 = reverse_index2
                    self.max_len = max_len

                    self.meanface_status = True

    def loss(self, *args, **kwargs) -> _PIPNet_Loss_Output_Type:
        return _loss_impl(*args, **kwargs)

    def detect(self, img: np.ndarray) -> np.ndarray:
        return _detect_impl(net=self, img=img)

    def training(
            self,
            annotation_path: str,
            criterion_cls: nn.Module = nn.MSELoss(),
            criterion_reg: nn.Module = nn.L1Loss(),
            learning_rate: float = 0.0001,
            cls_loss_weight: float = 10.,
            reg_loss_weight: float = 1.,
            num_nb: int = 10,
            num_epochs: int = 60,
            save_dir: Optional[str] = "./save",
            save_interval: Optional[int] = 10,
            save_prefix: Optional[str] = "",
            decay_steps: Optional[List[int]] = (30, 50),
            decay_gamma: Optional[float] = 0.1,
            device: Optional[Union[str, torch.device]] = "cuda",
            transform: Optional[transforms.LandmarksCompose] = None,
            norm_resize_transform: Optional[transforms.LandmarksCompose] = None,
            **kwargs: Any  # params for DataLoader
    ) -> nn.Module:
        device = device if torch.cuda.is_available() else "cpu"
        # prepare dataset
        default_dataset = _PIPTrainDataset(
            annotation_path=annotation_path,
            input_size=self.input_size,
            transform=transform
        )
        train_loader = DataLoader(default_dataset, **kwargs)

        return _training_impl(
            net=self,
            train_loader=train_loader,
            criterion_cls=criterion_cls,
            criterion_reg=criterion_reg,
            learning_rate=learning_rate,
            cls_loss_weight=cls_loss_weight,
            reg_loss_weight=reg_loss_weight,
            num_nb=num_nb,
            num_epochs=num_epochs,
            save_dir=save_dir,
            save_prefix=save_prefix,
            save_interval=save_interval,
            decay_steps=decay_steps,
            decay_gamma=decay_gamma,
            device=device
        )

    def evaluating(
            self,
            annotation_path: str,
            norm_indices: List[int] = (60, 72),
            dataset_type: Optional[str] = None
    ) -> Tuple[float, float, float]:  # NME, FR, AUC
        # prepare dataset
        eval_dataset = _PIPEvalDataset(annotation_path=annotation_path)

        return _evaluating_impl(
            net=self,
            eval_dataset=eval_dataset,
            norm_indices=norm_indices,
            dataset_type=dataset_type
        )

    def export(
            self,
            onnx_path: str = "./onnx/pipnet.onnx",
            opset: int = 12,
            simplify: bool = False,
            output_names: Optional[List[str]] = None
    ) -> None:
        _export_impl(
            net=self,
            onnx_path=onnx_path,
            opset=opset,
            simplify=simplify,
            output_names=output_names
        )

    def forward(self, *args, **kwargs) -> _PIPNet_Output_Type:
        raise NotImplementedError


@torch.no_grad()
def _detect_impl(
        net: _PIPNetImpl,
        img: np.ndarray
) -> np.ndarray:
    """
    :param img: source face image without background, RGB with HWC and range [0,255]
    :return: detected landmarks coordinates, shape [num, 2]
    """
    if not net.meanface_status:
        raise RuntimeError(
            f"Can not found any meanface landmarks settings !"
            f"Please check and setup meanface carefully before"
            f"running PIPNet ..."
        )

    net.eval()

    height, width, _ = img.shape
    img: np.ndarray = cv2.resize(img, (net.input_size, net.input_size))  # 256, 256
    img: Tensor = torch.from_numpy(_normalize(img=img)).unsqueeze(0)  # (1,3,256,256)
    outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net.forward(img)
    # (1,68,8,8)
    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
    assert tmp_batch == 1

    outputs_cls = outputs_cls.view(tmp_batch * tmp_channel, -1)  # (68.64)
    max_ids = torch.argmax(outputs_cls, 1)  # (68,)
    max_ids = max_ids.view(-1, 1)  # (68,1)
    max_ids_nb = max_ids.repeat(1, net.num_nb).view(-1, 1)  # (68,10) -> (68*10,1)

    outputs_x = outputs_x.view(tmp_batch * tmp_channel, -1)  # (68,64)
    outputs_x_select = torch.gather(outputs_x, 1, max_ids)  # (68,1)
    outputs_x_select = outputs_x_select.squeeze(1)  # (68,)
    outputs_y = outputs_y.view(tmp_batch * tmp_channel, -1)
    outputs_y_select = torch.gather(outputs_y, 1, max_ids)
    outputs_y_select = outputs_y_select.squeeze(1)  # (68,)

    outputs_nb_x = outputs_nb_x.view(tmp_batch * net.num_nb * tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)  # (68*10,1)
    outputs_nb_x_select = outputs_nb_x_select.squeeze(1).view(-1, net.num_nb)  # (68,10)
    outputs_nb_y = outputs_nb_y.view(tmp_batch * net.num_nb * tmp_channel, -1)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
    outputs_nb_y_select = outputs_nb_y_select.squeeze(1).view(-1, net.num_nb)  # (68,10)

    # tmp_width=tmp_height=8 max_ids->[0,63] calculate grid center (cx,cy) in 8x8 map
    lms_pred_x = (max_ids % tmp_width).view(-1, 1).float() + outputs_x_select.view(-1, 1)  # x=cx+offset_x
    lms_pred_y = (max_ids // tmp_width).view(-1, 1).float() + outputs_y_select.view(-1, 1)  # y=cy+offset_y
    lms_pred_x /= 1.0 * net.input_size / net.net_stride  # normalize coord (x*32)/256
    lms_pred_y /= 1.0 * net.input_size / net.net_stride  # normalize coord (y*32)/256

    lms_pred_nb_x = (max_ids % tmp_width).view(-1, 1).float() + outputs_nb_x_select  # (68,10)
    lms_pred_nb_y = (max_ids // tmp_width).view(-1, 1).float() + outputs_nb_y_select  # (68,10)
    lms_pred_nb_x = lms_pred_nb_x.view(-1, net.num_nb)  # (68,10)
    lms_pred_nb_y = lms_pred_nb_y.view(-1, net.num_nb)  # (68,10)
    lms_pred_nb_x /= 1.0 * net.input_size / net.net_stride  # normalize coord (nx*32)/256
    lms_pred_nb_y /= 1.0 * net.input_size / net.net_stride  # normalize coord (ny*32)/256

    # merge predictions
    tmp_nb_x = lms_pred_nb_x[net.reverse_index1, net.reverse_index2].view(net.num_lms, net.max_len)
    tmp_nb_y = lms_pred_nb_y[net.reverse_index1, net.reverse_index2].view(net.num_lms, net.max_len)
    tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
    tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
    lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1)  # (68,2)
    lms_pred_merge = lms_pred_merge.cpu().numpy()  # (68,2)

    lms_pred_merge[:, 0] *= float(width)
    lms_pred_merge[:, 1] *= float(height)

    return lms_pred_merge


def _loss_impl(
        outputs_cls: Tensor,
        outputs_x: Tensor,
        outputs_y: Tensor,
        outputs_nb_x: Tensor,
        outputs_nb_y: Tensor,
        labels_cls: Tensor,
        labels_x: Tensor,
        labels_y: Tensor,
        labels_nb_x: Tensor,
        labels_nb_y: Tensor,
        criterion_cls: nn.Module,
        criterion_reg: nn.Module,
        num_nb: int = 10
) -> _PIPNet_Loss_Output_Type:
    """
    :param outputs_cls: output heatmap Tensor e.g (b,68,8,8)
    :param outputs_x: output x offsets Tensor e.g (b,68,8,8)
    :param outputs_y: output y offsets Tensor e.g (b,68,8,8)
    :param outputs_nb_x: output neighbor's x offsets Tensor e.g (b,68*10,8,8)
    :param outputs_nb_y: output neighbor's y offsets Tensor e.g (b,68*10,8,8)
    :param labels_cls: output heatmap Tensor e.g (b,68,8,8)
    :param labels_x: output x offsets Tensor e.g (b,68,8,8)
    :param labels_y: output y offsets Tensor e.g (b,68,8,8)
    :param labels_nb_x: output neighbor's x offsets Tensor e.g (b,68*10,8,8)
    :param labels_nb_y: output neighbor's y offsets Tensor e.g (b,68*10,8,8)
    :param criterion_cls: loss criterion for heatmap classification, e.g MSELoss
    :param criterion_reg: loss criterion for offsets regression, e.g L1Loss
    :param num_nb: the number of Nearest-neighbor landmarks for NRM
    :return: losses Tensor values without weighted.
    """

    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
    labels_cls = labels_cls.view(tmp_batch * tmp_channel, -1)
    labels_max_ids = torch.argmax(labels_cls, 1)
    labels_max_ids = labels_max_ids.view(-1, 1)
    labels_max_ids_nb = labels_max_ids.repeat(1, num_nb).view(-1, 1)

    outputs_x = outputs_x.view(tmp_batch * tmp_channel, -1)
    outputs_x_select = torch.gather(outputs_x, 1, labels_max_ids)
    outputs_y = outputs_y.view(tmp_batch * tmp_channel, -1)
    outputs_y_select = torch.gather(outputs_y, 1, labels_max_ids)
    outputs_nb_x = outputs_nb_x.view(tmp_batch * num_nb * tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, labels_max_ids_nb)
    outputs_nb_y = outputs_nb_y.view(tmp_batch * num_nb * tmp_channel, -1)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, labels_max_ids_nb)

    labels_x = labels_x.view(tmp_batch * tmp_channel, -1)
    labels_x_select = torch.gather(labels_x, 1, labels_max_ids)
    labels_y = labels_y.view(tmp_batch * tmp_channel, -1)
    labels_y_select = torch.gather(labels_y, 1, labels_max_ids)
    labels_nb_x = labels_nb_x.view(tmp_batch * num_nb * tmp_channel, -1)
    labels_nb_x_select = torch.gather(labels_nb_x, 1, labels_max_ids_nb)
    labels_nb_y = labels_nb_y.view(tmp_batch * num_nb * tmp_channel, -1)
    labels_nb_y_select = torch.gather(labels_nb_y, 1, labels_max_ids_nb)

    labels_cls = labels_cls.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
    loss_cls = criterion_cls(outputs_cls, labels_cls)
    loss_x = criterion_reg(outputs_x_select, labels_x_select)
    loss_y = criterion_reg(outputs_y_select, labels_y_select)
    loss_nb_x = criterion_reg(outputs_nb_x_select, labels_nb_x_select)
    loss_nb_y = criterion_reg(outputs_nb_y_select, labels_nb_y_select)

    return loss_cls, loss_x, loss_y, loss_nb_x, loss_nb_y


def _training_impl(
        net: _PIPNetImpl,
        train_loader: DataLoader,
        criterion_cls: nn.Module = nn.MSELoss(),
        criterion_reg: nn.Module = nn.L1Loss(),
        learning_rate: float = 0.0001,
        cls_loss_weight: float = 10.,
        reg_loss_weight: float = 1.,
        num_nb: int = 10,
        num_epochs: int = 60,
        save_dir: Optional[str] = "./save",
        save_prefix: Optional[str] = "",
        save_interval: Optional[int] = 10,
        decay_steps: Optional[List[int]] = (30, 50),
        decay_gamma: Optional[float] = 0.1,
        device: Optional[Union[str, torch.device]] = "cuda"
):
    import logging

    if not net.meanface_status:
        raise RuntimeError(
            f"Can not found any meanface landmarks settings !"
            f"Please check and setup meanface carefully before"
            f"running PIPNet ..."
        )

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=decay_steps,
        gamma=decay_gamma
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logging.info('-' * 10)

        net.train()
        epoch_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels_cls, labels_x, labels_y, labels_nb_x, labels_nb_y = data
            inputs = inputs.to(device)
            labels_cls = labels_cls.to(device)
            labels_x = labels_x.to(device)
            labels_y = labels_y.to(device)
            labels_nb_x = labels_nb_x.to(device)
            labels_nb_y = labels_nb_y.to(device)
            outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
            loss_cls, loss_x, loss_y, loss_nb_x, loss_nb_y = net.loss(
                outputs_cls=outputs_cls,
                outputs_x=outputs_x,
                outputs_y=outputs_y,
                outputs_nb_x=outputs_nb_x,
                outputs_nb_y=outputs_nb_y,
                labels_cls=labels_cls,
                labels_x=labels_x,
                labels_y=labels_y,
                labels_nb_x=labels_nb_x,
                labels_nb_y=labels_nb_y,
                criterion_cls=criterion_cls,
                criterion_reg=criterion_reg,
                num_nb=num_nb
            )
            loss = cls_loss_weight * loss_cls + reg_loss_weight * loss_x \
                   + reg_loss_weight * loss_y + reg_loss_weight * loss_nb_x \
                   + reg_loss_weight * loss_nb_y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(
                    '[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <cls loss: {:.6f}> '
                    '<x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'
                        .format(epoch, num_epochs - 1, i, len(train_loader) - 1, loss.item(),
                                cls_loss_weight * loss_cls.item(), reg_loss_weight * loss_x.item(),
                                reg_loss_weight * loss_y.item(), reg_loss_weight * loss_nb_x.item(),
                                reg_loss_weight * loss_nb_y.item())
                )
                logging.info(
                    '[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <cls loss: {:.6f}> '
                    '<x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'
                        .format(epoch, num_epochs - 1, i, len(train_loader) - 1, loss.item(),
                                cls_loss_weight * loss_cls.item(), reg_loss_weight * loss_x.item(),
                                reg_loss_weight * loss_y.item(), reg_loss_weight * loss_nb_x.item(),
                                reg_loss_weight * loss_nb_y.item())
                )
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        if epoch % (save_interval - 1) == 0 and epoch > 0:
            epoch_loss = np.round(epoch_loss, 4)
            filename = os.path.join(save_dir, f'{save_prefix}-epoch{epoch}-loss{epoch_loss}.pth')
            torch.save(net.state_dict(), filename)
            print(filename, ' saved')
        # adjust lr
        scheduler.step()

    return net


def _evaluating_impl(
        net: _PIPNetImpl,
        eval_dataset: _PIPEvalDataset,
        norm_indices: List[int] = (60, 72),
        dataset_type: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    :param net: PIPNet instance
    :param eval_dataset: _PIPEvalDataset instance
    :param norm_indices: the indexes of two eyeballs.
    :param dataset_type: optional, specific dataset, e.g WFLW/COFW/300W
    :return: NME, FR, AUC
    """
    import tqdm
    norm_indices = list(norm_indices)
    if dataset_type is not None:
        if dataset_type.upper() in (
                'DATA_300W', 'DATA_300W_COFW_WFLW',
                'DATA_300W_CELEBA', '300W'
        ):
            norm_indices = [36, 45]
        elif dataset_type.upper() == 'COFW':
            norm_indices = [8, 9]
        elif dataset_type.upper() == 'WFLW':
            norm_indices = [60, 72]
        elif dataset_type.upper() == 'AFLW':
            norm_indices = None

    nmes = []
    # evaluating
    for img, lms_gt in tqdm.tqdm(eval_dataset, colour="green"):
        lms_pred = net.detect(img=img)  # (n,2)
        if norm_indices is not None:
            norm = np.linalg.norm(lms_gt[norm_indices[0]] - lms_gt[norm_indices[1]])
        else:
            norm = 1.
        nmes.append(metrics.nme(lms_pred=lms_pred, lms_gt=lms_gt, norm=norm))

    nme = np.mean(nmes).item()
    fr, auc = metrics.fr_and_auc(nmes=nmes)

    return nme, fr, auc


def _export_impl(
        net: _PIPNetImpl,
        onnx_path: str = "./onnx/pipnet.onnx",
        opset: int = 12,
        simplify: bool = False,
        output_names: Optional[List[str]] = None
):
    import onnx

    save_dir = os.path.dirname(onnx_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if output_names is None:
        output_names = ["outputs_cls", "outputs_x", "outputs_y",
                        "outputs_nb_x", "outputs_nb_y"]

    x = torch.randn((1, 3, net.input_size, net.input_size)).float()
    torch.onnx.export(
        net, x,
        onnx_path,
        verbose=False,
        opset_version=opset,
        input_names=['img'],
        output_names=output_names
    )
    # Checks
    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    print(onnx.helper.printable_graph(model_onnx.graph))  # print

    if simplify:
        try:
            import onnxsim
            model_onnx, check = onnxsim.simplify(
                model_onnx, check_n=3)
            assert check, 'assert check failed'
            onnx.save(model_onnx, onnx_path)

        except Exception as e:
            print(f"{onnx_path}:+ simplifier failure: {e}")
