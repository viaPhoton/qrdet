"""
This is a customed version of the QRDetector class from the qrdet library.

Author: viaPhoton, Eric Canas
Github: https://github.com/viaphoton/qrdet
"""

from __future__ import annotations
import os
import numpy as np
from urllib.parse import urlparse

from ultralytics import YOLO

from qrdet import _yolo_v8_results_to_dict, _prepare_input, BBOX_XYXY, CONFIDENCE


class QRDetector:

    def __init__(self, model_type: str = 'onnx', conf_th: float = 0.5, nms_iou: float = 0.3):

        """
        Initialize the QRDetector.
        It loads the weights of the YOLOv8 model and prepares it for inference.
        :param model_size: str. The size of the model to use. It can be 'n' (nano), 's' (small), 'm' (medium) or
                                'l' (large). Larger models are more accurate but slower. Default (and recommended): 's'.
        :param conf_th: float. The confidence threshold to use for the detections. Detection with a confidence lower
                                than this value will be discarded. Default: 0.5.
        :param nms_iou: float. The IoU threshold to use for the Non-Maximum Suppression. Detections with an IoU higher
                                than this value will be discarded. Default: 0.3.
        """
        
        path = self.safe_model_path( model_type )

        self.model = YOLO(path, task="segment")  # Load the ONNX model using ONNX Runtime
        
        self._conf_th = conf_th
        self._nms_iou = nms_iou

    def detect(self, image: np.ndarray|'PIL.Image'|'torch.Tensor'|str, is_bgr: bool = False,
               **kwargs) -> tuple[dict[str, np.ndarray|float|tuple[float, float]]]:
        """
        Detect QR codes in the given image.

        :param image: str|np.ndarray|PIL.Image|torch.Tensor. Numpy array (H, W, 3), Tensor (1, 3, H, W), or
                                            path/url to the image to predict. 'screen' for grabbing a screenshot.
        :param legacy: bool. If sent as **kwarg**, will parse the output to make it identical to 1.x versions.
                            Not Recommended. Default: False.
        :return: tuple[dict[str, np.ndarray|float|tuple[float, float]]]. A tuple of dictionaries containing the
            following keys:
            - 'confidence': float. The confidence of the detection.
            - 'bbox_xyxy': np.ndarray. The bounding box of the detection in the format [x1, y1, x2, y2].
            - 'cxcy': tuple[float, float]. The center of the bounding box in the format (x, y).
            - 'wh': tuple[float, float]. The width and height of the bounding box in the format (w, h).
            - 'polygon_xy': np.ndarray. The accurate polygon that surrounds the QR code, with shape (N, 2).
            - 'quadrilateral_xy': np.ndarray. The quadrilateral that surrounds the QR code, with shape (4, 2).
            - 'expanded_quadrilateral_xy': np.ndarray. An expanded version of quadrilateral_xy, with shape (4, 2),
                that always include all the points within polygon_xy.

            All these keys (except 'confidence') have a 'n' (normalized) version. For example, 'bbox_xyxy' is the
            bounding box in absolute coordinates, while 'bbox_xyxyn' is the bounding box in normalized coordinates
            (from 0. to 1.).
        """
        image = _prepare_input(source=image, is_bgr=is_bgr)
        # Predict
        results = self.model.predict(source=image, conf=self._conf_th, iou=self._nms_iou, half=False,
                                device=None, max_det=100, augment=False, agnostic_nms=True,
                                classes=None, verbose=False)
        assert len(results) == 1, f'Expected 1 result if no batch sent, got {len(results)}'
        
        results = _yolo_v8_results_to_dict(results = results[0], image=image)

        return results


    def safe_model_path(self, model_type) -> str:
        """
        Return the path to the weights file.
        :return: str. The path to the weights file.
        """
        
        path = f'/app/tracker-daemon/models/viaphoton.{model_type}'
        if os.path.isfile(path):
            return path
        else:
            raise Exception(f'Error, file @ {path} not found.')
