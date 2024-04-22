"""
This is a customed version of the QRDetector class from the qrdet library.

Author: viaPhoton, Eric Canas
Github: https://github.com/viaphoton/qrdet
"""

from __future__ import annotations
import os
import numpy as np
import requests
import tqdm
import boto3
from urllib.parse import urlparse

from ultralytics import YOLO

from qrdet import _yolo_v8_results_to_dict, _prepare_input, BBOX_XYXY, CONFIDENCE

_WEIGHTS_FOLDER = os.path.join(os.path.dirname(__file__), '.model')
_CURRENT_RELEASE_TXT_FILE = os.path.join(_WEIGHTS_FOLDER, 'current_release.txt')
_MODEL_FILE_NAME = 'qrdet-viaphoton.onnx'


class QRDetector:
    def __init__(self, model_zoo: str = None, conf_th: float = 0.5, nms_iou: float = 0.3):
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
        
        self.model_zoo = model_zoo
        path = self.__download_weights_or_return_path(model_zoo=model_zoo)
        assert os.path.exists(path), f'Could not find model weights at {path}.'

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


    def __download_weights_or_return_path(self, desc: str = 'Downloading weights...', model_zoo: str = None) -> None:
        """
        Download the weights of the YoloV8 QR Segmentation model.
        :param model_size: str. The size of the model to download. Can be 's', 'm', or 'l'. Default: 's'.
        :param desc: str. The description of the download. Default: 'Downloading weights...'.
        :param model_zoo: str. The URL of the model repository. Defaults to None, which uses the pre-defined URL.
        """
        self.downloading_model = True
        model_zoo_url = self._generate_presigned_url(model_zoo)

        path = os.path.join(_WEIGHTS_FOLDER, _MODEL_FILE_NAME)
        if os.path.isfile(path):
            if os.path.isfile(_CURRENT_RELEASE_TXT_FILE):
                with open(_CURRENT_RELEASE_TXT_FILE, 'r') as file:
                    current_release = file.read()
                if current_release == model_zoo:
                    self.downloading_model = False
                    return path
        elif not os.path.exists(_WEIGHTS_FOLDER):
            os.makedirs(_WEIGHTS_FOLDER)

        response = requests.get(model_zoo_url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        with tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=desc) as progress_bar:
            with open(path, 'wb') as file:
                for data in response.iter_content(chunk_size=1024):
                    progress_bar.update(len(data))
                    file.write(data)
        with open(_CURRENT_RELEASE_TXT_FILE, 'w') as file:
            file.write(model_zoo)
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            os.remove(path)
            raise EOFError('Error, something went wrong while downloading the weights.')

        self.downloading_model = False
        return path
    
    def _generate_presigned_url(self, model_zoo):
        # Parse the provided URL
        parsed_url = urlparse(model_zoo)
        if parsed_url.netloc.endswith('amazonaws.com'):
            bucket_name = parsed_url.netloc.split('.')[0]
            object_name = parsed_url.path.lstrip('/')  # Remove the leading '/'
            
            # Create an S3 client
            s3_client = boto3.client('s3')
            try:
                # Generate a presigned URL for the object
                presigned_url = s3_client.generate_presigned_url('get_object',
                                                                Params={'Bucket': bucket_name, 'Key': object_name},
                                                                ExpiresIn=3600)
                return presigned_url
            except Exception as e:
                print(f"Error generating presigned URL: {e}")
        else:
            print("Invalid S3 URL format")