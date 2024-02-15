import os

import insightface
import numpy as np
import onnxruntime
from torchvision import transforms

from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism


# Subclass of PrivacyMechanism that loads the insightface face detection model for use
class DetectFaceMechanism(PrivacyMechanism):
    def __init__(
        self, epsilon: float = 1, b: int = 1, det_size: tuple = (640, 640)
    ) -> None:
        super(DetectFaceMechanism, self).__init__()
        self.ToTensor = transforms.ToTensor()

        print("Loading face detection model.")
        onnxruntime.set_default_logger_severity(4)
        self.detect_model = insightface.model_zoo.get_model(
            os.path.expanduser("~//.insightface//models//buffalo_l//det_10g.onnx"),
            download=True,
        )
        self.det_size = det_size
        self.detect_model.prepare(
            ctx_id=0, det_size=self.det_size, input_size=self.det_size
        )

    def get_face_region(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # expects image to be in cv2 format
        bboxes, kpss = self.detect_model.detect(img)
        if len(bboxes) == 0:
            # if a face isn't detected, we pass the unaltered image through
            return img, np.array([0, 0, img.shape[1], img.shape[0]])
        bbox = bboxes[0]

        h0, w0, h1, w1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        crop_img = img[w0:w1, h0:h1, :]

        return crop_img, np.array([h0, w0, h1, w1])
