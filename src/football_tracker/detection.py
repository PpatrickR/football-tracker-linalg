"""Person detection via RT-DETR (Apache 2.0).

Single-class only: 'person'. Position roles (QB, WR, etc.) are derived from
field-coord alignment downstream, not from visual classification, so this
module never needs to know about football specifically.
"""
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


@dataclass
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    score: float

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center of the box: where the player meets the field."""
        x1, _, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, y2)


class PersonDetector:
    def __init__(
        self,
        model_id: str = "PekingU/rtdetr_r50vd_coco_o365",
        device: str | None = None,
        score_threshold: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = RTDetrImageProcessor.from_pretrained(model_id)
        self.model = (
            RTDetrForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        )
        self.score_threshold = score_threshold
        self._person_label_ids = {
            i
            for i, name in self.model.config.id2label.items()
            if name.lower() == "person"
        }
        if not self._person_label_ids:
            raise RuntimeError(
                f"Model {model_id} has no 'person' class in id2label"
            )

    @torch.inference_mode()
    def detect(self, image: Image.Image | np.ndarray) -> list[Detection]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.score_threshold
        )[0]
        return [
            Detection(bbox_xyxy=tuple(box.cpu().tolist()), score=float(score))
            for box, score, label in zip(
                results["boxes"], results["scores"], results["labels"]
            )
            if int(label) in self._person_label_ids
        ]
