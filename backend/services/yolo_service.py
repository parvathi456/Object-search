
from ultralytics import YOLO


class YoloService:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        """
        Accepts an OpenCV BGR frame.
        Returns detected objects with coordinates, labels,
        and confidence scores.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().tolist()
                )

                detections.append({
                    "class_name": result.names[class_id],
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                })

        return detections
