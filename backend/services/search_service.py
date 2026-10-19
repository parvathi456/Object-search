
from pathlib import Path

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
)

from services.embedding_service import EmbeddingService
from services.yolo_service import YoloService


COLLECTION_NAME = "video_objects"
QDRANT_URL = "http://localhost:6333"
BASE_DIR = Path(__file__).resolve().parent.parent

# Load models once per application process.
yolo_service = YoloService()
embedding_service = EmbeddingService()
qdrant = QdrantClient(url=QDRANT_URL)


def search_objects(
    query_image: Image.Image,
    selected_video_ids: list[str],
    limit: int = 20,
) -> list[dict]:
    """
    Find visually similar objects, restricted to selected videos.
    """
    if not selected_video_ids:
        return []

    detections = yolo_service.detect(
        __import__("numpy").array(query_image.convert("RGB"))[:, :, ::-1]
    )

    if not detections:
        return []

    matches_by_id = {}

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]

        # The uploaded reference image is RGB. Crop in PIL coordinates.
        query_crop = query_image.convert("RGB").crop(
            (x1, y1, x2, y2)
        )

        vector = embedding_service.embed_image(query_crop)

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="video_id",
                    match=MatchAny(any=selected_video_ids),
                ),
                FieldCondition(
                    key="class_name",
                    match=MatchValue(
                        value=detection["class_name"]
                    ),
                ),
            ]
        )

        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        for point in response.points:
            payload = point.payload or {}

            result = {
                "detection_id": str(point.id),
                "video_id": payload.get("video_id"),
                "timestamp_seconds": payload.get(
                    "timestamp_seconds"
                ),
                "class_name": payload.get("class_name"),
                "detection_confidence": payload.get(
                    "confidence"
                ),
                "similarity_score": float(point.score),
                "bbox": payload.get("bbox"),
                "thumbnail_url": (
                    "/thumbnails/"
                    + str(payload.get("crop_path", ""))
                ),
                "query_object_class": detection["class_name"],
            }

            key = str(point.id)

            if (
                key not in matches_by_id
                or result["similarity_score"]
                > matches_by_id[key]["similarity_score"]
            ):
                matches_by_id[key] = result

    results = list(matches_by_id.values())
    results.sort(
        key=lambda item: item["similarity_score"],
        reverse=True,
    )

    return results[:limit]
