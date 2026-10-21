
from pathlib import Path
from uuid import uuid4

import cv2
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from services.yolo_service import YoloService
from services.embedding_service import EmbeddingService


BASE_DIR = Path(__file__).resolve().parent.parent
THUMBNAIL_DIR = BASE_DIR / "thumbnails"

# COLLECTION_NAME = "video_objects"
COLLECTION_NAME = "video_objects_siglip"
QDRANT_URL = "http://localhost:6333"

# Load these once per worker process.
yolo_service = YoloService()
embedding_service = EmbeddingService()
# qdrant = QdrantClient(url=QDRANT_URL)
qdrant = QdrantClient(
    url="http://localhost:6333",
    timeout=120,
)


def ensure_collection():
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_service.embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    # Index payload fields used for filtering.
    for field_name in ("video_id", "class_name"):
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema="keyword",
        )


def delete_video_vectors(video_id: str):
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="video_id",
                    match=MatchValue(value=video_id),
                )
            ]
        ),
        wait=True,
    )


def index_video(
    video_id: str,
    video_path: str,
    sample_interval_seconds: float = 1.0,
) -> int:
    """
    Sample frames, detect objects, crop detections,
    create embeddings, and store vectors in Qdrant.

    Returns the number of indexed detections.
    """
    ensure_collection()

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError("Could not determine video FPS.")

    frame_step = max(
        1, round(fps * sample_interval_seconds)
    )

    video_crop_dir = THUMBNAIL_DIR / video_id
    video_crop_dir.mkdir(parents=True, exist_ok=True)

    # Re-indexing replaces this video's old vectors.
    delete_video_vectors(video_id)

    batch = []
    indexed_count = 0
    frame_number = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            current_frame = frame_number
            frame_number += 1

            if current_frame % frame_step != 0:
                continue

            timestamp = current_frame / fps
            height, width = frame.shape[:2]

            detections = yolo_service.detect(frame)

            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]

                x1 = max(0, min(x1, width))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(
                    crop, cv2.COLOR_BGR2RGB
                )
                pil_image = Image.fromarray(crop_rgb)

                crop_name = f"{uuid4().hex}.jpg"
                crop_path = video_crop_dir / crop_name
                pil_image.save(crop_path, quality=90)

                vector = embedding_service.embed_image(
                    pil_image
                )

                batch.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=vector,
                        payload={
                            "video_id": video_id,
                            "timestamp_seconds": round(
                                timestamp, 3
                            ),
                            "class_name": detection[
                                "class_name"
                            ],
                            "confidence": detection[
                                "confidence"
                            ],
                            "bbox": [
                                x1, y1, x2, y2
                            ],
                            "crop_path": (
                                f"{video_id}/{crop_name}"
                            ),
                        },
                    )
                )

                indexed_count += 1

                if len(batch) >= 64:
                    qdrant.upsert(
                        collection_name=COLLECTION_NAME,
                        points=batch,
                        wait=True,
                    )
                    batch.clear()

        if batch:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True,
            )

    finally:
        capture.release()

    return indexed_count
