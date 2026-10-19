
from pathlib import Path, PurePosixPath
import logging
import sqlite3

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.indexing_service import index_video

import json
from io import BytesIO

from PIL import Image, UnidentifiedImageError
from fastapi import Form

from services.search_service import search_objects
from fastapi.responses import FileResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"

for directory in (DATA_DIR, UPLOAD_DIR, THUMBNAIL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "search.db"
MAX_VIDEO_SIZE = 1024 * 1024 * 1024

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

app = FastAPI(title="Video Object Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/thumbnails",
    StaticFiles(directory=THUMBNAIL_DIR),
    name="thumbnails",
)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'uploaded',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


@app.on_event("startup")
def startup():
    initialize_database()


def set_video_status(video_id: str, status: str):
    with get_connection() as conn:
        conn.execute(
            "UPDATE videos SET status = ? WHERE id = ?",
            (status, video_id),
        )


def run_indexing(video_id: str, video_path: str):
    try:
        count = index_video(video_id, video_path)
        set_video_status(video_id, "indexed")
        logger.info(
            "Indexed video %s: %s detections",
            video_id,
            count,
        )
    except Exception:
        logger.exception("Indexing failed for %s", video_id)
        set_video_status(video_id, "failed")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "video-object-search",
    }


@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)):
    filename = PurePosixPath(
        (file.filename or "").replace("\\", "/")
    ).name
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Supported formats: MP4, AVI, MOV, MKV.",
        )

    video_id = __import__("uuid").uuid4().hex
    destination = UPLOAD_DIR / f"{video_id}{suffix}"
    total_bytes = 0

    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)

                if total_bytes > MAX_VIDEO_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Maximum video size is 1 GB.",
                    )

                output.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO videos
                    (id, filename, storage_path, status)
                VALUES (?, ?, ?, 'uploaded')
                """,
                (video_id, filename, str(destination)),
            )

        return {
            "id": video_id,
            "filename": filename,
            "status": "uploaded",
        }

    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.get("/api/videos")
def list_videos():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, filename, status, created_at
            FROM videos
            ORDER BY created_at DESC
        """).fetchall()

    return {"videos": [dict(row) for row in rows]}


@app.post("/api/videos/{video_id}/index")
def start_indexing(
    video_id: str,
    background_tasks: BackgroundTasks,
):
    with get_connection() as conn:
        video = conn.execute(
            "SELECT * FROM videos WHERE id = ?",
            (video_id,),
        ).fetchone()

    if video is None:
        raise HTTPException(
            status_code=404,
            detail="Video not found.",
        )

    if video["status"] == "processing":
        raise HTTPException(
            status_code=409,
            detail="Video is already being indexed.",
        )

    video_path = Path(video["storage_path"])

    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Stored video file is missing.",
        )

    set_video_status(video_id, "processing")
    background_tasks.add_task(
        run_indexing,
        video_id,
        str(video_path),
    )

    return {
        "video_id": video_id,
        "status": "processing",
        "message": "Video indexing started.",
    }



@app.post("/api/search")
async def search_videos(
    file: UploadFile = File(...),
    selected_video_ids: str = Form(...),
):
    # The frontend sends a JSON array in this form field.
    # try:
    #     print("RAW selected_video_ids:", repr(selected_video_ids))
    #     video_ids = json.loads(selected_video_ids)
    # except json.JSONDecodeError:
    #     raise HTTPException(
    #         status_code=400,
    #         detail="selected_video_ids must be a JSON array.",
    #     )

    # if (
    #     not isinstance(video_ids, list)
    #     or not video_ids
    #     or not all(
    #         isinstance(item, str) and item.strip()
    #         for item in video_ids
    #     )
    # ):
    #     raise HTTPException(
    #         status_code=400,
    #         detail="Select at least one video.",
    #     )
    
    try:
        print("RAW selected_video_ids:", repr(selected_video_ids))
        video_ids = json.loads(selected_video_ids)
    except json.JSONDecodeError:
        video_ids = selected_video_ids.strip()

    # Accept a single bare ID as well as a JSON array.
    if isinstance(video_ids, str):
        video_ids = [video_ids] if video_ids else []

    if (
        not isinstance(video_ids, list)
        or not video_ids
        or not all(
            isinstance(item, str) and item.strip()
            for item in video_ids
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Select at least one video.",
        )

    video_ids = [item.strip() for item in video_ids]
    video_ids = list(dict.fromkeys(video_ids))

    # video_ids = list(dict.fromkeys(video_ids))

    # Confirm every selected video exists and is indexed.
    placeholders = ",".join("?" for _ in video_ids)

    with get_connection() as conn:
        print("DATABASE:", conn.execute("PRAGMA database_list").fetchall())

        rows = conn.execute(
            f"""
            SELECT id, filename, status
            FROM videos
            WHERE id IN ({placeholders})
            """,
            video_ids,
        ).fetchall()

        print("SEARCH VIDEO IDS:", video_ids)
        print("DATABASE MATCHES:", [dict(row) for row in rows])
        
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, filename, status
            FROM videos
            WHERE id IN ({placeholders})
            """,
            video_ids,
        ).fetchall()

    found = {row["id"]: row for row in rows}

    if len(found) != len(video_ids):
        raise HTTPException(
            status_code=404,
            detail="One or more selected videos were not found.",
        )

    not_ready = [
        row["id"]
        for row in rows
        if row["status"] != "indexed"
    ]

    if not_ready:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "All selected videos must be indexed.",
                "videos_not_ready": not_ready,
            },
        )

    # Limit reference images to 10 MB.
    image_bytes = await file.read(10 * 1024 * 1024 + 1)
    await file.close()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Reference image must be 10 MB or smaller.",
        )

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        raise HTTPException(
            status_code=415,
            detail="Please upload a valid image.",
        )

    if image.width * image.height > 20_000_000:
        raise HTTPException(
            status_code=413,
            detail="Reference image resolution is too large.",
        )

    results = search_objects(
        query_image=image,
        selected_video_ids=video_ids,
        limit=20,
    )

    # Attach filenames so the frontend can label each result.
    filenames = {
        row["id"]: row["filename"]
        for row in rows
    }

    for result in results:
        result["video_filename"] = filenames.get(
            result["video_id"], "Unknown video"
        )

    return {
        "selected_video_count": len(video_ids),
        "result_count": len(results),
        "results": results,
    }



@app.get("/api/videos/{video_id}/file")
def get_video_file(video_id: str):
    with get_connection() as conn:
        video = conn.execute(
            "SELECT storage_path FROM videos WHERE id = ?",
            (video_id,),
        ).fetchone()

    if video is None:
        raise HTTPException(
            status_code=404,
            detail="Video not found.",
        )

    path = Path(video["storage_path"])

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video file is missing.",
        )

    return FileResponse(path)
