"""
FastAPI server for road degradation detection.
Provides REST API for image/video upload, detection, and results retrieval.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from pathlib import Path
import shutil
import uuid
import json
from datetime import datetime
from ultralytics import YOLO
import cv2
import numpy as np

# Configuration
MODEL_PATH = Path("models/best.pt")
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="Road Degradation Detection API",
    description="API for detecting and geolocating road infrastructure anomalies",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Load model
model = None

def load_model():
    """Load YOLO model."""
    global model
    if MODEL_PATH.exists():
        print(f"📥 Loading model: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
        print("✅ Model loaded successfully")
    else:
        print("⚠️ Model not found. Please train a model first.")


# Pydantic models
class DetectionResponse(BaseModel):
    """Detection response model."""
    class_id: int
    class_name: str
    confidence: float
    bbox: dict
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class JobStatus(BaseModel):
    """Job status model."""
    job_id: str
    status: str
    progress: Optional[float] = None
    result_path: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


# Job tracking
jobs = {}

CLASS_NAMES = ['pothole', 'longitudinal_crack', 'crazing', 'faded_marking']


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Road Degradation Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "detect_image": "/detect/image",
            "detect_video": "/detect/video",
            "job_status": "/job/{job_id}",
            "results": "/results/{filename}"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
):
    """
    Detect degradations in a single image.
    
    Args:
        file: Image file
        conf_threshold: Confidence threshold
        latitude: Optional GPS latitude
        longitude: Optional GPS longitude
    
    Returns:
        Detection results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Run detection
        results = model.predict(
            source=str(file_path),
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        # Process results
        detections = []
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            detection = DetectionResponse(
                class_id=cls,
                class_name=CLASS_NAMES[cls],
                confidence=conf,
                bbox={
                    'xmin': float(xyxy[0]),
                    'ymin': float(xyxy[1]),
                    'xmax': float(xyxy[2]),
                    'ymax': float(xyxy[3])
                },
                latitude=latitude,
                longitude=longitude
            )
            detections.append(detection)
        
        # Save annotated image
        annotated_img = result.plot()
        annotated_path = RESULTS_DIR / f"{file_id}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_img)
        
        return {
            "success": True,
            "image_id": file_id,
            "num_detections": len(detections),
            "detections": [det.dict() for det in detections],
            "annotated_image": f"/results/{annotated_path.name}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up uploaded file
        file_path.unlink(missing_ok=True)


@app.post("/detect/video")
async def detect_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    skip_frames: int = 5
):
    """
    Detect degradations in video (background task).
    
    Args:
        file: Video file
        conf_threshold: Confidence threshold
        skip_frames: Process every Nth frame
    
    Returns:
        Job ID for tracking
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create job
    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create job entry
    jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "file_path": str(file_path),
        "conf_threshold": conf_threshold,
        "skip_frames": skip_frames
    }
    
    # Start background task
    background_tasks.add_task(process_video_task, job_id, file_path, conf_threshold, skip_frames)
    
    return {
        "success": True,
        "job_id": job_id,
        "message": "Video processing started",
        "status_url": f"/job/{job_id}"
    }


def process_video_task(job_id: str, video_path: Path, conf_threshold: float, skip_frames: int):
    """
    Process video in background.
    
    Args:
        job_id: Job ID
        video_path: Path to video
        conf_threshold: Confidence threshold
        skip_frames: Process every Nth frame
    """
    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Process frames
        detections = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress = (frame_count / total_frames) * 100
            jobs[job_id]["progress"] = progress
            
            # Process every Nth frame
            if frame_count % skip_frames == 0:
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    verbose=False
                )
                
                result = results[0]
                boxes = result.boxes
                
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    detections.append({
                        'frame_number': frame_count,
                        'timestamp_sec': frame_count / fps,
                        'class_id': cls,
                        'class_name': CLASS_NAMES[cls],
                        'confidence': conf,
                        'bbox': {
                            'xmin': float(xyxy[0]),
                            'ymin': float(xyxy[1]),
                            'xmax': float(xyxy[2]),
                            'ymax': float(xyxy[3])
                        }
                    })
            
            frame_count += 1
        
        cap.release()
        
        # Save results
        result_path = RESULTS_DIR / f"{job_id}_detections.json"
        with open(result_path, 'w') as f:
            json.dump({
                'job_id': job_id,
                'total_frames': total_frames,
                'processed_frames': frame_count // skip_frames,
                'total_detections': len(detections),
                'detections': detections
            }, f, indent=2)
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100.0
        jobs[job_id]["result_path"] = f"/results/{result_path.name}"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["num_detections"] = len(detections)
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    
    finally:
        # Clean up video file
        video_path.unlink(missing_ok=True)


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get job status.
    
    Args:
        job_id: Job ID
    
    Returns:
        Job status
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return {
        "jobs": jobs,
        "total": len(jobs)
    }


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete job and its results.
    
    Args:
        job_id: Job ID
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete result file if exists
    if "result_path" in jobs[job_id]:
        result_file = RESULTS_DIR / Path(jobs[job_id]["result_path"]).name
        result_file.unlink(missing_ok=True)
    
    # Remove job
    del jobs[job_id]
    
    return {"success": True, "message": "Job deleted"}


@app.get("/classes")
async def get_classes():
    """Get list of detection classes."""
    return {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES)
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
