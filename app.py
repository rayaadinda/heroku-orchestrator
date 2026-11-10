import os
import json
import time
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="DoctorTool OCR Self-Learning Orchestrator", version="2.0.0")

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "ad7567bc6207096d4bad4857b406dd1b")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "doctortool-ocr-models")
R2_PUBLIC_BASE_URL = os.environ.get(
    "R2_PUBLIC_BASE_URL", "https://pub-8b3cc4a8d5ff4365b18471f08bb797cb.r2.dev"
)
WORKER_SHARED_SECRET = os.environ.get("WORKER_SHARED_SECRET", "doctortool-colab-worker-secret-2026")

QUEUE_OBJECT_KEY = "jobs/calibration_queue.json"
MANIFEST_OBJECT_KEY = "models/latest.json"

s3 = boto3.client(
    service_name="s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)


def read_r2_json(key: str, default_val):
    try:
        res = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
        return json.loads(res["Body"].read().decode("utf-8"))
    except ClientError:
        return default_val


def write_r2_json(key: str, payload) -> None:
    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
        CacheControl="no-cache, no-store, must-revalidate",
    )


class RegisterCalibrationRequest(BaseModel):
    id: str
    clinicId: str
    nurseUserId: str
    deviceKind: str
    brandName: str
    modelName: str
    groundTruth: Dict[str, Optional[float]]
    durationSeconds: float
    fileSizeBytes: int
    videoSha256: str
    videoObjectKey: str
    publicVideoUrl: str


class WorkerCompleteSlotPayload(BaseModel):
    version: str
    downloadUrl: str
    sha256: str
    fileSizeBytes: int
    inputSize: int = 320
    numClasses: int
    labels: List[str]
    map50: float


class WorkerCompleteJobRequest(BaseModel):
    jobIds: List[str]
    stage1: Optional[WorkerCompleteSlotPayload] = None
    stage2: Optional[WorkerCompleteSlotPayload] = None


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "doctortool-ocr-orchestrator", "timestamp": int(time.time() * 1000)}


@app.post("/api/v1/calibrations/register")
def register_calibration_job(req: RegisterCalibrationRequest):
    queue_data = read_r2_json(QUEUE_OBJECT_KEY, {"jobs": []})
    existing_ids = {j["id"] for j in queue_data["jobs"]}
    if req.id not in existing_ids:
        job_entry = req.model_dump()
        job_entry["status"] = "QUEUED"
        job_entry["queuedAt"] = int(time.time() * 1000)
        queue_data["jobs"].append(job_entry)
        write_r2_json(QUEUE_OBJECT_KEY, queue_data)
    return {"ok": True, "jobId": req.id, "queuedCount": len([j for j in queue_data["jobs"] if j["status"] == "QUEUED"])}


@app.get("/api/v1/worker/pending-jobs")
def get_pending_jobs(x_worker_token: str = Header(default="")):
    if x_worker_token != WORKER_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized worker token")
    queue_data = read_r2_json(QUEUE_OBJECT_KEY, {"jobs": []})
    pending = [j for j in queue_data["jobs"] if j.get("status") == "QUEUED"]
    return {"pendingCount": len(pending), "jobs": pending}


@app.post("/api/v1/worker/complete-job")
def complete_training_job(req: WorkerCompleteJobRequest, x_worker_token: str = Header(default="")):
    if x_worker_token != WORKER_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized worker token")

    manifest = read_r2_json(MANIFEST_OBJECT_KEY, {})
    now_ms = int(time.time() * 1000)

    if req.stage1:
        if req.stage1.map50 < 0.80:
            raise HTTPException(status_code=422, detail=f"Stage 1 mAP50 ({req.stage1.map50:.3f}) below safety gate 0.80")
        manifest["stage1_device_roi"] = {
            "slot": "stage1_device_roi",
            "version": req.stage1.version,
            "clinicScope": "global",
            "downloadUrl": req.stage1.downloadUrl,
            "sha256": req.stage1.sha256,
            "fileSizeBytes": req.stage1.fileSizeBytes,
            "inputSize": req.stage1.inputSize,
            "numClasses": req.stage1.numClasses,
            "labels": req.stage1.labels,
            "updatedAt": now_ms,
        }

    if req.stage2:
        if req.stage2.map50 < 0.80:
            raise HTTPException(status_code=422, detail=f"Stage 2 mAP50 ({req.stage2.map50:.3f}) below safety gate 0.80")
        manifest["stage2_digit_7seg"] = {
            "slot": "stage2_digit_7seg",
            "version": req.stage2.version,
            "clinicScope": "global",
            "downloadUrl": req.stage2.downloadUrl,
            "sha256": req.stage2.sha256,
            "fileSizeBytes": req.stage2.fileSizeBytes,
            "inputSize": req.stage2.inputSize,
            "numClasses": req.stage2.numClasses,
            "labels": req.stage2.labels,
            "updatedAt": now_ms,
        }

    manifest["updatedAt"] = now_ms
    write_r2_json(MANIFEST_OBJECT_KEY, manifest)

    queue_data = read_r2_json(QUEUE_OBJECT_KEY, {"jobs": []})
    completed_set = set(req.jobIds)
    for job in queue_data["jobs"]:
        if job["id"] in completed_set:
            job["status"] = "COMPLETED"
            job["completedAt"] = now_ms
    write_r2_json(QUEUE_OBJECT_KEY, queue_data)

    return {"ok": True, "manifestUpdated": True, "completedJobs": list(completed_set)}