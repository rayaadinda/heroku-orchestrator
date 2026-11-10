import modal
from train_worker import run_training_pipeline

app = modal.App("doctortool-ocr-worker")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("ultralytics>=8.4.0", "boto3", "opencv-python-headless", "requests")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=900,
)
def train_on_gpu(job_payload: dict):
    run_training_pipeline()


@app.function(image=image)
@modal.web_endpoint(method="POST")
def trigger_training(job_payload: dict):
    train_on_gpu.spawn(job_payload)
    return {"ok": True, "status": "gpu_training_dispatched", "jobId": job_payload.get("id")}
