import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import boto3
import cv2
import requests
from ultralytics import YOLO

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "ad7567bc6207096d4bad4857b406dd1b")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "18878903125e36ddfb5da4e4f9211b29")
R2_SECRET_ACCESS_KEY = os.environ.get(
    "R2_SECRET_ACCESS_KEY",
    "9f9821d954b6e6ea6cfbca24285a97125668992dd419928f78552ac68754e831",
)
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "doctortool-ocr-models")
R2_PUBLIC_BASE_URL = os.environ.get(
    "R2_PUBLIC_BASE_URL", "https://pub-8b3cc4a8d5ff4365b18471f08bb797cb.r2.dev"
)
HEROKU_BASE_URL = os.environ.get(
    "HEROKU_BASE_URL", "https://doctortool-ocr-orchestrator-d7500d9b91a8.herokuapp.com"
).rstrip("/")
WORKER_SHARED_SECRET = os.environ.get(
    "WORKER_SHARED_SECRET", "doctortool-colab-worker-secret-2026"
)

DEVICE_KIND_TO_STAGE1_CLASS = {
    "bloodPressure": 0,
    "oximeter": 1,
    "scale": 2,
    "thermometer": 3,
}

s3 = boto3.client(
    service_name="s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)


def get_pending_jobs():
    try:
        resp = requests.get(
            f"{HEROKU_BASE_URL}/api/v1/worker/pending-jobs",
            headers={"X-Worker-Token": WORKER_SHARED_SECRET},
            timeout=10,
        )
        if resp.ok:
            jobs = resp.json().get("jobs", [])
            if jobs:
                return jobs
    except Exception as e:
        print(f"Warning: Could not fetch jobs from Heroku ({e}), falling back to R2...")

    listed = s3.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix="calibrations/")
    fallback_jobs = []
    for obj in listed.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".json") and not key.endswith("_processed.json"):
            body = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)["Body"].read().decode("utf-8")
            meta = json.loads(body)
            if not meta.get("processedByWorker"):
                meta["metaObjectKey"] = key
                fallback_jobs.append(meta)
    return fallback_jobs


def run_training_pipeline():
    jobs = get_pending_jobs()
    if not jobs:
        print("No pending calibration jobs found. Exiting.")
        return

    print(f"Found {len(jobs)} pending calibration job(s). Starting automated fine-tuning...")

    work_dir = Path("work_self_learning")
    frames_dir = work_dir / "dataset_stage1" / "images" / "train"
    labels_dir = work_dir / "dataset_stage1" / "labels" / "train"
    val_frames_dir = work_dir / "dataset_stage1" / "images" / "val"
    val_labels_dir = work_dir / "dataset_stage1" / "labels" / "val"

    for d in [frames_dir, labels_dir, val_frames_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    teacher_pt = work_dir / "stage1_teacher.pt"
    try:
        s3.download_file(R2_BUCKET_NAME, "models/stage1/device_roi_model_v2.0.0.pt", str(teacher_pt))
        teacher_model = YOLO(str(teacher_pt))
    except Exception:
        teacher_model = YOLO("yolo26n.pt")

    extracted_count = 0
    processed_job_ids = []

    for job in jobs:
        job_id = job["id"]
        device_kind = job.get("deviceKind", "bloodPressure")
        forced_class_id = DEVICE_KIND_TO_STAGE1_CLASS.get(device_kind, 0)
        video_key = job.get("videoObjectKey")
        if not video_key:
            continue

        local_mp4 = work_dir / f"{job_id}.mp4"
        print(f"Downloading {video_key} -> {local_mp4}...")
        s3.download_file(R2_BUCKET_NAME, video_key, str(local_mp4))

        cap = cv2.VideoCapture(str(local_mp4))
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 4 != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness < 60.0:
                continue

            results = teacher_model.predict(frame, imgsz=320, conf=0.20, verbose=False)
            boxes = results[0].boxes

            yolo_lines = []
            found_device = False
            found_lcd = False

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    xywhn = box.xywhn[0].tolist()
                    if cls_id == 4 and not found_lcd:
                        yolo_lines.append(f"4 {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")
                        found_lcd = True
                    elif cls_id in [0, 1, 2, 3] and not found_device:
                        yolo_lines.append(f"{forced_class_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")
                        found_device = True

            if not found_device:
                yolo_lines.append(f"{forced_class_id} 0.500000 0.500000 0.760000 0.760000")
            if not found_lcd:
                yolo_lines.append("4 0.500000 0.480000 0.520000 0.480000")

            is_val = (extracted_count % 5 == 0)
            target_img = (val_frames_dir if is_val else frames_dir) / f"{job_id}_f{frame_idx}.jpg"
            target_lbl = (val_labels_dir if is_val else labels_dir) / f"{job_id}_f{frame_idx}.txt"

            cv2.imwrite(str(target_img), frame)
            target_lbl.write_text("\n".join(yolo_lines) + "\n")
            extracted_count += 1

        cap.release()
        processed_job_ids.append(job_id)

    if extracted_count == 0:
        print("No sharp frames could be extracted.")
        return

    print(f"Extracted {extracted_count} frames. Training YOLO26 for 15 epochs...")
    yaml_path = work_dir / "dataset_stage1" / "data.yaml"
    yaml_path.write_text(
        f"path: {work_dir.absolute() / 'dataset_stage1'}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: device_blood_pressure\n"
        "  1: device_oximeter\n"
        "  2: device_scale\n"
        "  3: device_thermometer\n"
        "  4: lcd_screen\n"
    )

    finetune_model = YOLO(str(teacher_pt) if teacher_pt.exists() else "yolo26n.pt")
    finetune_model.train(
        data=str(yaml_path),
        epochs=15,
        imgsz=320,
        batch=16,
        lr0=0.0005,
        freeze=10,
        project=str(work_dir / "runs"),
        name="stage1_auto",
        exist_ok=True,
    )

    best_pt = work_dir / "runs" / "stage1_auto" / "weights" / "best.pt"
    export_model = YOLO(str(best_pt))
    val_metrics = export_model.val(data=str(yaml_path), imgsz=320)
    new_map50 = float(val_metrics.box.map50)
    print(f"Fine-tuning validation mAP50: {new_map50:.4f}")

    exported_litert = Path(export_model.export(format="litert", imgsz=320))
    new_version = f"2.0.{int(time.time()) % 10000}"
    r2_key = f"models/stage1/device_roi_model_v{new_version}.tflite"

    sha256_hash = hashlib.sha256(exported_litert.read_bytes()).hexdigest()
    file_size = exported_litert.stat().st_size

    print(f"Uploading new model to R2: {r2_key} ({file_size} bytes)...")
    s3.upload_file(
        str(exported_litert),
        R2_BUCKET_NAME,
        r2_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )

    complete_payload = {
        "jobIds": processed_job_ids,
        "stage1": {
            "version": new_version,
            "downloadUrl": f"{R2_PUBLIC_BASE_URL}/{r2_key}",
            "sha256": sha256_hash,
            "fileSizeBytes": file_size,
            "inputSize": 320,
            "numClasses": 5,
            "labels": [
                "device_blood_pressure",
                "device_oximeter",
                "device_scale",
                "device_thermometer",
                "lcd_screen",
            ],
            "map50": new_map50,
        },
    }

    try:
        resp = requests.post(
            f"{HEROKU_BASE_URL}/api/v1/worker/complete-job",
            headers={"X-Worker-Token": WORKER_SHARED_SECRET},
            json=complete_payload,
            timeout=15,
        )
        print("Heroku complete-job status:", resp.status_code, resp.text)
    except Exception as e:
        print("Failed notifying Heroku, updating R2 manifest directly:", e)
        manifest_obj = json.loads(
            s3.get_object(Bucket=R2_BUCKET_NAME, Key="models/latest.json")["Body"].read().decode("utf-8")
        )
        manifest_obj["data"]["models"]["stage1_device_roi"] = complete_payload["stage1"]
        manifest_obj["data"]["serverTimestamp"] = int(time.time() * 1000)
        s3.put_object(
            Bucket=R2_BUCKET_NAME,
            Key="models/latest.json",
            Body=json.dumps(manifest_obj, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    print(f"Pipeline finished! Model v{new_version} published OTA.")


if __name__ == "__main__":
    run_training_pipeline()
