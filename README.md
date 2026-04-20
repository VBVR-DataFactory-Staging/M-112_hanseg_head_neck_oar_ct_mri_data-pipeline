# M-037 — Amos Multi Organ Segmentation

AMOS 2022 abdominal CT/MRI 15-organ segmentation.

This repository is part of the Med-VR data-pipeline suite for the VBVR
(Very Big Video Reasoning) benchmark. It produces standardized video-
reasoning task samples from the underlying raw medical dataset.

## Task

**Prompt shown to the model**:

> This is an abdominal CT / MRI scan. Simultaneously segment 15 abdominal organs: spleen (purple), left and right kidneys (orange), gallbladder (lime), esophagus (cyan), liver (green), stomach (yellow), aorta (red), inferior vena cava (blue), pancreas (pink), adrenal glands (teal), duodenum (brown), bladder (magenta), and prostate/uterus (gray). Fill each organ with its assigned color and draw a contour boundary.

## S3 Raw Data

```
s3://med-vr-datasets/M-037_052_AMOS/raw/
```

## Quick Start

```bash
pip install -r requirements.txt

# Generate samples (downloads raw from S3 on first run)
python examples/generate.py

# Generate only N samples
python examples/generate.py --num-samples 10

# Custom output directory
python examples/generate.py --output data/my_output
```

## Output Layout

```
data/questions/amos_multi_organ_segmentation_task/
├── task_0000/
│   ├── first_frame.png
│   ├── final_frame.png
│   ├── first_video.mp4
│   ├── last_video.mp4
│   ├── ground_truth.mp4
│   ├── prompt.txt
│   └── metadata.json
├── task_0001/
└── ...
```

## Example Output

See [`examples/example_output/`](examples/example_output/) for 2 reference
samples committed alongside the code.

## Configuration

`src/pipeline/config.py` (`TaskConfig`):

| Field | Default | Description |
|---|---|---|
| `domain` | `"amos_multi_organ_segmentation"` | Task domain string used in output paths. |
| `s3_bucket` | `"med-vr-datasets"` | S3 bucket containing raw data. |
| `s3_prefix` | `"M-037_052_AMOS/raw/"` | S3 key prefix for raw data. |
| `fps` | `10` | Output video FPS. |
| `raw_dir` | `Path("raw")` | Local raw cache directory. |
| `num_samples` | `None` | Max samples (None = all). |

## Repository Structure

```
M-037_amos_multi_organ_segmentation_data-pipeline/
├── core/                ← shared pipeline framework (verbatim)
├── eval/                ← shared evaluation utilities
├── src/
│   ├── download/
│   │   └── downloader.py   ← S3 raw-data downloader
│   └── pipeline/
│       ├── config.py        ← task config
│       ├── pipeline.py      ← TaskPipeline
│       ├── transforms.py    ← visualization helpers (shim)
│       └── _phase2/         ← vendored phase2 prototype logic
├── examples/
│   ├── generate.py
│   └── example_output/      ← committed reference samples
├── requirements.txt
├── README.md
└── LICENSE
```
