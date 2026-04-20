"""M-037: AMOS 2022 abdominal 15-organ segmentation.

Labels: 1=spleen, 2=right kidney, 3=left kidney, 4=gallbladder, 5=esophagus,
6=liver, 7=stomach, 8=aorta, 9=postcava, 10=pancreas, 11=right adrenal gland,
12=left adrenal gland, 13=duodenum, 14=bladder, 15=prostate/uterus.

Case B: abdominal CT/MRI axial slice sequence, FPS=10.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import nibabel as nib
from common import (
    DATA_ROOT, window_ct, to_rgb, overlay_multi, write_task,
    COLORS, fit_square, pick_annotated_idx,
)

PID = "M-037"
TASK_NAME = "amos_multi_organ_segmentation"
FPS = 10

ORGANS = [
    ("spleen",              "purple"),
    ("right_kidney",        "orange"),
    ("left_kidney",         "orange"),
    ("gallbladder",         "lime"),
    ("esophagus",           "cyan"),
    ("liver",               "green"),
    ("stomach",             "yellow"),
    ("aorta",               "red"),
    ("postcava_IVC",        "blue"),
    ("pancreas",            "pink"),
    ("right_adrenal_gland", "teal"),
    ("left_adrenal_gland",  "teal"),
    ("duodenum",            "brown"),
    ("bladder",             "magenta"),
    ("prostate_or_uterus",  "gray"),
]
COLOR_LIST = [(n, COLORS[c]) for n, c in ORGANS]

PROMPT = (
    "This is an abdominal CT / MRI scan from the AMOS 2022 dataset. "
    "Segment all 15 abdominal organs simultaneously: spleen (purple), left/right "
    "kidneys (orange), gallbladder (lime), esophagus (cyan), liver (green), "
    "stomach (yellow), aorta (red), IVC (blue), pancreas (pink), adrenal glands "
    "(teal), duodenum (brown), bladder (magenta), and prostate/uterus (gray). "
    "Overlay each organ with its assigned color and draw contour boundaries."
)


def process_case(img_path: Path, lbl_path: Path, task_idx: int):
    img_vol = np.transpose(nib.load(str(img_path)).get_fdata(), (2, 1, 0))
    lbl_vol = np.transpose(nib.load(str(lbl_path)).get_fdata(), (2, 1, 0)).astype(np.int32)

    n = img_vol.shape[0]
    step = max(1, n // 60)
    indices = list(range(0, n, step))[:60]

    first_frames, last_frames, gt_frames, flags = [], [], [], []
    for z in indices:
        ct = window_ct(img_vol[z])
        rgb = to_rgb(ct)
        rgb = fit_square(rgb, 512)
        lab = lbl_vol[z].astype(np.int32)
        lab_square = fit_square(lab.astype(np.int16), 512).astype(np.int32)
        ann = overlay_multi(rgb, lab_square, COLOR_LIST)
        first_frames.append(rgb)
        last_frames.append(ann)
        has = bool((lab_square > 0).any())
        flags.append(has)
        if has:
            gt_frames.append(ann)
    if not gt_frames:
        gt_frames = last_frames[:5]
    pick = pick_annotated_idx(flags)
    first_frame = first_frames[pick]
    final_frame = last_frames[pick]

    meta = {
        "task": "AMOS22 abdominal 15-organ segmentation",
        "dataset": "AMOS 2022",
        "case_id": img_path.stem.replace(".nii", ""),
        "modality": "CT or MRI (AMOS mixed cohort)",
        "organs": [n for n, _ in ORGANS],
        "colors": {n: c for n, c in ORGANS},
        "fps_source": "manual (case B slice sequence)",
        "num_slices_total": int(n),
        "num_slices_used": len(indices),
        "source_split": "train",
    }
    return write_task(PID, TASK_NAME, task_idx,
                      first_frame, final_frame,
                      first_frames, last_frames, gt_frames,
                      PROMPT, meta, FPS)


def main():
    root = DATA_ROOT / "_extracted" / "21_AMOS" / "amos22"
    cases = sorted(root.glob("imagesTr/amos_*.nii.gz"))
    for i, img in enumerate(cases[:2]):
        lbl = root / "labelsTr" / img.name
        d = process_case(img, lbl, i)
        print(f"  wrote {d}")


if __name__ == "__main__":
    main()
