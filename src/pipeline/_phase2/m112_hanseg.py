"""M-112: HaN-Seg head and neck OAR segmentation.

Dataset: HaN-Seg 2023 (https://zenodo.org/records/7442914)
  42 subjects with paired CT + MR-T1 + 30 OAR segmentations (NRRD format).

Raw layout (speculative, will refine after raw lands in S3):
    _extracted/M-112_HaN-Seg/HaN-Seg/set_1/case_01/
      case_01_IMG_CT.nrrd
      case_01_IMG_MR_T1.nrrd
      case_01_OAR_BrainStem.seg.nrrd
      case_01_OAR_... (30 structures)

Uses CT volume + all per-OAR masks fused into a multi-label volume.
Case B: axial slice sequence, fps=3.
"""
from __future__ import annotations
import re
import numpy as np
from pathlib import Path
from common import (
    DATA_ROOT, window_ct, to_rgb, overlay_multi, write_task,
    COLORS, fit_square, pick_annotated_idx,
)

PID = "M-112"
TASK_NAME = "hanseg_head_neck_oar_ct_mri"
FPS = 3

# 30 OAR targets in HaN-Seg (names mapped to display colors)
OARS = [
    ("Brainstem",              "green"),
    ("Chiasm",                 "cyan"),
    ("Glnd_Submand_L",         "yellow"),
    ("Glnd_Submand_R",         "yellow"),
    ("Parotid_L",              "orange"),
    ("Parotid_R",              "orange"),
    ("OpticNrv_L",             "red"),
    ("OpticNrv_R",             "red"),
    ("Eye_L",                  "pink"),
    ("Eye_R",                  "pink"),
    ("Larynx_G",               "purple"),
    ("Larynx_SG",              "purple"),
    ("Lips",                   "magenta"),
    ("Mandible",               "teal"),
    ("Oral_Cav",               "brown"),
    ("SpinalCord",             "blue"),
    ("Cavity_Oral",            "brown"),
    ("Musc_Constrict_I",       "lime"),
    ("Musc_Constrict_M",       "lime"),
    ("Musc_Constrict_S",       "lime"),
    ("Esophagus_S",            "navy"),
    ("Trachea",                "gray"),
    ("Bone_Mandible",          "teal"),
    ("BuccalMucosa",           "brown"),
    ("A_Carotid_L",            "red"),
    ("A_Carotid_R",            "red"),
    ("Cochlea_L",              "white"),
    ("Cochlea_R",              "white"),
    ("Cricopharyngeus",        "lime"),
    ("Glnd_Lacrimal_L",        "pink"),
]
COLOR_LIST = [(n, COLORS[c]) for n, c in OARS]

PROMPT = (
    "This is a head-and-neck CT scan from the HaN-Seg 2023 dataset. "
    "Segment the 30 organs-at-risk (OARs) simultaneously for radiotherapy planning: "
    "brainstem, optic chiasm, parotid/submandibular glands, optic nerves, eyes, "
    "larynx, lips, mandible, oral cavity, spinal cord, pharyngeal constrictor muscles, "
    "esophagus, trachea, carotid arteries, cochleae, and lacrimal glands. "
    "Overlay each OAR with a distinct color and draw contour boundaries on every slice."
)


def load_nrrd(path: Path):
    import nrrd
    data, _header = nrrd.read(str(path))
    # NRRD data is typically (x, y, z); transpose to (z, y, x)
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))
    return data


def process_case(case_dir: Path, task_idx: int):
    # find CT image
    ct_files = list(case_dir.glob("*IMG_CT*"))
    if not ct_files:
        ct_files = list(case_dir.glob("*CT*.nrrd")) or list(case_dir.glob("*CT*.nii*"))
    if not ct_files:
        return None
    ct_path = ct_files[0]
    img_vol = load_nrrd(ct_path) if ct_path.suffix == ".nrrd" else _load_nifti(ct_path)

    # fuse per-OAR masks into multi-label volume
    lbl_vol = np.zeros(img_vol.shape, dtype=np.int32)
    for label_idx, (oar_name, _color) in enumerate(OARS, start=1):
        pattern = f"*OAR_{oar_name}*"
        oar_files = list(case_dir.glob(pattern))
        if not oar_files:
            continue
        mask = load_nrrd(oar_files[0]) if oar_files[0].suffix == ".nrrd" else _load_nifti(oar_files[0])
        # binary mask → set voxels to this OAR's label index
        lbl_vol[mask > 0] = label_idx

    n = img_vol.shape[0]
    step = max(1, n // 60)
    indices = list(range(0, n, step))[:60]

    first_frames, last_frames, gt_frames, flags = [], [], [], []
    for z in indices:
        ct = window_ct(img_vol[z], wl=40, ww=400)
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

    meta = {
        "task": "HaN-Seg head and neck OAR segmentation",
        "dataset": "HaN-Seg 2023",
        "case_id": case_dir.name,
        "modality": "CT",
        "organs": [n for n, _ in OARS],
        "colors": {n: c for n, c in OARS},
        "fps_source": "derived (case B slice sequence, fps=3 per HARNESS)",
        "num_slices_total": int(n),
        "num_slices_used": len(indices),
        "source_split": "set_1",
    }
    return write_task(PID, TASK_NAME, task_idx,
                      first_frames[pick], last_frames[pick],
                      first_frames, last_frames, gt_frames,
                      PROMPT, meta, FPS)


def _load_nifti(path: Path):
    import nibabel as nib
    arr = nib.load(str(path)).get_fdata()
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))
    return arr


def main():
    root = DATA_ROOT / "_extracted" / "M-112_HaN-Seg"
    # try multiple possible layouts
    candidates = list(root.glob("**/case_*"))
    case_dirs = sorted({p for p in candidates if p.is_dir()})
    if not case_dirs:
        case_dirs = sorted({p for p in root.glob("**/set_*/*") if p.is_dir()})
    print(f"  {len(case_dirs)} HaN-Seg cases")
    for i, cd in enumerate(case_dirs):
        d = process_case(cd, i)
        if d:
            print(f"  wrote {d}")


if __name__ == "__main__":
    main()
