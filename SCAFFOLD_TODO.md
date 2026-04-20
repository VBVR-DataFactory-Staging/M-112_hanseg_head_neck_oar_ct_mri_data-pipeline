# M-112 scaffold TODO

Scaffolded from template: `M-037_amos_multi_organ_segmentation_data-pipeline` (2026-04-20)

## Status
- [x] config.py updated (domain=hanseg_head_neck_oar_ct_mri, s3_prefix=M-112_HaN-Seg/raw/, fps=3)
- [ ] core/download.py: update URL / Kaggle slug / HF repo_id
- [ ] src/download/downloader.py: adapt to dataset file layout
- [ ] src/pipeline/_phase2/*.py: adapt raw → frames logic (inherited from M-037_amos_multi_organ_segmentation_data-pipeline, likely needs rework)
- [ ] examples/generate.py: verify end-to-end on 3 samples

## Task prompt
This head-neck CT (or paired MR) slice (HaN-Seg). Segment 30 organs-at-risk (parotid, submandibular, brainstem, optic chiasm, etc.).

Fleet runs likely FAIL on first attempt for dataset parsing; iterate based on fleet logs at s3://vbvr-final-data/_logs/.
