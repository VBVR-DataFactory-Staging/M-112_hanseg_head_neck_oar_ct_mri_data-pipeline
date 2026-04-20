"""Pipeline configuration for M-037 (amos_multi_organ_segmentation)."""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Configuration for M-037 pipeline.

    Inherited from PipelineConfig:
        num_samples: Optional[int]  # Max samples (None = all)
        domain: str
        output_dir: Path
        split: str
    """

    domain: str = Field(default="hanseg_head_neck_oar_ct_mri")

    s3_bucket: str = Field(
        default="med-vr-datasets",
        description="S3 bucket containing the raw M-037 data",
    )
    s3_prefix: str = Field(
        default="M-112_HaN-Seg/raw/",
        description="S3 key prefix for the dataset raw data",
    )
    fps: int = Field(
        default=3,
        description="Frames per second for the generated videos",
    )
    raw_dir: Path = Field(
        default=Path("raw"),
        description="Local directory for downloaded raw data",
    )
    task_prompt: str = Field(
        default="This head-neck CT (or paired MR) slice (HaN-Seg). Segment 30 organs-at-risk (parotid, submandibular, brainstem, optic chiasm, etc.).",
        description="The task instruction shown to the reasoning model.",
    )
