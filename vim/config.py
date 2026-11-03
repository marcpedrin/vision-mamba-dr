"""
Config: loads configs/default.yaml into dataclasses.

    So the code can say cfg.model.embed_dim instead of magic numbers.
    Values that follow from the settings (J, M, E, cls index) are worked out here once.

"""

import copy
from dataclasses import asdict, dataclass, field

import yaml


@dataclass
class ModelConfig:
    #model: section of the yaml, Vim-Ti sizes

    #paper uses 224
    img_size: int = 128
    patch_size: int = 16
    #RGB
    in_chans: int = 3
    #D
    embed_dim: int = 192
    #E = expand * D
    expand: int = 2
    #N, hidden state size
    d_state: int = 16
    #number of blocks (paper 24)
    depth: int = 12
    #DR grades 0-4
    num_classes: int = 5
    #"middle" or "head"
    cls_position: str = "middle"
    #conv1d kernel, Algorithm 1 line 7
    conv_kernel: int = 4
    #starting range for Delta
    dt_min: float = 0.001
    dt_max: float = 0.1
    #official Mamba D*x skip (False = printed Algorithm 1)
    d_skip: bool = False

    @property
    def grid_size(self) -> int:
        #128 // 16 = 8 patches per side
        return self.img_size // self.patch_size

    @property
    def num_patches(self) -> int:
        #J = 8 * 8 = 64
        return self.grid_size * self.grid_size

    @property
    def seq_len(self) -> int:
        #M = 64 patches + 1 class token = 65
        return self.num_patches + 1

    @property
    def expand_dim(self) -> int:
        #E = 2 * 192 = 384
        return self.expand * self.embed_dim

    @property
    def cls_index(self) -> int:
        #middle -> 32
        if self.cls_position == "middle":
            return self.num_patches // 2
        #head -> 0 like ViT
        if self.cls_position == "head":
            return 0
        raise ValueError(
            f"cls_position must be 'middle' or 'head', got {self.cls_position!r}"
        )


@dataclass
class DataConfig:
    #data: section of the yaml

    #EyePACS, training
    eyepacs_image_dir: str = "datasets/eyepacs/train"
    eyepacs_labels_csv: str = "datasets/eyepacs/trainLabels.csv"
    eyepacs_image_ext: str = ".jpeg"
    #Messidor-2, eval only
    messidor_image_dir: str = "datasets/messidor2/images"
    messidor_labels_csv: str = "datasets/messidor2/messidor_data.csv"
    messidor_id_column: str = "image_id"
    messidor_label_column: str = "adjudicated_dr_grade"
    messidor_image_ext: str = ".png"
    #fraction of patients for val/test
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    #pixels this dark count as black border
    crop_threshold: int = 10
    #augmentation
    rotation_degrees: float = 15.0
    color_jitter: float = 0.1
    #dataloader
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class TrainConfig:
    #train: section of the yaml

    #paper 300
    epochs: int = 30
    #paper 1024
    batch_size: int = 32
    #AdamW
    lr: float = 5.0e-4
    weight_decay: float = 0.05
    #warmup then cosine down to min_lr
    warmup_epochs: int = 3
    min_lr: float = 1.0e-5
    grad_clip: float = 1.0
    #mixed precision, gpu only
    amp: bool = False
    class_weights: bool = True
    output_dir: str = "outputs/default"
    #--smoke-test settings
    smoke_test_images: int = 100
    smoke_test_epochs: int = 2
    smoke_test_batch_size: int = 8


@dataclass
class EvalConfig:
    #eval: section of the yaml
    checkpoint: str = "outputs/default/best.pt"
    results_json: str = "outputs/default/results.json"
    figures_dir: str = "report/figures"


@dataclass
class Config:
    #whole experiment, one sub config per yaml section
    seed: int = 42
    #default_factory so each Config gets its own fresh copy
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def apply_override(raw: dict, override: str) -> None:
    """Change one value in the raw yaml dict, e.g. "train.epochs=5"."""
    if "=" not in override:
        raise ValueError(
            f"Override must look like 'section.key=value', got {override!r}"
        )

    #split on first = only
    dotted_key, value_text = override.split("=", 1)

    #"train.epochs" -> ["train", "epochs"]
    parts = dotted_key.split(".")

    #walk down to the dict that holds the last key
    node = raw
    for part in parts[:-1]:
        #typos should fail, not make new sections
        if part not in node:
            raise KeyError(f"Unknown config section {part!r} in override {override!r}")
        node = node[part]

    key = parts[-1]
    if key not in node:
        raise KeyError(f"Unknown config key {key!r} in override {override!r}")

    #yaml picks the type: "5" -> 5, "true" -> True
    value = yaml.safe_load(value_text)
    #yaml reads "5e-4" as a string (needs "5.0e-4"), so try float
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    node[key] = value


def config_from_dict(raw: dict) -> Config:
    """Build a Config from the nested dict read from yaml."""
    #**dict turns keys into named args, unknown keys raise an error
    return Config(
        seed=raw.get("seed", 42),
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
        eval=EvalConfig(**raw.get("eval", {})),
    )


def load_config(path: str, overrides=None) -> Config:
    """Read the yaml, apply "section.key=value" overrides, return a Config."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    for override in overrides or []:
        apply_override(raw, override)

    return config_from_dict(raw)


def config_to_dict(cfg: Config) -> dict:
    """Config back to plain dicts (saved inside checkpoints)."""
    #properties are not included since theyre computed
    return copy.deepcopy(asdict(cfg))
