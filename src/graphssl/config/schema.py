"""Validated configuration dataclasses for all model components.

Each dataclass mirrors the corresponding YAML block and raises ValueError at
construction time if a value is out of range, so misconfigured runs fail at
load time rather than mid-training.

Usage (config-driven construction, uniform across all models)::

    from graphssl.models import BGRL
    model = BGRL(config=yaml_config["model"], in_channels=dataset.num_features)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch.nn as nn

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


@dataclass
class EncoderConfig:
    """Validated configuration for graph encoders.

    All three encoder backbones (GCN, GIN, Transformer) share this config.
    Fields that don't apply to a given encoder are silently ignored during
    build() via introspection — no extra per-encoder config class needed.
    """

    name: str
    hidden_dim: int
    num_layers: int
    mlp_ratio: float = 2.0
    drop: float = 0.2
    pool: bool = True
    norm_type: str = "batch"  # 'batch', 'layer', or 'none'
    edge_dim: Optional[int] = None  # enables edge-feature-aware conv (GINEConv / TransformerConv)
    node_emb_num_classes: Optional[int] = (
        None  # categorical node features (e.g. ZINC: 28 atom types)
    )
    edge_emb_num_classes: Optional[int] = (
        None  # categorical edge features (e.g. ZINC: 4 bond types)
    )

    def __post_init__(self):
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if not (0.0 <= self.drop < 1.0):
            raise ValueError(f"drop must be in [0, 1), got {self.drop}")
        if self.norm_type not in ("batch", "layer", "none"):
            raise ValueError(
                f"norm_type must be 'batch', 'layer', or 'none', got {self.norm_type!r}"
            )
        if self.edge_dim is not None and self.edge_dim <= 0:
            raise ValueError(f"edge_dim must be > 0, got {self.edge_dim}")
        if self.node_emb_num_classes is not None and self.node_emb_num_classes <= 0:
            raise ValueError(f"node_emb_num_classes must be > 0, got {self.node_emb_num_classes}")
        if self.edge_emb_num_classes is not None and self.edge_emb_num_classes <= 0:
            raise ValueError(f"edge_emb_num_classes must be > 0, got {self.edge_emb_num_classes}")
        if self.edge_emb_num_classes is not None and self.edge_dim is None:
            raise ValueError(
                "edge_emb_num_classes requires edge_dim to be set "
                "(the embedding output dimension must be defined)."
            )

    def build(self, in_channels: int) -> nn.Module:
        """Instantiate the encoder via the ENCODERS registry.

        Only the kwargs accepted by the encoder's __init__ are forwarded,
        so EncoderConfig can carry universal fields (num_layers, drop, …)
        without breaking encoders that don't expose those parameters (e.g. GCN).
        """
        import inspect

        from graphssl.registry import ENCODERS

        cls = ENCODERS.get_builder(self.name)
        valid = set(inspect.signature(cls.__init__).parameters) - {"self"}
        kwargs = {k: v for k, v in vars(self).items() if k != "name" and k in valid}
        return ENCODERS.build(self.name, in_channels=in_channels, **kwargs)


@dataclass
class HeadConfig:
    """Validated configuration for the DINOHead projection head."""

    name: str
    proj_hidden: int
    bottleneck_dim: int
    n_prototypes: int
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    center_momentum: float = 0.9
    warmup_teacher_temp: float = 0.04
    warmup_teacher_temp_epochs: int = 0

    def __post_init__(self):
        for fname, val in [
            ("proj_hidden", self.proj_hidden),
            ("bottleneck_dim", self.bottleneck_dim),
            ("n_prototypes", self.n_prototypes),
        ]:
            if val <= 0:
                raise ValueError(f"{fname} must be > 0, got {val}")
        if self.student_temp <= 0 or self.teacher_temp <= 0:
            raise ValueError("student_temp and teacher_temp must be > 0")
        if self.warmup_teacher_temp <= 0:
            raise ValueError("warmup_teacher_temp must be > 0")
        if self.warmup_teacher_temp > self.teacher_temp:
            raise ValueError(
                f"warmup_teacher_temp ({self.warmup_teacher_temp}) must be "
                f"<= teacher_temp ({self.teacher_temp})"
            )
        if self.warmup_teacher_temp_epochs < 0:
            raise ValueError(
                f"warmup_teacher_temp_epochs must be >= 0, got {self.warmup_teacher_temp_epochs}"
            )
        if not (0.0 <= self.center_momentum < 1.0):
            raise ValueError(f"center_momentum must be in [0, 1), got {self.center_momentum}")


@dataclass
class AugmentConfig:
    """Name and keyword arguments for a single augmentation step."""

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("Augmentation name cannot be empty")

    @classmethod
    def from_dict(cls, d: dict) -> AugmentConfig:
        d = dict(d)
        name = d.pop("name")
        return cls(name=name, kwargs=d)


# ---------------------------------------------------------------------------
# Per-model configs
# ---------------------------------------------------------------------------


@dataclass
class DGIConfig:
    """Validated configuration for DGI."""

    encoder: EncoderConfig
    corruption: str = "shuffle_nodes"
    shuffle_ratio: float = 1.0

    def __post_init__(self):
        if self.corruption not in ("shuffle_nodes", "shuffle_edges"):
            raise ValueError(
                f"corruption must be 'shuffle_nodes' or 'shuffle_edges', got {self.corruption!r}"
            )
        if not (0.0 < self.shuffle_ratio <= 1.0):
            raise ValueError(f"shuffle_ratio must be in (0, 1], got {self.shuffle_ratio}")

    @classmethod
    def from_dict(cls, d: dict) -> DGIConfig:
        return cls(
            encoder=EncoderConfig(**d["encoder"]),
            corruption=d.get("corruption", "shuffle_nodes"),
            shuffle_ratio=d.get("shuffle_ratio", 1.0),
        )


@dataclass
class GraphCLConfig:
    """Validated configuration for GraphCL."""

    encoder: EncoderConfig
    augment: List[AugmentConfig] = field(default_factory=list)
    proj_dim: int = 128
    tau: float = 0.5

    def __post_init__(self):
        if self.proj_dim <= 0:
            raise ValueError(f"proj_dim must be > 0, got {self.proj_dim}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")

    @classmethod
    def from_dict(cls, d: dict) -> GraphCLConfig:
        return cls(
            encoder=EncoderConfig(**d["encoder"]),
            augment=[AugmentConfig.from_dict(a) for a in d.get("augment", [])],
            proj_dim=d.get("proj_dim", 128),
            tau=d.get("tau", 0.5),
        )


@dataclass
class VICRegConfig:
    """Validated configuration for VICReg."""

    encoder: EncoderConfig
    augment: List[AugmentConfig] = field(default_factory=list)
    proj_dim: int = 256
    invariance: float = 25.0
    variance: float = 25.0
    covariance: float = 1.0

    def __post_init__(self):
        if self.proj_dim <= 0:
            raise ValueError(f"proj_dim must be > 0, got {self.proj_dim}")
        for name, val in [
            ("invariance", self.invariance),
            ("variance", self.variance),
            ("covariance", self.covariance),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")

    @classmethod
    def from_dict(cls, d: dict) -> VICRegConfig:
        return cls(
            encoder=EncoderConfig(**d["encoder"]),
            augment=[AugmentConfig.from_dict(a) for a in d.get("augment", [])],
            proj_dim=d.get("proj_dim", 256),
            invariance=d.get("invariance", 25.0),
            variance=d.get("variance", 25.0),
            covariance=d.get("covariance", 1.0),
        )


@dataclass
class BarlowTwinsConfig:
    """Validated configuration for Barlow Twins."""

    encoder: EncoderConfig
    augment: List[AugmentConfig] = field(default_factory=list)
    proj_dim: int = 256
    lambda_param: Optional[float] = None  # defaults to 1/proj_dim inside the model

    def __post_init__(self):
        if self.proj_dim <= 0:
            raise ValueError(f"proj_dim must be > 0, got {self.proj_dim}")
        if self.lambda_param is not None and self.lambda_param < 0:
            raise ValueError(f"lambda_param must be >= 0, got {self.lambda_param}")

    @classmethod
    def from_dict(cls, d: dict) -> BarlowTwinsConfig:
        return cls(
            encoder=EncoderConfig(**d["encoder"]),
            augment=[AugmentConfig.from_dict(a) for a in d.get("augment", [])],
            proj_dim=d.get("proj_dim", 256),
            lambda_param=d.get("lambda_param", None),
        )


@dataclass
class BGRLConfig:
    """Validated configuration for BGRL."""

    encoder: EncoderConfig
    augment: List[AugmentConfig] = field(default_factory=list)
    pred_hidden: int = 512
    ema_tau: float = 0.99
    ema_tau_end: float = 1.0
    total_steps: int = 0

    def __post_init__(self):
        if self.pred_hidden <= 0:
            raise ValueError(f"pred_hidden must be > 0, got {self.pred_hidden}")
        if not (0.0 < self.ema_tau < 1.0):
            raise ValueError(f"ema_tau must be in (0, 1), got {self.ema_tau}")
        if not (0.0 < self.ema_tau_end <= 1.0):
            raise ValueError(f"ema_tau_end must be in (0, 1], got {self.ema_tau_end}")
        if self.ema_tau > self.ema_tau_end:
            raise ValueError(
                f"ema_tau ({self.ema_tau}) must be <= ema_tau_end ({self.ema_tau_end})"
            )
        if self.total_steps < 0:
            raise ValueError(f"total_steps must be >= 0, got {self.total_steps}")

    @classmethod
    def from_dict(cls, d: dict) -> BGRLConfig:
        return cls(
            encoder=EncoderConfig(**d["encoder"]),
            augment=[AugmentConfig.from_dict(a) for a in d.get("augment", [])],
            pred_hidden=d.get("pred_hidden", 512),
            ema_tau=d.get("ema_tau", 0.99),
            ema_tau_end=d.get("ema_tau_end", 1.0),
            total_steps=d.get("total_steps", 0),
        )


@dataclass
class AFGRLConfig:
    """Validated configuration for AFGRL."""

    encoder: EncoderConfig
    pred_hidden: int = 512
    ema_tau: float = 0.99
    ema_tau_end: float = 1.0
    total_steps: int = 0
    topk: int = 5
    num_centroids: int = 50
    num_kmeans: int = 4
    clus_num_iters: int = 20

    def __post_init__(self):
        if self.pred_hidden <= 0:
            raise ValueError(f"pred_hidden must be > 0, got {self.pred_hidden}")
        if not (0.0 < self.ema_tau < 1.0):
            raise ValueError(f"ema_tau must be in (0, 1), got {self.ema_tau}")
        if not (0.0 < self.ema_tau_end <= 1.0):
            raise ValueError(f"ema_tau_end must be in (0, 1], got {self.ema_tau_end}")
        if self.ema_tau > self.ema_tau_end:
            raise ValueError(
                f"ema_tau ({self.ema_tau}) must be <= ema_tau_end ({self.ema_tau_end})"
            )
        if self.total_steps < 0:
            raise ValueError(f"total_steps must be >= 0, got {self.total_steps}")
        if self.topk <= 0:
            raise ValueError(f"topk must be > 0, got {self.topk}")
        if self.num_centroids <= 0:
            raise ValueError(f"num_centroids must be > 0, got {self.num_centroids}")
        if self.num_kmeans <= 0:
            raise ValueError(f"num_kmeans must be > 0, got {self.num_kmeans}")
        if self.clus_num_iters <= 0:
            raise ValueError(f"clus_num_iters must be > 0, got {self.clus_num_iters}")

    @classmethod
    def from_dict(cls, d: dict) -> AFGRLConfig:
        return cls(
            encoder=EncoderConfig(**d["encoder"]),
            pred_hidden=d.get("pred_hidden", 512),
            ema_tau=d.get("ema_tau", 0.99),
            ema_tau_end=d.get("ema_tau_end", 1.0),
            total_steps=d.get("total_steps", 0),
            topk=d.get("topk", 5),
            num_centroids=d.get("num_centroids", 50),
            num_kmeans=d.get("num_kmeans", 4),
            clus_num_iters=d.get("clus_num_iters", 20),
        )


@dataclass
class SupervisedConfig:
    """Validated configuration for the supervised baseline."""

    encoder: EncoderConfig

    @classmethod
    def from_dict(cls, d: dict) -> SupervisedConfig:
        return cls(encoder=EncoderConfig(**d["encoder"]))


@dataclass
class GraphDINOConfig:
    """Validated top-level configuration for GraphDINO."""

    encoder: EncoderConfig
    head: HeadConfig
    augment_teacher: List[AugmentConfig] = field(default_factory=list)
    augment_student: List[AugmentConfig] = field(default_factory=list)
    ema_tau: float = 0.996
    ema_tau_base: float = 0.996
    total_steps: int = 0
    freeze_last_layer_epochs: int = 1
    n_views: int = 2
    n_global_views: int = 2

    def __post_init__(self):
        if not (0.0 < self.ema_tau <= 1.0):
            raise ValueError(f"ema_tau must be in (0, 1], got {self.ema_tau}")
        if not (0.0 < self.ema_tau_base <= 1.0):
            raise ValueError(f"ema_tau_base must be in (0, 1], got {self.ema_tau_base}")
        if self.ema_tau_base > self.ema_tau:
            raise ValueError(
                f"ema_tau_base ({self.ema_tau_base}) must be <= ema_tau ({self.ema_tau})"
            )
        if self.total_steps < 0:
            raise ValueError(f"total_steps must be >= 0, got {self.total_steps}")
        if self.freeze_last_layer_epochs < 0:
            raise ValueError(
                f"freeze_last_layer_epochs must be >= 0, got {self.freeze_last_layer_epochs}"
            )
        if self.n_global_views < 1:
            raise ValueError(f"n_global_views must be >= 1, got {self.n_global_views}")
        if self.n_views < self.n_global_views:
            raise ValueError(
                f"n_views ({self.n_views}) must be >= n_global_views ({self.n_global_views})"
            )

    @classmethod
    def from_dict(cls, d: dict) -> GraphDINOConfig:
        encoder = EncoderConfig(**d["encoder"])
        head = HeadConfig(**d["head"])
        augment_teacher = [AugmentConfig.from_dict(a) for a in d.get("augment_teacher", [])]
        augment_student = [AugmentConfig.from_dict(a) for a in d.get("augment_student", [])]
        return cls(
            encoder=encoder,
            head=head,
            augment_teacher=augment_teacher,
            augment_student=augment_student,
            ema_tau=d.get("ema_tau", 0.996),
            ema_tau_base=d.get("ema_tau_base", d.get("ema_tau", 0.996)),
            total_steps=d.get("total_steps", 0),
            freeze_last_layer_epochs=d.get("freeze_last_layer_epochs", 1),
            n_views=d.get("n_views", 2),
            n_global_views=d.get("n_global_views", 2),
        )
