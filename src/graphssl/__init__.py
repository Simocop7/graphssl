"""GraphSSL: a modular Graph Machine Learning library in pure PyTorch."""

# ── Core abstractions ─────────────────────────────────────────────────────────
from graphssl.core.model import BaseModel, BaseSSLModel
from graphssl.core.encoder import BaseEncoder
from graphssl.core.augmentation import BaseAugmentation
from graphssl.core.callback import Callback
from graphssl.core.registry import Registry

# ── Encoders ──────────────────────────────────────────────────────────────────
from graphssl.encoders import GINEncoder, GCNEncoder, TransformerEncoder

# ── Models ────────────────────────────────────────────────────────────────────
from graphssl.models import (
    GraphDINO,
    Supervised,
    DGI,
    GraphCL,
    BGRL,
    AFGRL,
    VICReg,
    BarlowTwins,
)

# ── Losses ────────────────────────────────────────────────────────────────────
from graphssl.losses import (
    DINOLoss,
    NTXentLoss,
    VICRegLoss,
    BarlowTwinsLoss,
    CosineRegressionLoss,
    CombinedLoss,
)

# ── Neural network building blocks ────────────────────────────────────────────
from graphssl.nn import MLP, Projector, Predictor, DINOHead
from graphssl.nn import pool_graph_embeddings

# ── Augmentation ──────────────────────────────────────────────────────────────
from graphssl.augmentation import compose, MultiView
from graphssl.augmentation.transforms import (
    EdgeDrop,
    EdgeAdd,
    Subgraph,
    FeatMask,
    FeatNoise,
    FeatShuffle,
)

# ── Utilities ─────────────────────────────────────────────────────────────────
from graphssl.utils import update_ema_params, CosineDecayScheduler, CosineEMAScheduler

# ── Data ──────────────────────────────────────────────────────────────────────
from graphssl.data import DataModule

# ── Evaluation ────────────────────────────────────────────────────────────────
from graphssl.evaluation import LogRegEvaluator, KNNEvaluator, extract_embeddings

# ── Training ──────────────────────────────────────────────────────────────────
from graphssl.training import (
    DINOTrainer,
    EmbeddingLoggerCallback,
    LinearEvalCallback,
    VisualizationCallback,
)
