"""GraphSSL: a modular Graph Machine Learning library in pure PyTorch."""

# ── Core abstractions ─────────────────────────────────────────────────────────
# ── Augmentation ──────────────────────────────────────────────────────────────
from graphssl.augmentation import MultiView, compose
from graphssl.augmentation.transforms import (
    EdgeAdd,
    EdgeDrop,
    FeatMask,
    FeatNoise,
    FeatShuffle,
    Subgraph,
)
from graphssl.core.augmentation import BaseAugmentation
from graphssl.core.callback import Callback
from graphssl.core.encoder import BaseEncoder
from graphssl.core.model import BaseModel, BaseSSLModel
from graphssl.core.registry import Registry

# ── Data ──────────────────────────────────────────────────────────────────────
from graphssl.data import DataModule

# ── Encoders ──────────────────────────────────────────────────────────────────
from graphssl.encoders import GCNEncoder, GINEncoder, TransformerEncoder

# ── Evaluation ────────────────────────────────────────────────────────────────
from graphssl.evaluation import KNNEvaluator, LogRegEvaluator, extract_embeddings

# ── Losses ────────────────────────────────────────────────────────────────────
from graphssl.losses import (
    BarlowTwinsLoss,
    CombinedLoss,
    CosineRegressionLoss,
    DINOLoss,
    NTXentLoss,
    VICRegLoss,
)

# ── Models ────────────────────────────────────────────────────────────────────
from graphssl.models import (
    AFGRL,
    BGRL,
    DGI,
    BarlowTwins,
    GraphCL,
    GraphDINO,
    Supervised,
    VICReg,
)

# ── Neural network building blocks ────────────────────────────────────────────
from graphssl.nn import MLP, DINOHead, Predictor, Projector, pool_graph_embeddings

# ── Training ──────────────────────────────────────────────────────────────────
from graphssl.training import (
    DINOTrainer,
    EmbeddingLoggerCallback,
    LinearEvalCallback,
    VisualizationCallback,
)

# ── Utilities ─────────────────────────────────────────────────────────────────
from graphssl.utils import CosineDecayScheduler, CosineEMAScheduler, update_ema_params
