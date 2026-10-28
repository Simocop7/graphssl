"""GraphSSL: a modular Graph Machine Learning library in pure PyTorch."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

try:
    __version__ = _installed_version("graphssl")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0+unknown"

# ── Core abstractions ─────────────────────────────────────────────────────────
# ── Augmentation ──────────────────────────────────────────────────────────────
from graphssl.augmentation import MultiView, compose
from graphssl.augmentation.transforms import (
    AttrMask,
    EdgeAdd,
    EdgeDrop,
    FeatMask,
    FeatNoise,
    FeatShuffle,
    NodeDrop,
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
from graphssl.evaluation import (
    KNNEvaluator,
    LogRegEvaluator,
    RidgeEvaluator,
    effective_rank,
    extract_embeddings,
)

# ── Losses ────────────────────────────────────────────────────────────────────
from graphssl.losses import (
    BarlowTwinsLoss,
    CombinedLoss,
    CosineRegressionLoss,
    DINOLoss,
    NTXentLoss,
    SCELoss,
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
    GraphMAE,
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
