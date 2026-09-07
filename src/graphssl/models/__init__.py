# Side-effect imports register all encoders, heads, and augmentations.
from graphssl.augmentation import functional  # noqa: F401
from graphssl.core.model import BaseModel, BaseSSLModel
from graphssl.encoders import GCNEncoder, GINEncoder, TransformerEncoder  # noqa: F401
from graphssl.nn import DINOHead  # noqa: F401

from .afgrl import AFGRL
from .barlow_twins import BarlowTwins
from .bgrl import BGRL
from .dgi import DGI
from .graphcl import GraphCL
from .graphdino import GraphDINO
from .supervised import Supervised
from .vicreg import VICReg
