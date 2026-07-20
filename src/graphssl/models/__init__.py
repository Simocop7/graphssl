# Side-effect imports register all encoders, heads, and augmentations.
from graphssl.encoders import GINEncoder, GCNEncoder, TransformerEncoder   # noqa: F401
from graphssl.nn import DINOHead                                            # noqa: F401
from graphssl.augmentation import functional                                # noqa: F401

from graphssl.core.model import BaseModel, BaseSSLModel

from .graphdino import GraphDINO
from .supervised import Supervised
from .dgi import DGI
from .graphcl import GraphCL
from .bgrl import BGRL
from .afgrl import AFGRL
from .vicreg import VICReg
from .barlow_twins import BarlowTwins
