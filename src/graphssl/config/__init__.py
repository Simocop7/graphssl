from .schema import (
    EncoderConfig,
    HeadConfig,
    AugmentConfig,
    DGIConfig,
    GraphCLConfig,
    VICRegConfig,
    BarlowTwinsConfig,
    BGRLConfig,
    AFGRLConfig,
    SupervisedConfig,
    GraphDINOConfig,
)
from .load import LOADERS, load_config, build_model
