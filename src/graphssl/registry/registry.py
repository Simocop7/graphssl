from graphssl.core.registry import Registry

ENCODERS: Registry = Registry()
HEADS: Registry = Registry()
AUGMENTS: Registry = Registry()
OBJECTIVES: Registry = Registry()  # register custom SSL objectives
DATASETS: Registry = Registry()    # register custom dataset builders
LOADERS: Registry = Registry()
LOSSES: Registry = Registry()
