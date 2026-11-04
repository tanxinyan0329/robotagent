"""Importing this package registers every scene environment and both robot models."""
from . import kitchen  # noqa: F401
from .tabletop import MobilePandaTabletop  # noqa: F401

SCENES = {"kitchen": "MobilePandaKitchen-v1", "tabletop": "MobilePandaTabletop-v1"}
