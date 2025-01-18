"""
SODa Model
"""

from .soda import SODa
from .generator import BackboneGen, NeckGen, Head, BaseConfig
from typing import Dict
from .yolo import Yolo

config_list: Dict[str, BaseConfig] = {"yolo": Yolo}

__all__ = "SODa", "BackboneGen", "NeckGen", "Head", "BaseConfig"
