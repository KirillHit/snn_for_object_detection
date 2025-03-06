"""
Launch scripts

The choice of the startup script is made in the configuration file.
"""

from .interactive import interactive_spin
from .train import train_spin
from .eval import eval_spin
from .activity import activity_test

__all__ = "interactive_spin", "train_spin", "eval_spin", "activity_test"
