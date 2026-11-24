"""Forecast dense visual features with Fourier-World."""

from .model import FourierWorld
from .timed_model import TimedFourierWorld

__all__ = ["FourierWorld", "TimedFourierWorld"]
