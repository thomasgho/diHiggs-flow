"""
diHiggs-flow: Neural density estimation toolkit for di-Higgs production analysis at the LHC.

This package contains modules for:
- Data loading and preprocessing (dataloader)
- Neural normalizing flows implementation (flow)
- Gaussian process components (gaussian_process)
"""

from .dataloader import Dataset, SR, load
from .flow import train
from .gaussian_process import GPdensity

__all__ = ['Dataset', 'SR', 'load', 'train', 'GPdensity']