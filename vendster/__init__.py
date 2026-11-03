"""
Vendster: an adaptable buyer-vendor negotiation benchmark for evaluating the
downstream decision quality of forecasters consumed by LLM agents.

The v2 architecture is exposed via :mod:`vendster.core` (extension interfaces,
registries, config, factory). Higher-level modules (data sources, demand models,
forecasters, negotiation environment, experiment harness) register themselves
into the core registries and are imported on demand rather than eagerly, so that
lightweight uses (and extension authors) need not pull in heavy dependencies.
"""

__version__ = "2.0.0"
__author__ = "Charin Polpanumas"

from . import core

__all__ = ["core"]
