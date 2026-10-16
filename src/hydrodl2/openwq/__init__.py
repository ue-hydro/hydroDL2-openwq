"""OpenWQ water-quality coupling for hydroDL2 physics models (forward coupling).

The hydrology (differentiable HBV) runs in PyTorch exactly as before; the
coupler receives the storages and fluxes of every daily step through the
``wq_hook`` of the physics model and advances the OpenWQ state (C++) with them.
"""
from .coupler import OpenWQHbvCoupler, import_openwq

__all__ = ['OpenWQHbvCoupler', 'import_openwq', 'host']
