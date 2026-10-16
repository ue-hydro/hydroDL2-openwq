"""OpenWQ water-quality coupling of hydroDL2 (forward coupling).

This folder follows the layout of every OpenWQ host model:

* ``OpenWQ_hydrolink.h/.cpp``: the hydrolink (pybind11 host class + module ``openwq_py``),
* ``openwq/``: clone of the OpenWQ source tree, built with its CMake target ``python_openwq``,
* ``host.py``: Python side of the hydrolink (module loader, clock, case folder, config writer),
* ``coupler.py``: ``OpenWQHbvCoupler``, the ``wq_hook`` of ``Hbv_1_1p`` that drives OpenWQ,
* ``run_dhbv_openwq.py``: the dMG example with OpenWQ attached; ``tests/``: unit tests.

The hydrology (differentiable HBV) runs in PyTorch exactly as before; the hook only
reads tensors and advances the OpenWQ state (C++) for every daily step.
"""
from .host import LOSS, Clock, Host, case_dir, write_minimal_config
from .coupler import OpenWQHbvCoupler

__all__ = ['Host', 'LOSS', 'Clock', 'case_dir', 'write_minimal_config', 'OpenWQHbvCoupler']
