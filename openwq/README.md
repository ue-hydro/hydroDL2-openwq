# OpenWQ hydrolink for the differentiable HBV (Python host)

Same layout as the other OpenWQ host models (SUMMA, mizuRoute, CRHM, FLUXOS):

```
openwq/                      this folder: the hydrolink files of the host
  OpenWQ_hydrolink.h/.cpp    pybind11 host class (compartments, time loop calls) and module
  openwq_py*.so              compiled module (written here by the build, not committed)
  openwq/                    clone of the OpenWQ source tree (ue-hydro/openwq, not committed)
```

The Python side lives in the hydrodl2 package: `hydrodl2.openwq.host` loads the
module and adds the helpers (`Clock`, `case_dir`, `write_minimal_config`);
`hydrodl2.openwq.OpenWQHbvCoupler` is the hook of `Hbv_1_1p` that drives OpenWQ.

## Build

```bash
git clone --branch version_2_development_add_ML https://github.com/ue-hydro/openwq.git openwq/openwq
# inside the OpenWQ container, with a Python environment that has pybind11 and numpy:
cd openwq/openwq && mkdir -p work_cmake_py && cd work_cmake_py
cmake -DHOST_MODEL_TARGET=python_openwq -DCMAKE_BUILD_TYPE=Release \
      -Dpybind11_DIR=$(python -m pybind11 --cmakedir) \
      -DPYTHON_EXECUTABLE=$(which python) \
      -DSUNDIALS_DIR=/usr/local/sundials/lib/cmake/sundials \
      -DPhreeqcRM_DIR=<shared PhreeqcRM>/lib/cmake/PhreeqcRM ..
make -j 4 openwq_py
```

PhreeqcRM must be a shared (or PIC) library because the module is a shared
object; the static system PhreeqcRM of the container is not. The module
`openwq_py.<abi>.so` is written to this folder.

## API (0-based indices, m3 and m3 per host step)

```python
from hydrodl2.openwq import host as openwq
with openwq.case_dir(case):                 # folder with openWQ_master.json
    host = openwq.Host("openWQ_master.json")
    soil = host.add_compartment("SOIL", nx=n_basins, ny=1, nz=1)   # index = order
    host.add_ext_flux("PRECIP", nx=n_basins)
    host.add_depend("Tair_K", nx=n_basins)
    host.set_cell_label("basin"); host.set_cell_ids(soil, ["01013500", ...])
    host.init()                                                    # InitialConfig + ParseEWFandSS
    for t in range(n_steps):
        y, m, d, h, mi = clock.at(t)
        host.set_water_volume_array(soil, vol_m3[t])                # state at start of step
        host.time_start(y, m, d, h, mi)                             # RunTimeLoopStart
        host.inflow("PRECIP", soil, ix, 0, 0, rain_m3, y, m, d, h, mi)      # RunSpaceStep_IN
        host.move(soil, ix, 0, 0, other, ix, 0, 0, flux_m3, vol_src_m3, y, m, d, h, mi)  # RunSpaceStep
        host.move(soil, ix, 0, 0, openwq.LOSS, 0, 0, 0, q_m3, vol_src_m3, y, m, d, h, mi)
        host.time_end(y, m, d, h, mi)                               # RunTimeLoopEnd (output)
    host.mass_array(soil, ichem); host.total_mass(ichem)
```

OpenWQ's first interaction step only seeds the initial conditions and ignores
every flux (its time step is not known yet): prime it with one step before the
real loop, as `OpenWQHbvCoupler` does.

## Tests

```bash
python -m pytest tests/test_openwq_host.py tests/test_openwq_coupler.py -q
```
