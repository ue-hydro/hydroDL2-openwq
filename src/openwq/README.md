# OpenWQ coupling of the differentiable HBV (`src/openwq`)

Same layout as every other OpenWQ host model (SUMMA, mizuRoute, CRHM, FLUXOS):
one folder with the hydrolink and the supporting coupler files, and the OpenWQ
source tree cloned inside it.

```
src/openwq/
  OpenWQ_hydrolink.h/.cpp    hydrolink: pybind11 host class (compartments, time-loop calls) + module
  host.py                    Python side of the hydrolink: module loader, Clock, case_dir, config writer
  coupler.py                 OpenWQHbvCoupler: the wq_hook of Hbv_1_1p that drives OpenWQ
  run_dhbv_openwq.py         the dMG example (pretrained dHBV 1.1p, CAMELS) with OpenWQ attached
  tests/                     unit tests (hydrolink tracer closure, coupler)
  openwq/                    clone of the OpenWQ source tree (not committed)
  openwq_py*.so, _deps/      built module and dependency builds (not committed)
```

## Build (inside the OpenWQ container, Python environment with pybind11 and numpy active)

```bash
cd src/openwq
git clone --branch version_2_development_add_ML https://github.com/ue-hydro/openwq.git openwq
cd openwq
cmake -DHOST_MODEL_TARGET=python_openwq -DCMAKE_BUILD_TYPE=Release .
make -j 4
```

The target finds the active Python and its pybind11 by itself, and writes the
module `openwq_py.<abi>.so` to `src/openwq/`. A Python module is a shared
object and cannot link the usual static PhreeqcRM, so the target builds a shared
PhreeqcRM once into `src/openwq/_deps/phreeqcrm-install` (from
`-DPHREEQCRM_SOURCE_DIR=...`, else `/opt/phreeqcrm/phreeqcrm-master` of the
container, else a clone of usgs-coupled/phreeqcrm); pass `-DPhreeqcRM_DIR=...`
to use another shared PhreeqcRM instead.

## Use

```python
import openwq                               # src/ is importable (pip install -e . or sys.path)
with openwq.case_dir(case):                 # folder with openWQ_master.json
    host = openwq.Host("openWQ_master.json")
    soil = host.add_compartment("SOIL", nx=n_basins, ny=1, nz=1)   # index = order of calls
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

With the differentiable HBV:

```python
from openwq import OpenWQHbvCoupler
coupler = OpenWQHbvCoupler(case_dir, area_m2, start_date, basin_ids=gage_ids)
phy_model.wq_hook = coupler            # dMG: model.model_dict['Hbv_1_1p'].phy_model
out = model(dataset_sample, eval=True)
coupler.stream_conc                    # outlet concentration [time, basin] (mg/L)
```

Indices are 0-based, volumes in m3 and fluxes in m3 per host step. OpenWQ's
first interaction step only seeds the initial conditions and ignores every flux
(its time step is not known yet): prime it with one step before the real loop,
as `OpenWQHbvCoupler` does. Water-quality parameters are calibrated afterwards
with the OpenWQ calibration library with the hydrology fixed, as for the other
hosts.

## Tests

```bash
python -m pytest src/openwq/tests -q
```
