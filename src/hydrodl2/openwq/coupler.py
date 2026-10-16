"""OpenWQ coupler for the differentiable HBV (Hbv_1_1p) of hydroDL2.

Forward (one-way) coupling: the HBV water balance drives the OpenWQ solute
balance. Nothing of the hydrology is changed, the hook only reads tensors.

Mapping of HBV 1.1p to OpenWQ compartments (one OpenWQ cell per basin ``ix``
and per HBV instance ``iz`` of the ``nmul`` ensemble; each instance is a share
``w`` of the basin area so that mm x area x w gives m3):

======================  ===========  ==================================
HBV storage / stage     OpenWQ cmp   water transfers carried with solute
======================  ===========  ==================================
SNOWPACK                SNOWPACK     <- PRECIP (snow); -> MELTWATER (melt)
MELTWATER               MELTWATER    -> SNOWPACK (refreezing); -> SOIL (tosoil)
SM (soil moisture)      SOIL         <- PRECIP (rain); -> UPPER_ZONE (recharge+excess)
SUZ (upper zone)        UPPER_ZONE   -> LOWER_ZONE (PERC); -> routing (Q0+Q1)
SLZ (lower zone)        LOWER_ZONE   -> SOIL (capillary); -> routing (Q2)
unit-hydrograph lags    ROUTING      shift register, nz = routing_len - 1 lags
basin outlet            STREAM       -> out of the domain every step
======================  ===========  ==================================

Evapotranspiration removes water only (solute stays in SOIL), as for the other
OpenWQ host models. The gamma unit hydrograph of HBV is reproduced exactly as a
shift register: the runoff of a step is split over the lag cells with the UH
weights (the first weight goes straight to STREAM), and every step each lag cell
moves to the next one. The water that reaches STREAM in a step equals the routed
streamflow of HBV (checked in tests/test_openwq_coupler.py). The routing restart
carry-over of hydroDL2 (``routing_state``) is not represented: the lag cells
start empty, as ``uh_conv`` assumes when no carry-over is given.

OpenWQ semantics reminder: every transfer of a host step is computed from the
state at the start of the step and applied at the end of the step, so the
source water volume passed with each transfer is the volume available to that
storage during the step (start storage + inflows of the step). OpenWQ's first
interaction step only seeds the initial conditions and ignores every flux (its
time step is not known yet), so the coupler primes OpenWQ with one step dated
one host step before the first HBV day, with the initial storages: the output
then holds one record of the initial state before the first simulated day.

Usage::

    from hydrodl2.openwq import OpenWQHbvCoupler
    coupler = OpenWQHbvCoupler(case_dir, area_m2, start_date, basin_ids=gage_ids)
    phy_model.wq_hook = coupler        # Hbv_1_1p instance (dMG: model.model_dict[name].phy_model)
    out = phy_model(x_dict, parameters)  # or the dMG ModelHandler forward
    coupler.stream_conc[t, b]           # outlet concentration (g/m3 = mg/L) of species 0
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

import numpy as np
import torch

from hydrodl2.core.calc import uh_gamma


def import_openwq():
    """Return the Python host side of OpenWQ (hydrodl2.openwq.host), which loads
    the compiled hydrolink module ``openwq_py`` from ``<repo>/openwq`` (or
    ``$OPENWQ_HYDROLINK_DIR``)."""
    from hydrodl2.openwq import host
    return host


COMPARTMENTS = ['SNOWPACK', 'MELTWATER', 'SOIL', 'UPPER_ZONE', 'LOWER_ZONE', 'ROUTING', 'STREAM']
DEPENDS = ['SM', 'Tair_K', 'Tsoil_K', 'cellArea_m2', 'T']


def _np(x):
    """torch tensor -> float64 numpy (no grad)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


class OpenWQHbvCoupler:
    """Hook object for ``Hbv_1_1p.wq_hook`` driving an OpenWQ instance."""

    def __init__(self, case_dir, area_m2, start_date, basin_ids=None,
                 step_hours=24.0, master_json='openWQ_master.json',
                 species=0, record=True, verbose=False):
        """
        case_dir    folder with the OpenWQ master JSON (paths inside it are relative to it)
        area_m2     basin areas [ngrid] in m2
        start_date  datetime (or (y, m, d)) of the first step seen by the hook: the
                    first day of the physics loop (after the warm-up period when
                    ``warmup_states`` is True, else the first day of the input)
        basin_ids   optional ids [ngrid] used as OpenWQ cell ids (default: index)
        step_hours  host time step (HBV is daily)
        species     index of the species whose outlet concentration is recorded
        """
        self.openwq = import_openwq()
        self.case_dir = Path(case_dir)
        self.master_json = master_json
        self.area_m2 = np.asarray(area_m2, dtype=np.float64).reshape(-1)
        if not isinstance(start_date, _dt.datetime):
            start_date = _dt.datetime(*start_date)
        self.clock = self.openwq.Clock(start_date.year, start_date.month, start_date.day,
                                       start_date.hour, start_date.minute, step_hours=step_hours)
        self.basin_ids = None if basin_ids is None else [str(b) for b in basin_ids]
        self.species = int(species)
        self.record = record
        self.verbose = verbose
        self.host = None
        self._prev_cwd = None
        self.nsteps = self.ngrid = self.nmul = 0

    # ------------------------------------------------------------------
    # hook protocol
    # ------------------------------------------------------------------
    def begin(self, model, nsteps, ngrid):
        if self.area_m2.size != ngrid:
            raise ValueError("area_m2 has %d basins, the model batch has %d" % (self.area_m2.size, ngrid))
        if self.basin_ids is not None and len(self.basin_ids) != ngrid:
            raise ValueError("basin_ids has %d entries, the model batch has %d" % (len(self.basin_ids), ngrid))
        self.nsteps, self.ngrid, self.nmul = int(nsteps), int(ngrid), int(model.nmul)
        self.routing = bool(model.routing)
        if self.routing and getattr(model, 'comprout', False):
            raise NotImplementedError("comprout routing (per-instance UH) is not supported by the coupler")
        if self.routing and getattr(model, 'routing_state', None) is not None:
            raise NotImplementedError("routing restart carry-over (routing_state) is not supported by the coupler")

        # instance weights -> share of the basin area per HBV instance [ngrid, nmul]
        muwts = getattr(model, 'muwts', None)
        if muwts is None:
            w = np.full((ngrid, self.nmul), 1.0 / self.nmul)
        else:
            w = _np(muwts)
            w = np.broadcast_to(w.reshape(-1, self.nmul) if w.ndim > 1 else w.reshape(1, -1), (ngrid, self.nmul)).copy()
        self.w = w
        self.f = self.area_m2[:, None] * w * 1e-3          # mm -> m3 per (basin, instance)
        self.f_basin = self.area_m2 * 1e-3                   # mm -> m3 per basin

        # unit hydrograph weights [L, ngrid] (same function and arguments as Hbv_1_1p._PBM)
        if self.routing:
            L = int(model.routing_len)
            a = model.routing_param_dict['route_a'].repeat(nsteps, 1).unsqueeze(-1)
            b = model.routing_param_dict['route_b'].repeat(nsteps, 1).unsqueeze(-1)
            self.UH = _np(uh_gamma(a, b, lenF=L))[:, :, 0]   # [lenF, ngrid]
            self.L = self.UH.shape[0]
        else:
            self.L = 1
            self.UH = np.ones((1, ngrid))
        self.nlag = self.L - 1

        # OpenWQ host
        self._prev_cwd = os.getcwd()
        os.chdir(self.case_dir)
        host = self.openwq.Host(self.master_json)
        self.i = {}
        for name in COMPARTMENTS:
            if name == 'ROUTING':
                if self.nlag == 0:
                    continue
                self.i[name] = host.add_compartment(name, nx=ngrid, ny=1, nz=self.nlag)
            elif name == 'STREAM':
                self.i[name] = host.add_compartment(name, nx=ngrid, ny=1, nz=1)
            else:
                self.i[name] = host.add_compartment(name, nx=ngrid, ny=1, nz=self.nmul)
        host.add_ext_flux('PRECIP', nx=ngrid, ny=1, nz=self.nmul)
        self.idep = {name: host.add_depend(name, nx=ngrid, ny=1, nz=self.nmul) for name in DEPENDS}
        host.set_cell_label('basin')
        if self.basin_ids is not None:
            for name, icmp in self.i.items():
                nz = host.compartment_shape(icmp)[2]
                ids = []
                for b in range(ngrid):
                    for z in range(nz):
                        ids.append(self.basin_ids[b] if nz == 1 else '%s_z%d' % (self.basin_ids[b], z + 1))
                host.set_cell_ids(icmp, ids)
        host.init()
        self.host = host
        self.ichem = self.species
        if self.ichem >= host.num_species():
            raise ValueError("species index %d but the configuration has %d species" % (self.ichem, host.num_species()))

        # routing water held in the lag cells and in the outlet [m3]
        self.lag_vol = np.zeros((ngrid, max(self.nlag, 1)))
        self.stream_vol = np.zeros(ngrid)
        self._primed = False
        if self.record:
            self.stream_mass = np.zeros((nsteps, ngrid))     # mass arriving at the outlet in the step [g]
            self.stream_flow_m3 = np.zeros((nsteps, ngrid))  # water arriving at the outlet in the step [m3]
            self.stream_conc = np.zeros((nsteps, ngrid))     # concentration of the outflow [g/m3 = mg/L]
            self.mass_out = np.zeros((nsteps, ngrid))        # mass leaving the domain in the step [g]
            self.mass_in = np.zeros((nsteps, ngrid))         # precipitation water [m3] (mass = x inflow conc)
        if self.verbose:
            print("<OpenWQ-dHBV> %d basins x %d instances, %d steps, routing=%s (lags %d)"
                  % (ngrid, self.nmul, nsteps, self.routing, self.nlag))

    def step(self, t, s):
        host, i, ichem = self.host, self.i, self.ichem
        ng, nm, f = self.ngrid, self.nmul, self.f
        y, mo, d, h, mi = self.clock.at(t)
        v = {k: _np(s[k]) for k in s}  # [ngrid, nmul] arrays

        # start-of-step water volumes [m3]
        host.set_water_volume_array(i['SNOWPACK'], v['SNOWPACK0'] * f)
        host.set_water_volume_array(i['MELTWATER'], v['MELTWATER0'] * f)
        host.set_water_volume_array(i['SOIL'], v['SM0'] * f)
        host.set_water_volume_array(i['UPPER_ZONE'], v['SUZ0'] * f)
        host.set_water_volume_array(i['LOWER_ZONE'], v['SLZ0'] * f)
        if 'ROUTING' in i:
            host.set_water_volume_array(i['ROUTING'], self.lag_vol)
        host.set_water_volume_array(i['STREAM'], self.stream_vol)

        if not self._primed:
            # OpenWQ's first interaction step: initial conditions only (see module docstring)
            py, pmo, pd, ph, pmi = self.clock.at(t - 1)
            host.time_start(py, pmo, pd, ph, pmi)
            host.time_end(py, pmo, pd, ph, pmi)
            self._primed = True

        # dependency variables for the biogeochemistry
        tair_k = v['T'] + 273.15
        host.set_depend_array(self.idep['SM'], np.clip(v['SM0'] / np.maximum(v['FC'], 1e-12), 0.0, 1.0))
        host.set_depend_array(self.idep['Tair_K'], tair_k)
        host.set_depend_array(self.idep['Tsoil_K'], tair_k)
        host.set_depend_array(self.idep['cellArea_m2'], self.area_m2[:, None] * self.w)
        host.set_depend_array(self.idep['T'], v['T'])

        host.time_start(y, mo, d, h, mi)

        if self.record:
            # the whole outlet content of the previous step leaves the domain now
            self.mass_out[t] = host.mass_array(i['STREAM'], ichem)[:, 0, 0]

        # water available to each storage during the step [m3] (start + inflows)
        snow_in = v['SNOW'] * f
        rain_in = v['RAIN'] * f
        melt = v['melt'] * f
        refr = v['refreezing'] * f
        tosoil = v['tosoil'] * f
        rech = (v['recharge'] + v['excess']) * f
        capil = v['capillary'] * f
        perc = v['PERC'] * f
        q01 = (v['Q0'] + v['Q1']) * f
        q2 = v['Q2'] * f
        avail_snow = v['SNOWPACK0'] * f + snow_in + refr
        avail_melt = v['MELTWATER0'] * f + melt
        avail_soil = v['SM0'] * f + rain_in + tosoil + capil
        avail_upper = v['SUZ0'] * f + rech
        avail_lower = v['SLZ0'] * f + perc
        UH = self.UH

        move = host.move
        inflow = host.inflow
        for b in range(ng):
            for k in range(nm):
                if snow_in[b, k] > 0.0:
                    inflow('PRECIP', i['SNOWPACK'], b, 0, k, snow_in[b, k], y, mo, d, h, mi)
                if rain_in[b, k] > 0.0:
                    inflow('PRECIP', i['SOIL'], b, 0, k, rain_in[b, k], y, mo, d, h, mi)
                if melt[b, k] > 0.0:
                    move(i['SNOWPACK'], b, 0, k, i['MELTWATER'], b, 0, k, melt[b, k], avail_snow[b, k], y, mo, d, h, mi)
                if refr[b, k] > 0.0:
                    move(i['MELTWATER'], b, 0, k, i['SNOWPACK'], b, 0, k, refr[b, k], avail_melt[b, k], y, mo, d, h, mi)
                if tosoil[b, k] > 0.0:
                    move(i['MELTWATER'], b, 0, k, i['SOIL'], b, 0, k, tosoil[b, k], avail_melt[b, k], y, mo, d, h, mi)
                if rech[b, k] > 0.0:
                    move(i['SOIL'], b, 0, k, i['UPPER_ZONE'], b, 0, k, rech[b, k], avail_soil[b, k], y, mo, d, h, mi)
                if capil[b, k] > 0.0:
                    move(i['LOWER_ZONE'], b, 0, k, i['SOIL'], b, 0, k, capil[b, k], avail_lower[b, k], y, mo, d, h, mi)
                if perc[b, k] > 0.0:
                    move(i['UPPER_ZONE'], b, 0, k, i['LOWER_ZONE'], b, 0, k, perc[b, k], avail_upper[b, k], y, mo, d, h, mi)
                # runoff of the instance -> outlet (lag 0) and lag cells (lags 1..L-1)
                for src, q, avail in ((i['UPPER_ZONE'], q01[b, k], avail_upper[b, k]),
                                      (i['LOWER_ZONE'], q2[b, k], avail_lower[b, k])):
                    if q <= 0.0:
                        continue
                    for l in range(self.L):
                        ql = q * UH[l, b]
                        if ql <= 0.0:
                            continue
                        if l == 0:
                            move(src, b, 0, k, i['STREAM'], b, 0, 0, ql, avail, y, mo, d, h, mi)
                        else:
                            move(src, b, 0, k, i['ROUTING'], b, 0, l - 1, ql, avail, y, mo, d, h, mi)
            # shift register of the lag cells and outflow of the outlet
            for j in range(self.nlag):
                vol = self.lag_vol[b, j]
                if vol <= 0.0:
                    continue
                if j == 0:
                    move(i['ROUTING'], b, 0, 0, i['STREAM'], b, 0, 0, vol, vol, y, mo, d, h, mi)
                else:
                    move(i['ROUTING'], b, 0, j, i['ROUTING'], b, 0, j - 1, vol, vol, y, mo, d, h, mi)
            if self.stream_vol[b] > 0.0:
                move(i['STREAM'], b, 0, 0, self.openwq.LOSS, 0, 0, 0, self.stream_vol[b], self.stream_vol[b], y, mo, d, h, mi)

        # water balance of the routing cells and of the outlet for the end of the step
        q_tot = (q01 + q2).sum(axis=1)                           # [ngrid] runoff produced in the step
        new_lag = np.zeros_like(self.lag_vol)
        if self.nlag > 0:
            new_lag[:, :-1] = self.lag_vol[:, 1:]
            new_lag += (q_tot[:, None] * UH[1:, :].T)
        new_stream = (self.lag_vol[:, 0] if self.nlag > 0 else 0.0) + q_tot * UH[0, :]
        self.lag_vol, self.stream_vol = new_lag, new_stream

        # end-of-step volumes so that the output concentrations are consistent
        host.set_water_volume_array(i['SNOWPACK'], v['SNOWPACK'] * f)
        host.set_water_volume_array(i['MELTWATER'], v['MELTWATER'] * f)
        host.set_water_volume_array(i['SOIL'], v['SM'] * f)
        host.set_water_volume_array(i['UPPER_ZONE'], v['SUZ'] * f)
        host.set_water_volume_array(i['LOWER_ZONE'], v['SLZ'] * f)
        if 'ROUTING' in i:
            host.set_water_volume_array(i['ROUTING'], self.lag_vol)
        host.set_water_volume_array(i['STREAM'], self.stream_vol)

        host.time_end(y, mo, d, h, mi)

        if self.record:
            m = host.mass_array(i['STREAM'], ichem)[:, 0, 0]
            self.stream_mass[t] = m
            self.stream_flow_m3[t] = self.stream_vol
            with np.errstate(divide='ignore', invalid='ignore'):
                self.stream_conc[t] = np.where(self.stream_vol > 0, m / self.stream_vol, np.nan)
            self.mass_in[t] = (snow_in + rain_in).sum(axis=1)

    def end(self, model, Qs):
        self.model_streamflow_mm = _np(Qs)[:, :, 0] if Qs is not None else None
        if self._prev_cwd is not None:
            os.chdir(self._prev_cwd)
            self._prev_cwd = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def total_mass(self):
        """Mass of the recorded species still inside the domain [g]."""
        return self.host.total_mass(self.ichem)

    @staticmethod
    def write_tracer_config(case_dir, nmul=None, routing=True, species=('TRACER',),
                            precip_mg_l=None, initial_mg_l=None, decay_per_day=0.0, **kw):
        """Write a conservative-tracer OpenWQ configuration for this coupler
        (all compartments, PRECIP inflow) with openwq.write_minimal_config."""
        openwq = import_openwq()
        comps = [c for c in COMPARTMENTS if routing or c != 'ROUTING']
        return openwq.write_minimal_config(
            case_dir, comps, species=species,
            initial_mg_l=initial_mg_l or {},
            ewf_names=('PRECIP',),
            ewf_mg_l={'PRECIP': precip_mg_l or {}},
            timestep=(1, 'day'), decay_per_day=decay_per_day,
            project='dHBV-OpenWQ', **kw)
