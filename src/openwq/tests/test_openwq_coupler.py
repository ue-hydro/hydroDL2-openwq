"""Tests of the OpenWQ coupling of the differentiable HBV (Hbv_1_1p).

Needs the hydrolink module built in src/openwq (CMake target python_openwq of
the OpenWQ clone in src/openwq/openwq, see src/openwq/README.md).

1. the hook does not change the hydrology (every output identical with and
   without the hook),
2. the shift-register routing of the coupler reproduces the gamma unit
   hydrograph of HBV (water reaching the outlet == routed streamflow),
3. a conservative tracer entering with precipitation closes the mass balance
   (entered == still inside + left through the outlet), with and without routing.

Run:  python -m pytest src/openwq/tests -q   (or run this file)
"""
import datetime
import tempfile

import numpy as np
import torch

from hydrodl2.models.hbv.hbv_1_1p import Hbv_1_1p
from openwq import OpenWQHbvCoupler


def make_model(nmul=2, routing=True):
    config = {
        'warmup': 0, 'warmup_states': True, 'dy_drop': 0.0,
        'dynamic_params': {'Hbv_1_1p': ['parBETA']},
        'routing': routing, 'nmul': nmul, 'nearzero': 1e-5,
    }
    return Hbv_1_1p(config, device=torch.device('cpu'))


def synthetic_forcing(nsteps, ngrid, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(nsteps)
    prcp = rng.gamma(1.5, 4.0, (nsteps, ngrid)) * (rng.random((nsteps, ngrid)) < 0.35)
    tmean = 6.0 + 12.0 * np.sin(2 * np.pi * t / 365.0)[:, None] + rng.normal(0, 3, (nsteps, ngrid))
    pet = np.clip(2.0 + 2.0 * np.sin(2 * np.pi * t / 365.0), 0.2, None)[:, None] * np.ones((1, ngrid))
    x = np.stack([prcp, tmean, pet], -1).astype(np.float32)
    return {'x_phy': torch.tensor(x)}


def random_parameters(model, nsteps, ngrid, seed=1):
    g = torch.Generator().manual_seed(seed)
    n = len(model.parameter_bounds) * model.nmul + (2 if model.routing else 0)
    return torch.randn(nsteps, ngrid, n, generator=g)


def run_pair(tmp, nsteps=120, ngrid=3, nmul=2, routing=True, c_rain=5.0):
    """Run HBV without and with the OpenWQ hook; return outputs and coupler."""
    area_m2 = np.array([50.0, 120.0, 300.0])[:ngrid] * 1e6
    x = synthetic_forcing(nsteps, ngrid)
    ref_model = make_model(nmul, routing)
    params = random_parameters(ref_model, nsteps, ngrid)
    with torch.no_grad():
        out_ref = ref_model(x, params)

    OpenWQHbvCoupler.write_tracer_config(tmp, routing=routing, precip_mg_l={'TRACER': c_rain})
    coupler = OpenWQHbvCoupler(tmp, area_m2, datetime.datetime(1995, 10, 1),
                               basin_ids=['%08d' % (b + 1) for b in range(ngrid)])
    model = make_model(nmul, routing)
    model.wq_hook = coupler
    with torch.no_grad():
        out = model(x, params)
    return out_ref, out, coupler, area_m2, c_rain


def _check_identical(out_ref, out):
    for k in out_ref:
        if out_ref[k] is None:
            assert out[k] is None
            continue
        assert torch.equal(out_ref[k], out[k]), "output %s changed by the hook" % k


def _check_closure(coupler, c_rain):
    entered = (coupler.mass_in * c_rain).sum()
    left = coupler.mass_out.sum()
    inside = coupler.total_mass()
    assert entered > 0
    np.testing.assert_allclose(left + inside, entered, rtol=1e-6)
    # outlet concentrations are finite and non-negative (they can exceed the
    # rain concentration because evapotranspiration removes water only)
    conc = coupler.stream_conc[np.isfinite(coupler.stream_conc)]
    assert conc.size > 0 and conc.min() >= 0.0


def test_hook_keeps_hydrology_and_routing():
    with tempfile.TemporaryDirectory() as tmp:
        out_ref, out, coupler, area_m2, c_rain = run_pair(tmp, routing=True)
        _check_identical(out_ref, out)
        routed_m3 = out['streamflow'][:, :, 0].numpy().astype(np.float64) * area_m2 * 1e-3
        np.testing.assert_allclose(coupler.stream_flow_m3, routed_m3, rtol=2e-4, atol=1e-3)
        _check_closure(coupler, c_rain)


def test_closure_without_routing():
    with tempfile.TemporaryDirectory() as tmp:
        out_ref, out, coupler, area_m2, c_rain = run_pair(tmp, routing=False)
        _check_identical(out_ref, out)
        routed_m3 = out['streamflow'][:, :, 0].numpy().astype(np.float64) * area_m2 * 1e-3
        np.testing.assert_allclose(coupler.stream_flow_m3, routed_m3, rtol=2e-4, atol=1e-3)
        _check_closure(coupler, c_rain)


if __name__ == '__main__':
    test_hook_keeps_hydrology_and_routing()
    test_closure_without_routing()
    print('OK')


# ---------------------------------------------------------------------------
# water mirror: every HBV transfer reaches OpenWQ
# ---------------------------------------------------------------------------
def test_water_balance_audit():
    """The transfers sent to OpenWQ explain every storage change of HBV
    (start + in - out - ET == end) in all compartments, to HBV's nearzero clamps."""
    with tempfile.TemporaryDirectory() as tmp:
        nsteps, ngrid, nmul = 300, 3, 2
        area_m2 = np.array([50.0, 120.0, 300.0]) * 1e6
        x = synthetic_forcing(nsteps, ngrid, seed=3)
        OpenWQHbvCoupler.write_tracer_config(tmp, routing=True, precip_mg_l={'TRACER': 1.0})
        for mode in ('input', 'soil'):
            coupler = OpenWQHbvCoupler(tmp, area_m2, datetime.datetime(1995, 10, 1),
                                       recharge_from=mode, audit=True)
            model = make_model(nmul, True)
            model.wq_hook = coupler
            with torch.no_grad():
                model(x, random_parameters(model, nsteps, ngrid, seed=4))
            res = coupler.audit_max_residual_mm
            assert max(res.values()) < 1e-4, (mode, res)   # HBV clamps SM/SLZ at nearzero = 1e-5 mm (float32)


def test_uniform_tracer_mirrors_water():
    """Without evapotranspiration every water transfer carries solute, so a
    tracer at the same concentration in the initial storages and in the
    precipitation must stay at that concentration in every cell of every
    compartment at every step: at the end of the run mass == c * water volume
    everywhere and the outlet concentration is c on every day with flow."""
    for seed_f, seed_p in ((5, 6), (11, 12)):
        with tempfile.TemporaryDirectory() as tmp:
            nsteps, ngrid, nmul, c = 250, 2, 2, 3.0
            area_m2 = np.array([80.0, 200.0]) * 1e6
            x = synthetic_forcing(nsteps, ngrid, seed=seed_f)
            x['x_phy'][:, :, 2] = 0.0   # PET = 0 -> ETact = 0
            # initial concentration c in the storages that hold water initially; the
            # routing cells and the outlet start empty (an initial concentration there
            # would be applied by OpenWQ when water first arrives, on top of the solute
            # that arrives with it)
            stores = ['SNOWPACK', 'MELTWATER', 'SOIL', 'UPPER_ZONE', 'LOWER_ZONE']
            OpenWQHbvCoupler.write_tracer_config(tmp, routing=True, precip_mg_l={'TRACER': c},
                                                 initial_mg_l={k: {'TRACER': c} for k in stores})
            coupler = OpenWQHbvCoupler(tmp, area_m2, datetime.datetime(1995, 10, 1), audit=True)
            model = make_model(nmul, True)
            model.wq_hook = coupler
            with torch.no_grad():
                model(x, random_parameters(model, nsteps, ngrid, seed=seed_p))
            assert max(coupler.audit_max_residual_mm.values()) < 1e-4
            net = coupler.audit_max_network_mm
            assert max(net['composition'].values()) < 1e-4, net
            assert max(net['over_export'].values()) < 1e-4, net
            host, f = coupler.host, coupler.f
            end = dict(zip(['SNOWPACK', 'MELTWATER', 'SOIL', 'UPPER_ZONE', 'LOWER_ZONE'],
                           [s.numpy().astype(np.float64) for s in model._state_cache]))
            for name, vol_mm in end.items():
                mass = host.mass_array(coupler.i[name], 0)[:, 0, :]
                np.testing.assert_allclose(mass, c * vol_mm * f, rtol=1e-5, atol=c * 1e-3, err_msg=name)
            np.testing.assert_allclose(host.mass_array(coupler.i['ROUTING'], 0)[:, 0, :], c * coupler.lag_vol,
                                       rtol=1e-5, atol=c * 1e-3, err_msg='ROUTING')
            np.testing.assert_allclose(host.mass_array(coupler.i['STREAM'], 0)[:, 0, 0], c * coupler.stream_vol,
                                       rtol=1e-5, atol=c * 1e-3, err_msg='STREAM')
            # outlet: c on every day with a meaningful flow (HBV's float32 states leave
            # a round-off of ~1e-5 mm per step, visible only while the stream is dry)
            flow_mm = coupler.stream_flow_m3 / coupler.f_basin[None, :]
            ok = np.isfinite(coupler.stream_conc) & (flow_mm > 1e-3)
            assert ok.sum() > 0.8 * ok.size
            np.testing.assert_allclose(coupler.stream_conc[ok], c, rtol=1e-4)
            # the mass that left the domain is c times the water that reached the outlet
            # (the arrivals of the last step have not left yet)
            np.testing.assert_allclose(coupler.mass_out.sum(), c * coupler.stream_flow_m3[:-1].sum(), rtol=1e-5)
