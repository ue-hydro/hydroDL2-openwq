"""Tracer-closure test of the OpenWQ Python hydrolink (hydrodl2.openwq.host).

Three buckets in series (SOIL -> UPPER -> LOWER -> loss) for a few basins,
driven from Python with a conservative tracer:

* water volumes and fluxes are generated here (a toy hydrology),
* the tracer enters with the PRECIP external water flux at a fixed concentration,
* every water transfer moves tracer mass with the water (NATIVE_TD_ADV),
* the sum of the mass still in the domain plus the mass that left must equal
  the mass that entered (closure), and the compartment masses must match a
  pure-Python reference bookkeeping of the same transfers.

Run:  python -m pytest tests/test_openwq_host.py -q   (or run this file)
"""
import tempfile
from pathlib import Path

import numpy as np

from hydrodl2.openwq import host as openwq  # noqa: E402


def toy_hydrology(nt, nb, seed=0):
    """Daily volumes (m3) of three buckets and the fluxes between them."""
    rng = np.random.default_rng(seed)
    rain = rng.gamma(2.0, 5.0, size=(nt, nb)) * (rng.random((nt, nb)) < 0.4)  # m3/day
    vol = np.zeros((nt + 1, nb, 3))
    vol[0] = [100.0, 50.0, 200.0]
    q_s2u = np.zeros((nt, nb)); q_u2l = np.zeros((nt, nb)); q_out = np.zeros((nt, nb))
    for t in range(nt):
        s, u, l = vol[t, :, 0] + rain[t], vol[t, :, 1], vol[t, :, 2]
        q_s2u[t] = 0.10 * s
        q_u2l[t] = 0.20 * (u + q_s2u[t])
        q_out[t] = 0.05 * (l + q_u2l[t])
        vol[t + 1, :, 0] = s - q_s2u[t]
        vol[t + 1, :, 1] = u + q_s2u[t] - q_u2l[t]
        vol[t + 1, :, 2] = l + q_u2l[t] - q_out[t]
    return rain, vol, q_s2u, q_u2l, q_out


def reference_tracer(rain, vol, q_s2u, q_u2l, q_out, c_rain_g_m3, c0_g_m3):
    """Pure-Python mass bookkeeping of the same transfers (g, the OpenWQ internal unit).

    OpenWQ accumulates every transfer of a host time step as a derivative from
    the state at the START of the step and applies them together at the end of
    the step (operator splitting), so the reference does the same: all fluxes of
    a step use the start-of-step masses, then the state is updated once.
    """
    nt, nb = rain.shape
    # The initial condition is applied by OpenWQ as a derivative of the first
    # host step (it becomes effective at the end of step 0, like an inflow), so
    # the transfers of step 0 act on an empty domain.
    m_ic = vol[0] * np.asarray(c0_g_m3)  # [nb, 3]
    m = np.zeros_like(m_ic)
    m_out = np.zeros(nb)
    for t in range(nt):
        m0 = m.copy()
        d_ic = m_ic if t == 0 else 0.0
        v_s = vol[t, :, 0] + rain[t]
        v_u = vol[t, :, 1] + q_s2u[t]
        v_l = vol[t, :, 2] + q_u2l[t]
        d_in = rain[t] * c_rain_g_m3
        d_s2u = np.where(v_s > 0, q_s2u[t] / v_s * m0[:, 0], 0.0)
        d_u2l = np.where(v_u > 0, q_u2l[t] / v_u * m0[:, 1], 0.0)
        d_out = np.where(v_l > 0, q_out[t] / v_l * m0[:, 2], 0.0)
        m[:, 0] = m0[:, 0] + d_in - d_s2u
        m[:, 1] = m0[:, 1] + d_s2u - d_u2l
        m[:, 2] = m0[:, 2] + d_u2l - d_out
        m += d_ic
        m_out += d_out
    return m, m_out


def run_case(folder, nt=40, nb=3):
    rain, vol, q_s2u, q_u2l, q_out = toy_hydrology(nt, nb)
    c_rain = 2.0   # mg/L
    c0 = [10.0, 0.0, 0.0]  # mg/L initial in SOIL, UPPER, LOWER
    comps = ["SOIL", "UPPER_ZONE", "LOWER_ZONE"]
    master = openwq.write_minimal_config(
        folder, comps, species=("TRACER",),
        initial_mg_l={c: {"TRACER": v} for c, v in zip(comps, c0)},
        ewf_names=("PRECIP",), ewf_mg_l={"PRECIP": {"TRACER": c_rain}},
        timestep=(1, "day"), project="tracer chain test")

    with openwq.case_dir(folder):
        host = openwq.Host(master)
        icmp = [host.add_compartment(c, nx=nb) for c in comps]
        host.add_ext_flux("PRECIP", nx=nb)
        host.set_cell_label("basin")
        host.init()
        assert host.compartment_names() == comps
        assert host.num_species() >= 1
        clock = openwq.Clock(2001, 1, 1, step_hours=24)
        m_out = np.zeros(nb)
        for t in range(nt):
            y, mo, d, h, mi = clock.at(t)
            # state at the start of the step (what the host model knows)
            for k in range(3):
                host.set_water_volume_array(icmp[k], vol[t, :, k])
            host.time_start(y, mo, d, h, mi)
            for b in range(nb):
                if rain[t, b] > 0:
                    host.inflow("PRECIP", icmp[0], b, 0, 0, float(rain[t, b]), y, mo, d, h, mi)
            for b in range(nb):
                v_s = vol[t, b, 0] + rain[t, b]
                host.move(icmp[0], b, 0, 0, icmp[1], b, 0, 0, float(q_s2u[t, b]), float(v_s), y, mo, d, h, mi)
                v_u = vol[t, b, 1] + q_s2u[t, b]
                host.move(icmp[1], b, 0, 0, icmp[2], b, 0, 0, float(q_u2l[t, b]), float(v_u), y, mo, d, h, mi)
                v_l = vol[t, b, 2] + q_u2l[t, b]
                host.move(icmp[2], b, 0, 0, openwq.LOSS, 0, 0, 0, float(q_out[t, b]), float(v_l), y, mo, d, h, mi)
            host.time_end(y, mo, d, h, mi)
        m_end = np.stack([host.mass_array(icmp[k], 0)[:, 0, 0] for k in range(3)], axis=1)
        out_dir = host.output_dir()
    # mg/L == g/m3: OpenWQ keeps mass in g internally
    ref_m, ref_out = reference_tracer(rain, vol, q_s2u, q_u2l, q_out, c_rain, np.asarray(c0))
    return m_end, ref_m, ref_out, out_dir, rain, vol, c_rain, c0


def test_tracer_chain_closure():
    with tempfile.TemporaryDirectory() as tmp:
        m_end, ref_m, ref_out, out_dir, rain, vol, c_rain, c0 = run_case(tmp)
        m_in = vol[0].dot(np.asarray(c0)).sum() + (rain * c_rain).sum()  # g
        # compartment masses match the reference bookkeeping
        print('openwq/reference ratio:', (m_end / ref_m).round(6).tolist())
        np.testing.assert_allclose(m_end, ref_m, rtol=1e-6, atol=1e-9)
        # closure: in-domain + left = entered
        np.testing.assert_allclose(m_end.sum() + ref_out.sum(), m_in, rtol=1e-6)
        # output was written
        h5 = list(Path(tmp, out_dir).rglob("*.h5"))
        assert h5, "no HDF5 output written in %s" % out_dir


def test_bad_indices_raise():
    with tempfile.TemporaryDirectory() as tmp:
        master = openwq.write_minimal_config(tmp, ["SOIL"])
        with openwq.case_dir(tmp):
            host = openwq.Host(master)
            soil = host.add_compartment("SOIL", nx=2)
            host.add_ext_flux("PRECIP", nx=2)
            try:
                host.time_start(2001, 1, 1)
                raise AssertionError("time_start before init must fail")
            except RuntimeError:
                pass
            host.init()
            for bad in [(soil, 2, 0, 0), (soil, 0, 1, 0), (5, 0, 0, 0)]:
                try:
                    host.set_water_volume(*bad, 1.0)
                    raise AssertionError("bad cell %r accepted" % (bad,))
                except (IndexError, ValueError):
                    pass


if __name__ == "__main__":
    test_bad_indices_raise()
    test_tracer_chain_closure()
    print("OK")
