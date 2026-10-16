#!/usr/bin/env python
"""Run the pretrained differentiable HBV 1.1p (dMG example) with OpenWQ.

Same set-up as generic_deltaModel/example/hydrology/example_dhbv_1_1p.ipynb
(sim mode on the CAMELS 531 basins with the pretrained dHBV 1.1p), restricted
to a few basins, with the OpenWQ coupler attached to the HBV physics model.
The hydrology is unchanged; OpenWQ writes its HDF5 outputs in the case folder,
this script saves a figure and CSV files with HBV and OpenWQ results, and then
builds the standard OpenWQ results report (interactive HTML with the basin map)
with the hdf5_support_lib of the OpenWQ clone (--no-report to skip,
--report-only to rebuild it for an existing case).

Needs the dMG assets (1-camels.zip, 3-dhbv_1_1p.zip from the mhpi-release S3
bucket) unpacked in --assets, the hydrolink module built in this folder (see
README.md) and an OpenWQ case folder (--case). When the case folder has no
openWQ_master.json, a tracer configuration is written: tracer in precipitation
(--precip-mg-l), initial soil-water concentration (--initial-soil-mg-l) and a
first-order decay (--decay-per-day).

Example:
  python run_dhbv_openwq.py --dmg-root ../../../generic_deltaModel \\
      --assets ../../../dhbv_data --case ../../../dhbv_openwq_case --basins 5
"""
import argparse
import datetime
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# dMG configuration (hydra composition of the example yaml, with CPU overrides)
# ----------------------------------------------------------------------------
def load_dmg_config(dmg_root, assets, device='cpu', config_name='config_dhbv_1_1p'):
    import hydra
    from dmg.core.utils import initialize_config

    conf_dir = str(Path(dmg_root) / 'example' / 'conf')

    def find_asset(name):
        # the zips unpack with or without a top-level folder: search recursively
        hits = sorted(glob.glob(str(Path(assets) / '**' / name), recursive=True))
        if not hits:
            raise SystemExit('%s not found under %s (unpack 1-camels.zip and 3-dhbv_1_1p.zip there)' % (name, assets))
        return hits[0]

    ckpts = {int(re.search(r'_ep(\d+)\.pt$', c).group(1)): c
             for c in glob.glob(str(Path(assets) / '**' / 'hbv_1_1p_ep*.pt'), recursive=True)}
    if not ckpts:
        raise SystemExit('no hbv_1_1p_ep*.pt checkpoint under %s' % assets)
    with hydra.initialize_config_dir(config_dir=conf_dir, version_base='1.3'):
        default_epoch = int(hydra.compose(config_name=config_name)['test']['test_epoch'])
    epoch = default_epoch if default_epoch in ckpts else max(ckpts)   # the example evaluates epoch 50
    model_dir = Path(ckpts[epoch]).parent
    overrides = [
        'mode=sim',
        'device=%s' % device,
        'model_dir=%s' % model_dir,
        'test.test_epoch=%d' % epoch,
        'observations.data_path=%s' % find_asset('camels_daymetv2'),
        'observations.gage_info=%s' % find_asset('gage_id.npy'),
        'observations.subset_path=%s' % find_asset('531sub_id.txt'),
    ]
    if device == 'cpu':
        overrides.append('model.nn.name=LstmModel')
    with hydra.initialize_config_dir(config_dir=conf_dir, version_base='1.3'):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    config = initialize_config(cfg, make_dirs=False)
    # initialize_config derives model_dir from output_dir: point it back at the
    # pretrained checkpoint folder (as the dMG notebook does for downloaded models)
    config['model_dir'] = str(model_dir)
    return config


# ----------------------------------------------------------------------------
# results: CSV + figure
# ----------------------------------------------------------------------------
def save_results(case, basin_ids, dates, q_sim, q_obs, coupler, precip_mg_l, title=''):
    """Write per-basin CSV files and a figure with HBV streamflow (sim, obs) and
    the OpenWQ outlet concentration. q_sim/q_obs are [time, basin] in mm/day on
    the dates of the hook (q_sim may be shorter: it is aligned at the end)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    nt = len(dates)
    conc = coupler.stream_conc
    flow_m3 = coupler.stream_flow_m3
    off_sim = nt - q_sim.shape[0]
    off_obs = nt - q_obs.shape[0] if q_obs is not None else None
    nb = len(basin_ids)
    fig, axes = plt.subplots(nb, 2, figsize=(14, 2.6 * nb + 0.8), squeeze=False, sharex=True)
    for b, gid in enumerate(basin_ids):
        df = pd.DataFrame({'date': dates})
        df['hbv_streamflow_mm_day'] = np.r_[np.full(off_sim, np.nan), q_sim[:, b]]
        if q_obs is not None:
            df['obs_streamflow_mm_day'] = np.r_[np.full(off_obs, np.nan), q_obs[:, b]]
        df['openwq_outlet_flow_m3_day'] = flow_m3[:, b]
        df['openwq_outlet_conc_mg_L'] = conc[:, b]
        df['openwq_outlet_mass_g_day'] = coupler.stream_mass[:, b]
        df.to_csv(Path(case) / ('dhbv_openwq_%s.csv' % gid), index=False)

        ax = axes[b, 0]
        if q_obs is not None:
            ax.plot(dates[off_obs:], q_obs[:, b], color='0.55', lw=0.8, label='observed')
        ax.plot(dates[off_sim:], q_sim[:, b], color='tab:blue', lw=0.9, label='dHBV 1.1p (routed)')
        ax.set_ylabel('streamflow\n[mm/day]')
        ax.set_title('basin %s: HBV streamflow' % gid, fontsize=10, loc='left')
        ax.legend(fontsize=8, loc='upper right')
        ax = axes[b, 1]
        ax.plot(dates, conc[:, b], color='tab:red', lw=0.9, label='outlet concentration')
        ax.axhline(precip_mg_l, color='0.4', ls='--', lw=0.8, label='precipitation (%g mg/L)' % precip_mg_l)
        ax.set_ylabel('tracer\n[mg/L]')
        ax.set_title('basin %s: OpenWQ outlet tracer' % gid, fontsize=10, loc='left')
        ax.legend(fontsize=8, loc='upper right')
    for ax in axes[-1]:
        ax.set_xlabel('date')
    fig.suptitle(title or 'dHBV 1.1p + OpenWQ (forward coupling)', fontsize=12)
    fig.tight_layout()
    png = Path(case) / 'dhbv_openwq_results.png'
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def hdf5_compartment_series(case, compartment, species='TRACER', units=None):
    """Read one OpenWQ HDF5 output file (<compartment>@<species>#<units>-main.h5;
    the units are those of the master file, e.g. KG for masses or MG|L for
    concentrations, found automatically when ``units`` is None).

    Returns a dict with 'timestamps' (datetimes), 'cells' (cell labels), 'xyz'
    (cell x,y,z indices), 'units' and every 2-D dataset of the file as
    array[time, cell] (the values are in the dataset 'concentrations' whatever
    the units), or None when h5py or the file is missing. Read in a process that
    is not holding the OpenWQ host (the files are closed on exit)."""
    try:
        import h5py
    except ImportError:
        return None
    pattern = '%s@%s#%s-main.h5' % (compartment, species, (units or '*').replace('/', '|'))
    hits = sorted(glob.glob(str(Path(case) / 'openwq_out' / 'HDF5' / pattern)))
    if not hits:
        return None
    path = Path(hits[0])
    file_units = re.search(r'#(.*)-main\.h5$', path.name).group(1).replace('|', '/')
    try:
        f = h5py.File(path, 'r')
    except OSError:
        return None
    out = {'units': file_units}
    with f:
        for k in f.keys():
            d = np.asarray(f[k])
            if d.dtype.kind in 'SO':
                d = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in d.reshape(-1)])
            if k == 'timestamps':
                ts = []
                for x in d:
                    try:
                        ts.append(datetime.datetime.strptime(str(x).title(), '%Y%b%d-%H:%M:%S'))
                    except ValueError:
                        ts.append(str(x))
                out['timestamps'] = ts
            elif k == 'xyz_elements':
                out['xyz'] = d
            elif d.ndim == 2:
                out[k] = d
            else:
                out['cells'] = d
    return out


# ----------------------------------------------------------------------------
# standard OpenWQ results report (hdf5_support_lib of the OpenWQ clone)
# ----------------------------------------------------------------------------
HYDROLINK_DIR = Path(__file__).resolve().parent
HDF5_LIB = HYDROLINK_DIR / 'openwq' / 'supporting_scripts' / '2_Read_Outputs' / 'hdf5_support_lib'
COMPARTMENTS_ALL = ['SNOWPACK', 'MELTWATER', 'SOIL', 'UPPER_ZONE', 'LOWER_ZONE', 'ROUTING', 'STREAM']


def make_basin_shapefile(assets, basin_ids, case):
    """Subset of the CAMELS basin polygons (loc/camels671.shp of the 1-camels
    assets) for the simulated basins, with a 'basin' field equal to the OpenWQ
    cell labels (zero-padded gage ids). Returns the path or None."""
    try:
        import geopandas as gpd
    except ImportError:
        return None
    hits = glob.glob(str(Path(assets) / '**' / 'camels671.shp'), recursive=True)
    if not hits:
        return None
    g = gpd.read_file(hits[0])
    g['basin'] = g['gage_id'].astype(int).map(lambda v: '%08d' % v)
    sel = g[g['basin'].isin([str(b) for b in basin_ids])]
    if sel.empty:
        return None
    out = Path(case) / 'shapefiles'
    out.mkdir(exist_ok=True)
    shp = out / 'camels_basins.shp'
    sel[['basin', 'gage_id', 'lat', 'lon', 'geometry']].to_file(shp)
    return str(shp)


def results_report(case, compartments, species=('TRACER',), basin_shapefile=None, basin_key='basin',
                   host_label='dhbv (hydroDL2 differentiable HBV 1.1p)', name='dhbv_openwq_results_report'):
    """Run the standard OpenWQ results report (Read_h5_driver + Plot_h5_driver)
    in a separate process (the HDF5 files are only readable once the OpenWQ host
    of this process is gone). Returns the HTML path or None."""
    case = Path(case).resolve()
    out_html = case / 'openwq_out' / ('%s.html' % name)
    snippet = """
import sys
sys.path.insert(0, %(lib)r)
import Read_h5_driver as h5_rlib
import Plot_h5_driver as h5_plib
res = h5_rlib.Read_h5_driver(
    openwq_info={"path_to_results": %(out)r, "mapping_key": %(key)r},
    output_format="HDF5", debugmode=False, cmp=%(cmp)r, space_elem="all",
    chemSpec=%(species)r, chemUnits="MG/L", noDataFlag=-9999, sediment_as_well=False)
r = h5_plib.Plot_h5_driver(
    what2map="openwq", hostmodel="dhbv", host_label=%(label)r, mapping_key_values="all",
    openwq_results=res, chemSpec=%(species)r, sediment_as_well=False, debugmode=False,
    output_path=%(html)r, basin_shapefile=%(shp)r, basin_mapping_key=%(key)r,
    feature_label=%(key)r, separator=" | ")
print("RESULT:", r)
""" % dict(lib=str(HDF5_LIB), out=str(case / 'openwq_out'), key=basin_key, cmp=list(compartments),
           species=list(species), label=host_label, html=str(out_html), shp=basin_shapefile)
    import subprocess
    env = dict(os.environ, BROWSER='true')
    log = case / 'results_report.log'
    with open(log, 'w') as f:
        rc = subprocess.run([sys.executable, '-c', snippet], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(case)).returncode
    if rc == 0 and out_html.exists():
        return str(out_html)
    print('results report failed (exit %d), see %s' % (rc, log))
    return None


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dmg-root', required=True, help='generic_deltaModel repository root')
    ap.add_argument('--assets', required=True, help='folder with 1-camels/ and 3-dhbv_1_1p/ unpacked')
    ap.add_argument('--case', required=True, help='OpenWQ case folder (openWQ_master.json)')
    ap.add_argument('--basins', type=int, default=5, help='number of basins (first N of the 531 subset)')
    ap.add_argument('--gage-ids', nargs='*', type=int, default=None, help='USGS gage ids instead of --basins')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--precip-mg-l', type=float, default=1.0, help='tracer concentration of precipitation')
    ap.add_argument('--initial-soil-mg-l', type=float, default=0.0, help='initial tracer concentration of the soil water')
    ap.add_argument('--decay-per-day', type=float, default=0.0, help='first-order decay of the tracer [1/day]')
    ap.add_argument('--no-report', action='store_true', help='skip the OpenWQ results report (HTML) after the run')
    ap.add_argument('--report-only', action='store_true', help='only (re)build the results report of an existing case')
    ap.add_argument('--basin-shapefile', default=None, help="basin polygons for the report map (field 'basin' = cell label); "
                    'default: subset of the CAMELS loc/camels671.shp of --assets')
    args = ap.parse_args()

    if args.report_only:
        case = Path(args.case).resolve()
        summary = np.load(case / 'dhbv_openwq_summary.npz', allow_pickle=True)
        basin_ids = [str(b) for b in summary['basin_ids']]
        shp = args.basin_shapefile or make_basin_shapefile(args.assets, basin_ids, case)
        html = results_report(case, COMPARTMENTS_ALL, basin_shapefile=shp)
        print('results report:', html)
        return

    dmg_root = Path(args.dmg_root).resolve()
    sys.path.insert(0, str(dmg_root))
    from dmg import ModelHandler  # noqa: E402
    from dmg.core.data import txt_to_array  # noqa: E402
    from dmg.core.utils import import_data_loader, print_config, set_randomseed  # noqa: E402
    try:
        from dmg.core.utils import import_data_sampler  # noqa: E402
    except ImportError:  # older dMG
        import_data_sampler = None
    from openwq import OpenWQHbvCoupler  # noqa: E402

    config = load_dmg_config(dmg_root, Path(args.assets).resolve(), device=args.device)
    print_config(config)
    set_randomseed(config['seed'])

    # 1. model (LSTM + HBV 1.1p) with the pretrained weights, and the data
    model = ModelHandler(config, verbose=True)
    data_loader = import_data_loader(config['data_loader'])(config, test_split=True, overwrite=False)
    dataset = data_loader.dataset
    gage_ids = [int(g) for g in txt_to_array(config['observations']['subset_path'])]

    # 2. basin subset (dataset order == subset list order, as in the dMG notebook)
    if args.gage_ids:
        idx = [gage_ids.index(g) for g in args.gage_ids]
    else:
        idx = list(range(min(args.basins, len(gage_ids))))
    i_s, i_e = min(idx), max(idx) + 1
    if idx != list(range(i_s, i_e)):
        raise SystemExit('--gage-ids must be contiguous in the 531 list for this example (got %s)' % idx)
    if import_data_sampler is not None:
        sampler = import_data_sampler(config['data_sampler'])(config)
    else:
        from dmg.core.data.samplers.hydro_sampler import HydroSampler
        sampler = HydroSampler(config)
    sample = sampler.get_validation_sample(dataset, i_s, i_e)
    basin_ids = ['%08d' % g for g in gage_ids[i_s:i_e]]

    # 3. basin areas [m2] from the attributes (area_gages2 in km2)
    attrs = config['model']['nn']['attributes']
    area_km2 = dataset['c_nn'][i_s:i_e, attrs.index(config['observations']['area_name'])].cpu().numpy().astype(float)
    area_m2 = area_km2 * 1e6

    # 4. first day seen by the physics loop
    phy = model.model_dict[model.models[0]].phy_model
    start = datetime.datetime.strptime(config['sim']['start_time'], '%Y/%m/%d')
    if phy.warmup_states:
        start += datetime.timedelta(days=int(phy.warmup))

    # 5. OpenWQ case + coupler
    case = Path(args.case).resolve()
    case.mkdir(parents=True, exist_ok=True)
    if not (case / 'openWQ_master.json').exists():
        OpenWQHbvCoupler.write_tracer_config(
            case, routing=phy.routing, precip_mg_l={'TRACER': args.precip_mg_l},
            initial_mg_l={'SOIL': {'TRACER': args.initial_soil_mg_l}},
            decay_per_day=args.decay_per_day, units='MG/L')   # HDF5 output as concentrations
        print('wrote tracer OpenWQ configuration in', case)
    coupler = OpenWQHbvCoupler(case, area_m2, start, basin_ids=basin_ids, verbose=True)
    phy.wq_hook = coupler

    # 6. forward (inference) with OpenWQ running inside the HBV loop
    out = model(sample, eval=True)[model.models[0]]
    q_sim = out['streamflow'][:, :, 0].cpu().numpy().astype(float)
    q_obs = None
    if 'target' in sample:
        q_obs = sample['target'][:, :, 0].cpu().numpy().astype(float)
        q_obs = np.where(q_obs < 0, np.nan, q_obs)
    nt = coupler.nsteps
    dates = np.array([start + datetime.timedelta(days=t) for t in range(nt)])

    entered = (coupler.mass_in * args.precip_mg_l).sum()
    left, inside = coupler.mass_out.sum(), coupler.total_mass()
    print('\nbasins:', basin_ids, '| areas km2:', np.round(area_km2, 1).tolist())
    print('hook steps: %d (from %s), HBV output steps: %d' % (nt, start.date(), q_sim.shape[0]))
    print('streamflow (mm/day), last 5 days x basins:\n', np.round(q_sim[-5:], 3))
    print('outlet tracer concentration (mg/L), last 5 days x basins:\n', np.round(coupler.stream_conc[-5:], 3))
    print('tracer mass [g]: entered with precipitation %.6g, left through the outlets %.6g, inside %.6g, closure error %.3g'
          % (entered, left, inside, (entered + coupler.initial_mass - left - inside) / max(entered, 1e-12)))
    png = save_results(case, basin_ids, dates, q_sim, q_obs, coupler, args.precip_mg_l,
                       title='dHBV 1.1p (pretrained, CAMELS) + OpenWQ tracer: rain %g mg/L, soil IC %g mg/L, decay %g/day'
                             % (args.precip_mg_l, args.initial_soil_mg_l, args.decay_per_day))
    np.savez(case / 'dhbv_openwq_summary.npz', basin_ids=np.array(basin_ids), dates=dates.astype('datetime64[D]'),
             streamflow_mm=q_sim, stream_conc_mg_l=coupler.stream_conc, stream_flow_m3=coupler.stream_flow_m3,
             area_m2=area_m2)
    print('figure:', png, '| CSV per basin + summary npz in', case, '| OpenWQ HDF5 in', case / 'openwq_out')

    if not args.no_report:
        del coupler.host, phy.wq_hook   # release the OpenWQ host so its HDF5 files can be read
        shp = args.basin_shapefile or make_basin_shapefile(args.assets, basin_ids, case)
        html = results_report(case, [c for c in COMPARTMENTS_ALL if (case / 'openwq_out' / 'HDF5' / ('%s@TRACER#MG|L-main.h5' % c)).exists()],
                              basin_shapefile=shp)
        print('results report:', html)


if __name__ == '__main__':
    main()
