#!/usr/bin/env python
"""Run the pretrained differentiable HBV 1.1p (dMG example) with OpenWQ.

Same setup as generic_deltaModel/example/hydrology/example_dhbv_1_1p.ipynb
(sim mode on the CAMELS 531 basins with the pretrained dHBV 1.1p), restricted
to a few basins, with the OpenWQ coupler attached to the HBV physics model.
The hydrology is unchanged; OpenWQ writes its HDF5 outputs in the case folder.

Needs the dMG assets (1-camels.zip, 3-dhbv_1_1p.zip from the mhpi-release S3
bucket) unpacked in --assets, the hydrolink module built in this folder (see
README.md) and an OpenWQ case folder with openWQ_master.json (--case; a
conservative tracer configuration is written when the folder has none).

Example:
  python run_dhbv_openwq.py \
      --dmg-root ../../../generic_deltaModel --assets ./mhpi_assets \
      --case ./dhbv_openwq_case --basins 5
"""
import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dmg-root', required=True, help='generic_deltaModel repository root')
    ap.add_argument('--assets', required=True, help='folder with 1-camels/ and 3-dhbv_1_1p/ unpacked')
    ap.add_argument('--case', required=True, help='OpenWQ case folder (openWQ_master.json)')
    ap.add_argument('--config', default=None, help='dMG yaml (default: <dmg-root>/example/conf/config_dhbv_1_1p.yaml)')
    ap.add_argument('--basins', type=int, default=5, help='number of basins (first N of the 531 subset)')
    ap.add_argument('--gage-ids', nargs='*', type=int, default=None, help='USGS gage ids instead of --basins')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--precip-mg-l', type=float, default=1.0, help='tracer concentration of precipitation (if a config is written)')
    args = ap.parse_args()

    dmg_root = Path(args.dmg_root).resolve()
    sys.path.insert(0, str(dmg_root))
    from example import load_config  # noqa: E402  (generic_deltaModel/example/__init__.py)
    from dmg import ModelHandler  # noqa: E402
    from dmg.core.data import txt_to_array  # noqa: E402
    from dmg.core.utils import import_data_loader, print_config, set_randomseed  # noqa: E402
    try:
        from dmg.core.utils import import_data_sampler  # noqa: E402
    except ImportError:  # older dMG
        import_data_sampler = None
    from openwq import OpenWQHbvCoupler  # noqa: E402

    cfg_path = args.config or str(dmg_root / 'example' / 'conf' / 'config_dhbv_1_1p.yaml')
    assets = Path(args.assets).resolve()
    config = load_config(cfg_path, make_dirs=False)
    config['mode'] = 'sim'
    config['device'] = args.device
    if args.device == 'cpu' and config['model']['nn']['name'] == 'CudnnLstmModel':
        config['model']['nn']['name'] = 'LstmModel'
    config['model_dir'] = str(assets / '3-dhbv_1_1p')
    config['observations']['data_path'] = str(assets / '1-camels' / 'camels_daymetv2')
    config['observations']['gage_info'] = str(assets / '1-camels' / 'gage_id.npy')
    config['observations']['subset_path'] = str(assets / '1-camels' / '531sub_id.txt')
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
        OpenWQHbvCoupler.write_tracer_config(case, routing=phy.routing, precip_mg_l={'TRACER': args.precip_mg_l})
        print('wrote conservative-tracer OpenWQ configuration in', case)
    coupler = OpenWQHbvCoupler(case, area_m2, start, basin_ids=basin_ids, verbose=True)
    phy.wq_hook = coupler

    # 6. forward (inference) with OpenWQ running inside the HBV loop
    out = model(sample, eval=True)[model.models[0]]
    q = out['streamflow'][:, :, 0].cpu().numpy()
    print('\nstreamflow (mm/day), first 5 days x basins:\n', q[:5])
    print('outlet tracer concentration (mg/L), first 5 days x basins:\n', coupler.stream_conc[:5])
    entered = (coupler.mass_in * args.precip_mg_l).sum()
    print('tracer mass: entered %.6g g, left %.6g g, inside %.6g g' % (entered, coupler.mass_out.sum(), coupler.total_mass()))
    np.savez(case / 'dhbv_openwq_summary.npz', basin_ids=np.array(basin_ids), streamflow_mm=q,
             stream_conc_mg_l=coupler.stream_conc, stream_flow_m3=coupler.stream_flow_m3, area_m2=area_m2)
    print('OpenWQ outputs in', case / 'openwq_out', '; summary in', case / 'dhbv_openwq_summary.npz')


if __name__ == '__main__':
    main()
