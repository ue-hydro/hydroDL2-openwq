// OpenWQ hydrolink for Python host models (pybind11): differentiable HBV (hydroDL2 / dMG)
// Copyright 2020, Diogo Costa, diogo.pinhodacosta@canada.ca
// This file is part of OpenWQ model.
//
// This program, openWQ, is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
// ----------------------------------------------------------------------------
// Hydrolink of OpenWQ for Python host models.
//
// Layout (as for the other hosts): this folder holds the hydrolink files and
// the OpenWQ source tree is cloned in ./openwq (built with the CMake target
// python_openwq of ./openwq/CMakeLists.txt, which compiles this file into the
// Python extension module openwq_py placed in this folder).
//
// The Python hydrological model (differentiable HBV in hydroDL2/dMG, through
// hydrodl2.openwq) declares its compartments, external water fluxes and
// dependency variables, then drives the OpenWQ time loop with the same five
// calls the Fortran/C++ couplers use:
//
//     host = openwq_py.Host("openWQ_master.json")   (hydrodl2.openwq.host.Host)
//     host.add_compartment("SOIL", nx, ny, nz)      (index = order of calls)
//     host.add_ext_flux("PRECIP", nx, ny, nz)
//     host.add_depend("Tair_K", nx, ny, nz)
//     host.add_flux_export("Q", src_compartment, nx, ny, nz)
//     host.set_cell_label("basin")
//     host.init()                                   InitialConfig + ParseEWFandSS
//     for every host time step:
//         host.set_water_volume(icmp, array[nx, ny, nz])      (m3)
//         host.set_depend(idep, array[nx, ny, nz])
//         host.time_start(y, m, d, h, mi)           RunTimeLoopStart
//         host.inflow("PRECIP", icmp, ix, iy, iz, wflux_m3)     RunSpaceStep_IN
//         host.move(src, ix, iy, iz, dst, ix, iy, iz, wflux_m3, wmass_src_m3)
//         host.move(src, ix, iy, iz, -1, 0, 0, 0, wflux_m3, wmass_src_m3)  loss
//         host.time_end(y, m, d, h, mi)             RunTimeLoopEnd (writes output)
//
// Indices are 0-based, like inside OpenWQ (the Fortran couplers subtract 1).
// Water volumes are in m3 and fluxes in m3 per host time step, as for every
// other host model. The master JSON path is resolved relative to the current
// working directory, so the Python host should chdir() to the case folder (or
// pass an absolute path and keep the relative paths inside the JSON files
// valid from the current working directory).
// ----------------------------------------------------------------------------

#ifndef OPENWQ_HYDROLINK_PY_INCLUDED
#define OPENWQ_HYDROLINK_PY_INCLUDED

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <map>

#include "global/OpenWQ_hostModelConfig.hpp"
#include "global/OpenWQ_json.hpp"
#include "global/OpenWQ_wqconfig.hpp"
#include "global/OpenWQ_vars.hpp"
#include "couplercalls/headerfile_CC.hpp"
#include "readjson/headerfile_nlohmann.hpp"
#include "initiate/headerfile_INIT.hpp"
#include "extwatflux_ss/headerfile_EWF_SS.hpp"
#include "units/headerfile_units.hpp"
#include "utils/headerfile_UTILS.hpp"
#include "compute/headerfile_compute.hpp"
#include "output/headerfile_OUT.hpp"
#include "models_CH/headerfile_CH.hpp"
#include "models_TD/headerfile_TD.hpp"
#include "models_LE/headerfile_LE.hpp"
#include "models_SI/headerfile_SI.hpp"
#include "models_TS/headerfile_TS.hpp"

namespace py = pybind11;

class OpenWQPyHost {
public:
    explicit OpenWQPyHost(const std::string& master_json)
        : master_json_(master_json) {}

    // ------------------------------------------------------------------
    // Declaration phase (before init)
    // ------------------------------------------------------------------
    int add_compartment(const std::string& name, int nx, int ny, int nz) {
        require_not_init("add_compartment");
        check_dims(name, nx, ny, nz);
        const int idx = (int) cmp_names_.size();
        hostcfg_->add_HydroComp(idx, name, nx, ny, nz);
        cmp_names_.push_back(name);
        cmp_nx_.push_back(nx); cmp_ny_.push_back(ny); cmp_nz_.push_back(nz);
        return idx;
    }

    int add_ext_flux(const std::string& name, int nx, int ny, int nz) {
        require_not_init("add_ext_flux");
        check_dims(name, nx, ny, nz);
        const int idx = (int) ewf_names_.size();
        hostcfg_->add_HydroExtFlux(idx, name, nx, ny, nz);
        ewf_names_.push_back(name);
        return idx;
    }

    int add_depend(const std::string& name, int nx, int ny, int nz) {
        require_not_init("add_depend");
        check_dims(name, nx, ny, nz);
        const int idx = (int) dep_names_.size();
        hostcfg_->add_HydroDepend(idx, name, nx, ny, nz);
        dep_names_.push_back(name);
        dep_nx_.push_back(nx); dep_ny_.push_back(ny); dep_nz_.push_back(nz);
        return idx;
    }

    int add_flux_export(const std::string& name, int src_compartment, int nx, int ny, int nz) {
        require_not_init("add_flux_export");
        check_dims(name, nx, ny, nz);
        check_cmp(src_compartment);
        const int idx = (int) fexp_names_.size();
        hostcfg_->add_FluxConcExport(idx, name, src_compartment, nx, ny, nz);
        fexp_names_.push_back(name);
        fexp_nx_.push_back(nx); fexp_ny_.push_back(ny); fexp_nz_.push_back(nz);
        return idx;
    }

    void set_cell_label(const std::string& label) {
        require_not_init("set_cell_label");
        cell_label_ = label;
    }

    // Cell ids of one compartment: list of nx*ny*nz strings, iz fastest
    // (same ordering as a C array [nx][ny][nz]). Default = "<ix>_z<iz+1>"
    // ("<ix>" when nz == 1), which is what the other couplers use.
    void set_cell_ids(int icmp, const std::vector<std::string>& ids) {
        require_not_init("set_cell_ids");
        check_cmp(icmp);
        const size_t n = (size_t) cmp_nx_[icmp] * cmp_ny_[icmp] * cmp_nz_[icmp];
        if (ids.size() != n) {
            std::ostringstream s;
            s << "set_cell_ids(" << cmp_names_[icmp] << "): expected " << n
              << " ids, got " << ids.size();
            throw std::invalid_argument(s.str());
        }
        cell_ids_[icmp] = ids;
    }

    // ------------------------------------------------------------------
    // init: allocate state, read the JSON configuration, parse EWF/SS
    // ------------------------------------------------------------------
    void init() {
        require_not_init("init");
        if (cmp_names_.empty())
            throw std::runtime_error("init(): declare at least one compartment first");

        vars_ = std::make_unique<OpenWQ_vars>(
            hostcfg_->get_num_HydroComp(),
            hostcfg_->get_num_HydroExtFlux());

        // cell ids (used by the output and by the SS/EWF cell lookups)
        hostcfg_->set_cellid_to_wqlabel(cell_label_);
        for (int c = 0; c < (int) cmp_names_.size(); c++) {
            arma::Cube<double> domain_xyz(cmp_nx_[c], cmp_ny_[c], cmp_nz_[c]);
            hostcfg_->set_cellid_to_wq_size(domain_xyz);
            auto it = cell_ids_.find(c);
            for (int x = 0; x < cmp_nx_[c]; x++)
                for (int y = 0; y < cmp_ny_[c]; y++)
                    for (int z = 0; z < cmp_nz_[c]; z++) {
                        std::string id;
                        if (it != cell_ids_.end()) {
                            id = it->second[((size_t) x * cmp_ny_[c] + y) * cmp_nz_[c] + z];
                        } else {
                            id = std::to_string(x);
                            if (cmp_ny_[c] > 1) id += "_y" + std::to_string(y + 1);
                            if (cmp_nz_[c] > 1) id += "_z" + std::to_string(z + 1);
                        }
                        hostcfg_->set_cellid_to_wq_at(c, x, y, z, id);
                    }
        }

        wqconfig_->set_OpenWQ_masterjson(master_json_);

        couplercalls_->InitialConfig(
            *hostcfg_, *json_, *wqconfig_, *units_, *utils_, *readjson_,
            *vars_, *initiate_, *TD_, *LE_, *CH_, *SI_, *TS_,
            *extwatflux_ss_, *output_);

        couplercalls_->ParseEWFandSS(
            *json_, *vars_, *hostcfg_, *wqconfig_, *units_, *utils_,
            *output_, *extwatflux_ss_);

        initialized_ = true;
    }

    // ------------------------------------------------------------------
    // Runtime: host state and time loop
    // ------------------------------------------------------------------
    void set_water_volume(int icmp, int ix, int iy, int iz, double vol_m3) {
        require_init("set_water_volume");
        check_cell(icmp, ix, iy, iz);
        hostcfg_->set_waterVol_hydromodel_at(icmp, ix, iy, iz, vol_m3);
    }

    // whole compartment at once: array shaped [nx, ny, nz] (ny/nz may be squeezed)
    void set_water_volume_array(int icmp, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
        require_init("set_water_volume_array");
        check_cmp(icmp);
        const double* p = checked_ptr(arr, cmp_nx_[icmp], cmp_ny_[icmp], cmp_nz_[icmp], "set_water_volume_array");
        for (int x = 0; x < cmp_nx_[icmp]; x++)
            for (int y = 0; y < cmp_ny_[icmp]; y++)
                for (int z = 0; z < cmp_nz_[icmp]; z++)
                    hostcfg_->set_waterVol_hydromodel_at(icmp, x, y, z,
                        p[((size_t) x * cmp_ny_[icmp] + y) * cmp_nz_[icmp] + z]);
    }

    double get_water_volume(int icmp, int ix, int iy, int iz) {
        require_init("get_water_volume");
        check_cell(icmp, ix, iy, iz);
        return hostcfg_->get_waterVol_hydromodel_at(icmp, ix, iy, iz);
    }

    void set_depend(int idep, int ix, int iy, int iz, double value) {
        require_init("set_depend");
        check_dep(idep);
        hostcfg_->set_dependVar_at(idep, ix, iy, iz, value);
    }

    void set_depend_array(int idep, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
        require_init("set_depend_array");
        check_dep(idep);
        const double* p = checked_ptr(arr, dep_nx_[idep], dep_ny_[idep], dep_nz_[idep], "set_depend_array");
        for (int x = 0; x < dep_nx_[idep]; x++)
            for (int y = 0; y < dep_ny_[idep]; y++)
                for (int z = 0; z < dep_nz_[idep]; z++)
                    hostcfg_->set_dependVar_at(idep, x, y, z,
                        p[((size_t) x * dep_ny_[idep] + y) * dep_nz_[idep] + z]);
    }

    void set_flux_volume(int iexp, int ix, int iy, int iz, double vol_m3) {
        require_init("set_flux_volume");
        if (iexp < 0 || iexp >= (int) fexp_names_.size())
            throw std::out_of_range("set_flux_volume: bad flux export index");
        hostcfg_->set_fluxVol_hydromodel_at(iexp, ix, iy, iz, vol_m3);
    }

    void time_start(int y, int m, int d, int h, int mi) {
        require_init("time_start");
        couplercalls_->RunTimeLoopStart(
            *hostcfg_, *json_, *wqconfig_, *units_, *utils_, *readjson_,
            *vars_, *initiate_, *TD_, *LE_, *CH_, *SI_, *TS_,
            *extwatflux_ss_, *compute_, *output_, to_time(y, m, d, h, mi));
    }

    // water flux inside the domain (recipient = -1: loss from the domain)
    void move(int src, int ix_s, int iy_s, int iz_s,
              int dst, int ix_r, int iy_r, int iz_r,
              double wflux_m3, double wmass_src_m3,
              int y, int m, int d, int h, int mi) {
        require_init("move");
        check_cell(src, ix_s, iy_s, iz_s);
        if (dst != -1) check_cell(dst, ix_r, iy_r, iz_r);
        couplercalls_->RunSpaceStep(
            *hostcfg_, *json_, *wqconfig_, *units_, *utils_, *readjson_,
            *vars_, *initiate_, *TD_, *TS_, *LE_, *CH_, *SI_,
            *extwatflux_ss_, *compute_, *output_, to_time(y, m, d, h, mi),
            src, ix_s, iy_s, iz_s, dst, ix_r, iy_r, iz_r,
            wflux_m3, wmass_src_m3);
    }

    // external water flux into the domain
    void inflow(const std::string& ewf_name, int dst, int ix_r, int iy_r, int iz_r,
                double wflux_m3, int y, int m, int d, int h, int mi) {
        require_init("inflow");
        check_cell(dst, ix_r, iy_r, iz_r);
        couplercalls_->RunSpaceStep_IN(
            *hostcfg_, *json_, *wqconfig_, *units_, *utils_, *readjson_,
            *vars_, *initiate_, *TD_, *CH_, *TS_,
            *extwatflux_ss_, *compute_, *output_, to_time(y, m, d, h, mi),
            ewf_name, dst, ix_r, iy_r, iz_r, wflux_m3);
    }

    void time_end(int y, int m, int d, int h, int mi) {
        require_init("time_end");
        couplercalls_->RunTimeLoopEnd(
            *hostcfg_, *json_, *wqconfig_, *units_, *utils_, *readjson_,
            *vars_, *initiate_, *TD_, *LE_, *CH_, *SI_, *TS_,
            *extwatflux_ss_, *compute_, *output_, to_time(y, m, d, h, mi));
    }

    // ------------------------------------------------------------------
    // State access
    // ------------------------------------------------------------------
    int num_species() {
        require_init("num_species");
        return (int) (*vars_->chemass)(0).n_elem;
    }

    double mass(int icmp, int ichem, int ix, int iy, int iz) {
        require_init("mass");
        check_cell(icmp, ix, iy, iz);
        check_chem(ichem);
        return (*vars_->chemass)(icmp)(ichem)(ix, iy, iz);
    }

    // mass of one species in one compartment as array [nx, ny, nz]
    py::array_t<double> mass_array(int icmp, int ichem) {
        require_init("mass_array");
        check_cmp(icmp);
        check_chem(ichem);
        const arma::Cube<double>& c = (*vars_->chemass)(icmp)(ichem);
        py::array_t<double> out({cmp_nx_[icmp], cmp_ny_[icmp], cmp_nz_[icmp]});
        auto r = out.mutable_unchecked<3>();
        for (int x = 0; x < cmp_nx_[icmp]; x++)
            for (int y = 0; y < cmp_ny_[icmp]; y++)
                for (int z = 0; z < cmp_nz_[icmp]; z++)
                    r(x, y, z) = c(x, y, z);
        return out;
    }

    // sum of a species over every compartment (domain mass)
    double total_mass(int ichem) {
        require_init("total_mass");
        check_chem(ichem);
        double tot = 0.0;
        for (int c = 0; c < (int) cmp_names_.size(); c++)
            tot += arma::accu((*vars_->chemass)(c)(ichem));
        return tot;
    }

    std::vector<std::string> compartment_names() const { return cmp_names_; }
    std::vector<std::string> ext_flux_names()    const { return ewf_names_; }
    std::vector<std::string> depend_names()      const { return dep_names_; }
    std::vector<std::string> flux_export_names() const { return fexp_names_; }

    int compartment_index(const std::string& name) const {
        for (int c = 0; c < (int) cmp_names_.size(); c++)
            if (cmp_names_[c] == name) return c;
        return -1;
    }

    std::vector<int> compartment_shape(int icmp) const {
        check_cmp(icmp);
        return {cmp_nx_[icmp], cmp_ny_[icmp], cmp_nz_[icmp]};
    }

    std::string output_dir() {
        require_init("output_dir");
        return wqconfig_->get_output_dir();
    }

    bool initialized() const { return initialized_; }

private:
    std::string master_json_;
    std::string cell_label_ = "cell";
    bool initialized_ = false;

    std::vector<std::string> cmp_names_, ewf_names_, dep_names_, fexp_names_;
    std::vector<int> cmp_nx_, cmp_ny_, cmp_nz_;
    std::vector<int> dep_nx_, dep_ny_, dep_nz_;
    std::vector<int> fexp_nx_, fexp_ny_, fexp_nz_;
    std::map<int, std::vector<std::string>> cell_ids_;

    std::unique_ptr<OpenWQ_hostModelconfig> hostcfg_   = std::make_unique<OpenWQ_hostModelconfig>();
    std::unique_ptr<OpenWQ_couplercalls>  couplercalls_ = std::make_unique<OpenWQ_couplercalls>();
    std::unique_ptr<OpenWQ_json>          json_          = std::make_unique<OpenWQ_json>();
    std::unique_ptr<OpenWQ_wqconfig>      wqconfig_      = std::make_unique<OpenWQ_wqconfig>();
    std::unique_ptr<OpenWQ_units>         units_         = std::make_unique<OpenWQ_units>();
    std::unique_ptr<OpenWQ_utils>         utils_         = std::make_unique<OpenWQ_utils>();
    std::unique_ptr<OpenWQ_readjson>      readjson_      = std::make_unique<OpenWQ_readjson>();
    std::unique_ptr<OpenWQ_initiate>      initiate_      = std::make_unique<OpenWQ_initiate>();
    std::unique_ptr<OpenWQ_extwatflux_ss> extwatflux_ss_ = std::make_unique<OpenWQ_extwatflux_ss>();
    std::unique_ptr<OpenWQ_compute>       compute_       = std::make_unique<OpenWQ_compute>();
    std::unique_ptr<OpenWQ_output>        output_        = std::make_unique<OpenWQ_output>();
    std::unique_ptr<OpenWQ_vars>          vars_;
    std::unique_ptr<OpenWQ_TD_model>      TD_ = std::make_unique<OpenWQ_TD_model>();
    std::unique_ptr<OpenWQ_LE_model>      LE_ = std::make_unique<OpenWQ_LE_model>();
    std::unique_ptr<OpenWQ_CH_model>      CH_ = std::make_unique<OpenWQ_CH_model>();
    std::unique_ptr<OpenWQ_SI_model>      SI_ = std::make_unique<OpenWQ_SI_model>();
    std::unique_ptr<OpenWQ_TS_model>      TS_ = std::make_unique<OpenWQ_TS_model>();

    time_t to_time(int y, int m, int d, int h, int mi) {
        return units_->convertTime_ints2time_t(*wqconfig_, y, m, d, h, mi, 0);
    }

    void require_init(const char* who) const {
        if (!initialized_)
            throw std::runtime_error(std::string(who) + "(): call init() first");
    }
    void require_not_init(const char* who) const {
        if (initialized_)
            throw std::runtime_error(std::string(who) + "(): not allowed after init()");
    }
    static void check_dims(const std::string& name, int nx, int ny, int nz) {
        if (nx < 1 || ny < 1 || nz < 1)
            throw std::invalid_argument(name + ": nx, ny, nz must be >= 1");
    }
    void check_cmp(int icmp) const {
        if (icmp < 0 || icmp >= (int) cmp_names_.size())
            throw std::out_of_range("bad compartment index " + std::to_string(icmp));
    }
    void check_dep(int idep) const {
        if (idep < 0 || idep >= (int) dep_names_.size())
            throw std::out_of_range("bad dependency index " + std::to_string(idep));
    }
    void check_chem(int ichem) {
        if (ichem < 0 || ichem >= num_species())
            throw std::out_of_range("bad species index " + std::to_string(ichem));
    }
    void check_cell(int icmp, int ix, int iy, int iz) const {
        check_cmp(icmp);
        if (ix < 0 || ix >= cmp_nx_[icmp] || iy < 0 || iy >= cmp_ny_[icmp] || iz < 0 || iz >= cmp_nz_[icmp]) {
            std::ostringstream s;
            s << "cell (" << ix << "," << iy << "," << iz << ") outside compartment "
              << cmp_names_[icmp] << " [" << cmp_nx_[icmp] << "," << cmp_ny_[icmp] << "," << cmp_nz_[icmp] << "]";
            throw std::out_of_range(s.str());
        }
    }
    static const double* checked_ptr(py::array_t<double, py::array::c_style | py::array::forcecast>& arr,
                                     int nx, int ny, int nz, const char* who) {
        if (arr.size() != (py::ssize_t) nx * ny * nz) {
            std::ostringstream s;
            s << who << ": array has " << arr.size() << " elements, expected "
              << nx << "*" << ny << "*" << nz;
            throw std::invalid_argument(s.str());
        }
        return arr.data();
    }
};

#endif // OPENWQ_HYDROLINK_PY_INCLUDED
