// OpenWQ hydrolink for Python host models (pybind11): module definition
// Copyright 2020, Diogo Costa, diogo.pinhodacosta@canada.ca
// This file is part of OpenWQ model (GNU GPL v3, see OpenWQ_hydrolink.h).

#include "OpenWQ_hydrolink.h"

PYBIND11_MODULE(openwq_py, m) {
    m.doc() = "OpenWQ water-quality model: generic Python host (forward coupling)";

    py::class_<OpenWQPyHost>(m, "Host")
        .def(py::init<const std::string&>(), py::arg("master_json"))
        .def("add_compartment", &OpenWQPyHost::add_compartment,
             py::arg("name"), py::arg("nx"), py::arg("ny") = 1, py::arg("nz") = 1)
        .def("add_ext_flux", &OpenWQPyHost::add_ext_flux,
             py::arg("name"), py::arg("nx"), py::arg("ny") = 1, py::arg("nz") = 1)
        .def("add_depend", &OpenWQPyHost::add_depend,
             py::arg("name"), py::arg("nx"), py::arg("ny") = 1, py::arg("nz") = 1)
        .def("add_flux_export", &OpenWQPyHost::add_flux_export,
             py::arg("name"), py::arg("src_compartment"), py::arg("nx"), py::arg("ny") = 1, py::arg("nz") = 1)
        .def("set_cell_label", &OpenWQPyHost::set_cell_label, py::arg("label"))
        .def("set_cell_ids", &OpenWQPyHost::set_cell_ids, py::arg("icmp"), py::arg("ids"))
        .def("init", &OpenWQPyHost::init)
        .def("set_water_volume", &OpenWQPyHost::set_water_volume,
             py::arg("icmp"), py::arg("ix"), py::arg("iy"), py::arg("iz"), py::arg("vol_m3"))
        .def("set_water_volume_array", &OpenWQPyHost::set_water_volume_array, py::arg("icmp"), py::arg("vol_m3"))
        .def("get_water_volume", &OpenWQPyHost::get_water_volume,
             py::arg("icmp"), py::arg("ix"), py::arg("iy"), py::arg("iz"))
        .def("set_depend", &OpenWQPyHost::set_depend,
             py::arg("idep"), py::arg("ix"), py::arg("iy"), py::arg("iz"), py::arg("value"))
        .def("set_depend_array", &OpenWQPyHost::set_depend_array, py::arg("idep"), py::arg("values"))
        .def("set_flux_volume", &OpenWQPyHost::set_flux_volume,
             py::arg("iexp"), py::arg("ix"), py::arg("iy"), py::arg("iz"), py::arg("vol_m3"))
        .def("time_start", &OpenWQPyHost::time_start,
             py::arg("year"), py::arg("month"), py::arg("day"), py::arg("hour") = 0, py::arg("minute") = 0)
        .def("move", &OpenWQPyHost::move,
             py::arg("src"), py::arg("ix_s"), py::arg("iy_s"), py::arg("iz_s"),
             py::arg("dst"), py::arg("ix_r"), py::arg("iy_r"), py::arg("iz_r"),
             py::arg("wflux_m3"), py::arg("wmass_src_m3"),
             py::arg("year"), py::arg("month"), py::arg("day"), py::arg("hour") = 0, py::arg("minute") = 0)
        .def("inflow", &OpenWQPyHost::inflow,
             py::arg("ewf_name"), py::arg("dst"), py::arg("ix_r"), py::arg("iy_r"), py::arg("iz_r"),
             py::arg("wflux_m3"),
             py::arg("year"), py::arg("month"), py::arg("day"), py::arg("hour") = 0, py::arg("minute") = 0)
        .def("time_end", &OpenWQPyHost::time_end,
             py::arg("year"), py::arg("month"), py::arg("day"), py::arg("hour") = 0, py::arg("minute") = 0)
        .def("num_species", &OpenWQPyHost::num_species)
        .def("mass", &OpenWQPyHost::mass,
             py::arg("icmp"), py::arg("ichem"), py::arg("ix"), py::arg("iy"), py::arg("iz"))
        .def("mass_array", &OpenWQPyHost::mass_array, py::arg("icmp"), py::arg("ichem"))
        .def("total_mass", &OpenWQPyHost::total_mass, py::arg("ichem"))
        .def("compartment_names", &OpenWQPyHost::compartment_names)
        .def("ext_flux_names", &OpenWQPyHost::ext_flux_names)
        .def("depend_names", &OpenWQPyHost::depend_names)
        .def("flux_export_names", &OpenWQPyHost::flux_export_names)
        .def("compartment_index", &OpenWQPyHost::compartment_index, py::arg("name"))
        .def("compartment_shape", &OpenWQPyHost::compartment_shape, py::arg("icmp"))
        .def("output_dir", &OpenWQPyHost::output_dir)
        .def("initialized", &OpenWQPyHost::initialized);
}
