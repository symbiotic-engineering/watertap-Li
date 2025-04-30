from watertap.ui.fsapi import FlowsheetInterface
from watertap.core.util.initialization import assert_degrees_of_freedom
from watertap.flowsheets.Configuration_2.Configuration_2 import (
    build,
    set_operating_conditions,
    initialize_system,
    solve,
    add_costing,
    ERDtype,
)
from pyomo.environ import units as pyunits, assert_optimial_termination
from pyomo.util.check_units import assert_units_consistent


def export_to_ui():
    return FlowsheetInterface(
        name="Configuration_2",
        do_export=export_variables,
        do_build=build_flowsheet,
        do_solve=solve_flowsheet,
 )
def export_variables(flowsheet=None, exports=None, build_options=None, **kwargs):
    fs = flowsheet
    # --- Input data ---
    # Feed conditions
    exports.add(
        obj=fs.feed.flow_vol[0],
        name="Volumetric flow rate",
        ui_units=pyunits.m**3 / pyunits.hr,
        display_units="m3/hr",
        rounding=0,
        description="Inlet volumetric flow rate",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed",
    )
    exports.add(
        obj=fs.feed.conc_mass_comp[0, "Li_+"],
        name="Lithium concentration",
        ui_units=pyunits.kg / pyunits.m**3,
        display_units="kg/m3",
        rounding=6,
        description="Inlet lithium concentration",
        is_input=True,
        input_category="Feed",
        is_ouput=True,
        output_category="Feed",

    )
    exports.add(
        obj=fs.feed.conc_mass_comp[0, "Mg_2+"],
        name="Magnesium concentration",
        ui_units=pyunits.kg / pyunits.m**3,
        display_units="kg/m3",
        rounding=6,
        description="Inlet magnesium concentration",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed",
    )
    exports.add(
        obj=fs.feed.conc_mass_comp[0, "Na_+"],
        name="Sodium concentration",
        ui_units=pyunits.kg / pyunits.m**3,
        display_units="kg/m3",
        rounding=5,
        description="Inlet sodium concentration",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed" 
    )
    exports.add(
        obj=fs.feed.conc_mass_comp[0, "Ca_2+"],
        name="Calcium concentration",
        ui_units=pyunits.kg / pyunits.m**3,
        display_units="kg/m3",
        rounding=2,
        description="Inlet calcium concentration",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed",
    )
    exports.add(
        obj=fs.feed.conc_mass_comp[0, "Cl_-"],
        name="Chlorine concentration",
        ui_units=pyunits.kg / pyunits.m**3,
        display_units="kg/m3",
        rounding=5,
        description="Inlet chlorine concentration",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed",
    )
    exports.add(
        obj=fs.feed.conc_mass_comp[0, "SO4_2-"],
        name="Sulfate concentration",
        ui_units=pyunits.kg / pyunits.m**3,
        display_units="kg/m3",
        rounding=5,
        description="Inlet sulfate concentration",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed",
    )
    exports.add(
        obj=fs.feed.conc_mass_comp[0, "H_+"],
        name="Natural hydrogen concentration",
        ui_units=pyunits.kg / pyunits.m**3,
        display_units="kg/m3",
        rounding=3,
        description="Inlet natural hydrogen concentration",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed", 
    )
    exports.add(
        obj=fs.feed.conc_mass_comp[0, "OH_-"],
        name="Natural hydroxide concentration",
        ui_units=pyunits.kg / pyunits.m**3,
        display_units="kg/m3",
        rounding=8,
        description="Inlet natural hydroxide concentration",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed",
    )
    exports.add(
        obj=fs.feed.conc_mass_comp[0, "tss"],
        name="Total suspended solids concentration",
        ui_units=pyunits.kg /pyunits.m**3,
        display_units="kg/m3",
        rounding=3,
        description="Inlet total suspended solids concentration",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed",
    )
    exports.add(
        obj=fs.feed.conc_mass_comp[0, "H2"],
        name="Hydrogen product",
        ui_units=pyunits.kg / pyunits.m**3,
        display_units="kg/m3",
        rounding=3,
        description="Natural hydrogen product",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed",
    )
    exports.add(
        obj=fs.feed.conc_mass_comp[0, "O2"],
        name="Oxygen product",
        ui_units=pyunits.kg / pyunits.m**3,
        display_units="kg/m3",
        rounding=4,
        description="Natural oxygen product",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed",
    )
    exports.add(
        obj=fs.tb_prtrt_Lix.properties_out[0].temperature,\
        name="Solution temperature",
        ui_units=pyunits.K,
        display_units="K",
        rounding=2,
        is_input=True,
        input_category="Feed",
    )
    #Unit model data, Lithium Extraction
    exports.add(
        obj=fs.Liextracton.P1.eta_motor,
        name="Li extracton pump motor efficiency",
        ui_units=pyunits.dimensionless,
        display_units="fraction", 
        rounding=2,
        description="Li extracton pump motor efficiency",
        is_input=True,
        input_category="Lithium Extraction",
        is_output=False,
    )
