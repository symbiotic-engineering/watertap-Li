import os
import idaes.logger as idaeslog
from pyomo.environ import (
    ConcreteModel,
    Objective,
    Expression,
    value,
    TransformationFactory,
    units as pyunits, 
    Block,
)
from pyomo.network import Arc, SequentialDecomposition
from pyomo.util.check_units import assert_units_consistent

from idaes.core import (
    FlowsheetBlock, 
    MomentumBalanceType, 
    UnitModelCostingBlock,
    UnitModelBlockData,
)
from watertap.core.solvers import get_solver
from idaes.core.util.initialization import (
    propagate_state,
    fix_state_vars,
    revert_state_vars,
)
from idaes.core.util.exceptions import ConfigurationError
from idaes.models.unit_models.translator import Translator
from idaes.models.unit_models import (
    Mixer, 
    Separator,
    Product,
)
from idaes.models.unit_models.mixer import MomentumMixingType

from idaes.models.unit_models.separator import (
    SplittingType,
    EnergySplittingType, 
)
import idaes.core.util.scaling as iscale
from watertap.core.util.initialization import assert_degrees_of_freedom, check_solve
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
    DiffusivityCalculation,
    MaterialBalanceType,
    ElectricalMobilityCalculation,
)
from watertap.core.wt_database import Database
import watertap.core.zero_order_properties as prop_ZO
from watertap.unit_models.zero_order import (
    FeedZO,
    SWOnshoreIntakeZO,
    ScreenZO,
    ChemicalAdditionZO,
    ChlorinationZO,
    StaticMixerZO,
    CoagulationFlocculationZO,
    SedimentationZO,
    AirFlotationZO,
    FixedBedZO,
    StorageTankZO,
    IonExchangeZO,
    TriMediaFiltrationZO,
    BackwashSolidsHandlingZO,
    CartridgeFiltrationZO,
    GACZO,
    LandfillZO,
)
from watertap.unit_models.ion_exchange_0D import (
    IsothermType,
    IonExchange0D,
)
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.unit_models.pressure_exchanger import PressureExchanger
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice

from watertap.unit_models.electrolyzer import Electrolyzer

from watertap.costing.zero_order_costing import ZeroOrderCosting
from watertap.costing import WaterTAPCosting


# Set up logger
_log = idaeslog.getLogger(__name__)

def build_flowsheet(erd_type1=None, erd_type2=None, elec_type=None):
    m = build(erd_type1=erd_type1,erd_type2=erd_type2,elec_type=elec_type)
    set_operating_conditions(m)
    assert_degrees_of_freedom(m, 0)
    return m

def solve_flowsheet(flowsheet=None):
    m = flowsheet.parent_block() 
    initialize_system(m)
    assert_degrees_of_freedom(m, 0)
    optimize_operation(m) #unfixes specific variables for cost optimization
    solve(m, checkpoint="solve flowsheet after initializing system")
    display_results(m)
    add_costing(m)
    initialize_costing(m)
    assert_degrees_of_freedom(m, 0)
    solve(m, checkpoint="solve flowsheet with costing")

def main(erd_type1="pressure_exchanger", erd_type2="pressure_exchanger", elec_type= "PEM"):
    m = build_flowsheet(erd_type1=erd_type1, erd_type2=erd_type2, elec_type=elec_type)

    initialize_system(m)
    assert_degrees_of_freedom(m, 0)

    solve(m, checkpoint=f" solve flowsheet after initializing {erd_type1, erd_type2, elec_type} system")
    display_results(m)
    
    add_costing(m)
    initialize_costing(m)
    assert_degrees_of_freedom(m,0) #ensure problem is square 

    optimize_operation(m) #unfixes specific variables for cost optimization

    solve(m, tee=True, checkpoint=f" solve {erd_type1, erd_type2, elec_type} flowsheet with costing")
    display_costing(m)

    return m

def build(erd_type1=None, erd_type2=None, elec_type=None):
    # flowsheet set up
    m = ConcreteModel()
    m.db = Database()
    m.erd_type1 = erd_type1
    m.erd_type2 = erd_type2
    m.elec_type = elec_type

    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.prop_prtrt = prop_ZO.WaterParameterBlock(solute_list=["Li_+","Ca_2+","Mg_2+","Cl_-","SO4_2-","Na_+","H_+","OH_-","O2-v","H2-v","tss"])
    m.fs.prop_Lixstor = prop_ZO.WaterParameterBlock(solute_list=["Li_+"])
    m.fs.prop_psttrt = prop_ZO.WaterParameterBlock(solute_list=["tds","H_+", "OH_-","O2-v","H2-v"])
    m.fs.prop_H2stor = prop_ZO.WaterParameterBlock(solute_list=["H_+","OH_-","O2-v","H2-v"])
    m.fs.prop_od = MCASParameterBlock(material_flow_basis="mass",
                                        ignore_neutral_charge=True,
                                        solute_list=["Li_+","Ca_2+","Mg_2+","Cl_-","SO4_2-","Na_+","H_+","OH_-","O2-v","H2-v"],
                                        diffusivity_data={
                                            ("Liq", "Li_+"): 1.03e-9,
                                            ("Liq", "Ca_2+"): 7.93e-10,
                                            ("Liq", "Mg_2+"): 7.05e-10,
                                            ("Liq", "Cl_-"): 2.03e-9,
                                            ("Liq", "SO4_2-"): 1.07e-9,
                                            ("Liq", "Na_+"): 1.33e-9,
                                            ("Liq", "H_+"): 9.13e-9,
                                            ("Liq", "OH_-"): 5.27e-9,
                                            ("Liq", "O2-v"): 2e-9,
                                            ("Liq", "H2-v"): 4.58e-9
                                        },
                                        mw_data={
                                            "H2O": 0.018, 
                                            "Li_+": 0.007, 
                                            "Ca_2+": 0.040, 
                                            "Mg_2+": 0.024,
                                            "Cl_-": 0.035,
                                            "SO4_2-": 0.096,
                                            "Na_+": 0.023,
                                            "H_+": 0.001,
                                            "OH_-": 0.017,
                                            "O2-v": 0.032,
                                            "H2-v":0.002
                                        },
                                        elec_mobility_data={
                                            "Li_+": 4.08e-8,
                                            "Ca_2+": 6.28e-8, 
                                            "Mg_2+": 5.58e-8,
                                            "Cl_-": 8.04e-8,
                                            "SO4_2-": 8.47e-8,
                                            "Na_+": 5.26e-8,
                                            "H_+": 3.68e-7 ,
                                            "OH_-": 2.05e-7
                                        },
                                        charge={
                                            "Li_+": 1,
                                            "Ca_2+": 2,
                                            "Mg_2+": 2,
                                            "Cl_-": -1,
                                            "SO4_2-": -2,
                                            "Na_+": 1,
                                            "H_+": 1,
                                            "OH_-":-1
                                        },
                                        diffus_calculation=DiffusivityCalculation.none,
                                        elec_mobility_calculation=ElectricalMobilityCalculation.none,
                                        )
    # Build the state block and specify a time (0 = steady state).
    m.fs.state_block = m.fs.prop_od.build_state_block([0])

    # Specify the state variables of the stream. Note, now we specify mass flowrate (`flow_mass_phase_comp`) instead of molar flowrate (`flow_mol_phase_comp`).
    m.fs.state_block[0].flow_mass_phase_comp["Liq", "Li_+"].fix(0.00017)
    m.fs.state_block[0].flow_mass_phase_comp["Liq", "Ca_2+"].fix(0.04)
    m.fs.state_block[0].flow_mass_phase_comp["Liq", "Mg_2+"].fix(0.00128)
    m.fs.state_block[0].flow_mass_phase_comp["Liq", "Cl_-"].fix(0.0194)
    m.fs.state_block[0].flow_mass_phase_comp["Liq", "SO4_2-"].fix(0.0027)
    m.fs.state_block[0].flow_mass_phase_comp["Liq", "Na_+"].fix(0.0108)
    m.fs.state_block[0].flow_mass_phase_comp["Liq", "H_+"].fix(0.06)
    m.fs.state_block[0].flow_mass_phase_comp["Liq", "OH_-"].fix(0.0000214)
    m.fs.state_block[0].flow_mass_phase_comp["Liq", "O2-v"].fix(0.477)
    m.fs.state_block[0].flow_mass_phase_comp["Liq", "H2-v"].fix(0.12)
    m.fs.state_block[0].flow_mass_phase_comp["Liq", "H2O"].fix(0.926)
    m.fs.state_block[0].pressure.fix(101325)
    m.fs.state_block[0].temperature.fix(293.15)

    # "Touch" build-on-demand variables so that they are created
    m.fs.state_block[0].flow_mass_phase_comp
    m.fs.state_block[0].flow_mol_phase_comp
    m.fs.state_block[0].conc_mass_phase_comp
    m.fs.state_block[0].flow_vol_phase
    m.fs.state_block[0].molality_phase_comp
    m.fs.state_block[0].conc_mass_phase_comp
    m.fs.state_block[0].total_hardness
    m.fs.state_block[0].ionic_strength_molal
    m.fs.state_block[0].pressure_osm_phase


    density= 1023.5 * pyunits.kg/ pyunits.m**3
    m.fs.prop_prtrt.dens_mass_default = density

    #block structure
    prtrt = m.fs.pretreatment = Block()
    Lix = m.fs.Liextraction = Block()
    Lixstor =m.fs.Listorage = Block()
    desal = m.fs.desalination = Block()
    psttrt = m.fs.posttreatment = Block()
    H2x = m.fs.H2extraction = Block()
    H2xstor = m.fs.H2storage = Block()

    #Unit models
    m.fs.feed = FeedZO(property_package=m.fs.prop_prtrt)

    #Pretreatment
    prtrt.intake = SWOnshoreIntakeZO(property_package=m.fs.prop_prtrt)
    prtrt.ferric_chloride_addition = ChemicalAdditionZO(property_package=m.fs.prop_prtrt, database=m.db, process_subtype="ferric_chloride")
    prtrt.chlorination = ChlorinationZO(property_package=m.fs.prop_prtrt, database=m.db)
    prtrt.static_mixer = StaticMixerZO(property_package=m.fs.prop_prtrt, database=m.db)
    prtrt.storage = StorageTankZO(property_package=m.fs.prop_prtrt, database=m.db)
    prtrt.screening = ScreenZO(property_package=m.fs.prop_prtrt, database=m.db)
    prtrt.coag_and_floc = CoagulationFlocculationZO(
        property_package=m.fs.prop_prtrt, database=m.db
    )
    prtrt.sedimentation = SedimentationZO(property_package=m.fs.prop_prtrt, database=m.db)
    prtrt.flotation = AirFlotationZO(property_package=m.fs.prop_prtrt, database=m.db, )
    prtrt.gravity_basin = FixedBedZO(
        property_package=m.fs.prop_prtrt, database=m.db, process_subtype="gravity_basin"
    )
    prtrt.mfiltration = TriMediaFiltrationZO(property_package=m.fs.prop_prtrt, database=m.db)
    prtrt.mbackwash_pump =BackwashSolidsHandlingZO(property_package=m.fs.prop_prtrt, database=m.db)
    prtrt.anti_scalant_addition = ChemicalAdditionZO(property_package=m.fs.prop_prtrt, database=m.db, process_subtype="anti-scalant")
    prtrt.cfiltration = CartridgeFiltrationZO(property_package=m.fs.prop_prtrt, database=m.db)
    prtrt.gac = GACZO(property_package=m.fs.prop_prtrt, database=m.db, process_subtype="pressure_vessel")
    prtrt.gbackwash_pump = BackwashSolidsHandlingZO(property_package=m.fs.prop_prtrt, database=m.db)

    prtrt.mf_disposal = LandfillZO(property_package=m.fs.prop_prtrt, database=m.db)
    prtrt.gac_disposal = LandfillZO(property_package=m.fs.prop_prtrt, database=m.db)

    #Li extraction
    Lix.P1 = Pump(property_package=m.fs.prop_od)
    isotherm = IsothermType.langmuir
    target_ion = "Li_+"
    regenerant = "HCl"
    hazardous_waste = True
    ix_config = {
        "property_package": m.fs.prop_od,
        "isotherm": isotherm,
        "target_ion": target_ion,
        "hazardous_waste": hazardous_waste,
        "regenerant": regenerant,
    }
    Lix.IX= IonExchange0D(**ix_config)
    Lix.regen = Product(property_package=m.fs.prop_od)
    
    #Touch Li extraction unit properties so they are available for scaling, initializing and reporting
    Lix.IX.process_flow.properties_in[0].conc_mass_phase_comp[...]
    Lix.IX.process_flow.properties_out[0].conc_mass_phase_comp[...]
    Lix.IX.regeneration_stream[0].conc_mass_phase_comp[...]

    #Li Storage Unit
    Lixstor.storage = StorageTankZO(property_package=m.fs.prop_Lixstor, database=m.db)
    
    #Desalination (1st Pass)
    desal.FP_P1 = Pump(property_package=m.fs.prop_od)
    desal.FP_RO = ReverseOsmosis0D(
        property_package=m.fs.prop_od,
        has_pressure_change=True,
        pressure_change_type=PressureChangeType.calculated,
        mass_transfer_coefficient=MassTransferCoefficient.calculated,
        concentration_polarization_type=ConcentrationPolarizationType.calculated,
    )
    desal.FP_RO.width.setub(5000)
    desal.FP_RO.area.setub(20000)
    if erd_type1 == "pressure_exchanger":
        desal.FP_S1 = Separator(property_package=m.fs.prop_od, outlet_list=["FP_P1", "FP_PXR"])
        desal.FP_M1 = Mixer(property_package=m.fs.prop_od, momentum_mixing_type=MomentumMixingType.equality,
                         inlet_list=["FP_P1", "FP_P2"])
        desal.FP_PXR = PressureExchanger(property_package=m.fs.prop_od)
        desal.FP_P2 = Pump(property_package=m.fs.prop_od)
    elif erd_type1 == "pump_as_turbine":
        desal.FP_ERD = EnergyRecoveryDevice(property_package=m.fs.prop_od)
    else:
        raise ConfigurationError(
            "erd_type1 was {}, but can only"
            "be pressure_exchanger or pump_as_turbine"
            "".format(erd_type1)
        )
    desal.FP_disposal = Product(property_package=m.fs.prop_od)

    
    #Desalination (2nd Pass)
    desal.SP_P1 = Pump(property_package=m.fs.prop_od)
    desal.SP_RO = ReverseOsmosis0D(
        property_package=m.fs.prop_od,
        has_pressure_change=True,
        pressure_change_type=PressureChangeType.calculated,
        mass_transfer_coefficient=MassTransferCoefficient.calculated,
        concentration_polarization_type=ConcentrationPolarizationType.calculated,
    )
    if erd_type2 == "pressure_exchanger":
        desal.SP_S1 = Separator(property_package=m.fs.prop_od, outlet_list=["SP_P1", "SP_PXR"])
        desal.SP_M1 = Mixer(property_package=m.fs.prop_od, momentum_mixing_type=MomentumMixingType.equality,
                            inlet_list=["SP_P1", "SP_P2"])
        desal.SP_PXR = PressureExchanger(property_package=m.fs.prop_od)
        desal.SP_P2 = Pump(property_package=m.fs.prop_od)
    elif erd_type2 == "pump_as_turbine":
        desal.FP_ERD = EnergyRecoveryDevice(property_package=m.fs.prop_od)
    else:
        raise ConfigurationError(
            "erd_type2 was {}, but can only"
            "be pressure_exchanger or pump_as_turbine"
            "".format(erd_type2)
        )
    desal.SP_disposal = Product(property_package=m.fs.prop_od)
    
    #Demineralization 
    psttrt.IX = IonExchangeZO(property_package=m.fs.prop_zo, database=m.db, process_subtype="demineralization")
    psttrt.storage = StorageTankZO(property_package=m.fs.prop_zo, database=m.db)

    #H2 Extraction
    if elec_type == "PEM":
        H2x.ER = Electrolyzer(property_package=m.fs.prop_od)
        #Touch electrolysis properties
        H2x.ER.anolyte.properties_in[0].flow_vol_phase
        H2x.ER.anolyte.properties_in[0].conc_mass_phase_comp
        H2x.ER.anolyte.properties_in[0].conc_mol_phase_comp
        H2x.ER.catholyte.properties_in[0].flow_vol_phase
        H2x.ER.catholyte.properties_in[0].conc_mass_phase_comp
        H2x.ER.catholyte.properties_in[0].conc_mol_phase_comp
        H2x.ER.anolyte.properties_out[0].flow_vol_phase
        H2x.ER.anolyte.properties_out[0].conc_mass_phase_comp
        H2x.ER.anolyte.properties_out[0].conc_mol_phase_comp
        H2x.ER.catholyte.properties_out[0].flow_vol_phase
        H2x.ER.catholyte.properties_out[0].conc_mass_phase_comp
        H2x.ER.catholyte.properties_out[0].conc_mol_phase_comp


        H2x.Compressor_O2 = PressureExchanger(
            property_package=m.fs.prop_od,
            compressor = True,
        )
        H2x.Compressor_H2 = PressureExchanger(
            property_package=m.fs.prop_od,
            compressor = True,

        )
        H2x.prod_O2 = Product(property_package=m.fs.prop_od)
        H2x.prod_O2.properties[0].conc_mass_phase_comp[...]
        H2x.prod_H2 = Product(property_package=m.fs.prop_od)
        H2x.prod_H2.properties[0].conc_mass_phase_comp[...]
    elif elec_type == "AWE":
        H2x.ER = Electrolyzer(property_package=m.fs.prop_od)
        #Touch electrolysis properties
        H2x.ER.anolyte.properties_in[0].flow_vol_phase
        H2x.ER.anolyte.properties_in[0].conc_mass_phase_comp
        H2x.ER.anolyte.properties_in[0].conc_mol_phase_comp
        H2x.ER.catholyte.properties_in[0].flow_vol_phase
        H2x.ER.catholyte.properties_in[0].conc_mass_phase_comp
        H2x.ER.catholyte.properties_in[0].conc_mol_phase_comp
        H2x.ER.anolyte.properties_out[0].flow_vol_phase
        H2x.ER.anolyte.properties_out[0].conc_mass_phase_comp
        H2x.ER.anolyte.properties_out[0].conc_mol_phase_comp
        H2x.ER.catholyte.properties_out[0].flow_vol_phase
        H2x.ER.catholyte.properties_out[0].conc_mass_phase_comp
        H2x.ER.catholyte.properties_out[0].conc_mol_phase_comp


        H2x.Compressor_O2 = PressureExchanger(
            property_package=m.fs.prop_od,
            compressor = True,
        )
        H2x.Compressor_H2 = PressureExchanger(
            property_package=m.fs.prop_od,
            compressor = True,

        )
        H2x.prod_O2 = Product(property_package=m.fs.prop_od)
        H2x.prod_O2.properties[0].conc_mass_phase_comp[...]
        H2x.prod_H2 = Product(property_package=m.fs.prop_od)
        H2x.prod_H2.properties[0].conc_mass_phase_comp[...]  

    else: 
        pass 

    #Gas Storage
    H2xstor.O2 = StorageTankZO(property_package=m.fs.prop_zo, database=m.db, process_subtype="default")
    H2xstor.H2 = StorageTankZO(property_package=m.fs.prop_zo, database=m.db, process_subtype="default")

    #Translator Blocks

    #Pretreatment --> Li extraction
    m.fs.tb_prtrt_Lix = Translator(
        inlet_property_package=m.fs.prop_prtrt, outlet_property_package=m.fs.prop_od
    )
    @m.fs.tb_prtrt_Lix.Constraint(["H2O","Li_+","Ca_2+","Mg_2+","Cl_-","SO4_2-","Na_+","H_+","OH_-","O2-v","H2-v","tss"])
    def eq_flow_mass_comp(blk, j):
        if j == "tss":
            return ( 
                blk.properties_in[0].flow_mass_comp["tss"]
                + blk.properties_in[0].flow_mass_comp["Na_+"]
                == blk.properties_out[0].flow_mass_phase_comp["Liq", "Na_+"]  
            )
        else: 
            jj = j
            return (
                blk.properties_in[0].flow_mass_comp[j]
                == blk.properties_out[0].flow_mass_phase_comp["Liq", jj]
            )
    
    #Li extraction to Li storage 
    m.fs.tb_Lix_Lixstor = Translator(
        inlet_property_package=m.fs.prop_od, outlet_property_package=m.fs.prop_Lixstor
    )
    @m.fs.tb_Lix_Lixstor.Constraint(["H2O","Li_+","Ca_2+","Mg_2+","Cl_-","SO4_2-","Na_+","H_+","OH_-","O2-v","H2-v"])
    def eq_flow_mass_comp(blk, j):
        if j == "Li_+":
            return (
                blk.properties_in[0].flow_mass_phase_comp["Liq", "Li_+"]
                == blk.properties_out[0].flow_mass_comp["Li_+"]
            )
        else: 
           jj = j 
           return (
               blk.properties_in[0].flow_mass_phase_comp["Liq", j]
               == blk.properties_out[0].flow_mass_comp[jj]
           )
    
    #Li storage to Desalination
    m.fs.tb_Lixstor_desal = Translator(
        inlet_property_package=m.fs.prop_Lixstor, outlet_property_package=m.fs.prop_od
    )
    @m.fs.tb_Lixstor_desal.Constraint(["H2O","Li_+","Ca_2+","Mg_2+","Cl_-","SO4_2-","Na_+","H_+","OH_-","O2-v","H2-v"])
    def eq_flow_mass_comp(blk, j):
        if j == "Li_+":
            return (
                blk.properties_in[0].flow_mass_comp["Li_+"]
                == blk.properties_out[0].flow_mass_phase_comp["Liq", "Li_+"]
            )
        else: 
            jj = j 
            return (
                blk.properties_in[0].flow_mass_comp[j]
                == blk.properties_out[0].flow_mass_phase_comp["Liq", jj]
            )
    #Desalination --> Posttreatment
    m.fs.tb_desal_psttrt = Translator(
        inlet_property_package=m.fs.prop_od, outlet_property_package=m.fs.prop_psttrt
    )
    @m.fs.tb_desal_psttrt.Constraint(["H2O","Li_+","Ca_2+","Mg_2+","Cl_-","SO4_2-","Na_+","H_+","OH_-","O2-v","H2-v", "tds"])
    def eq_flow_mass_comp(blk, j):
        if j == "Li_+" or "Mg_2+" or "Cl_-" or "SO4_2-" or "Na_+":
            return (
                blk.properties_in[0].flow_mass_phase_comp["Liq", "Li_+"]
                + blk.properties_in[0].flow_mass_phase_comp["Liq", "Ca_2+"]
                + blk.properties_in[0].flow_mass_phase_comp["Liq", "Mg_2+"]
                + blk.properties_in[0].flow_mass_phase_comp["Liq", "Cl_-"]
                + blk.properties_in[0].flow_mass_phase_comp["Liq", "SO4_2-"]
                + blk.properties_in[0].flow_mass_phase_comp["Liq", "Na_+"]
                == blk.properties_out[0].flow_mass_comp["tds"]
            )
        else:
            jj = j 
            return (
            blk.properties_in[0].flow_mass_phase_comp["Liq", j]
            == blk.properties_out[0].flow_mass_comp[jj]
            )

    #Posttreatment --> H2 Extraction 
    m.fs.tb_psttrt_H2x = Translator(
        inlet_property_package=m.fs.prop_psttrt, outlet_property_package=m.fs.prop_od
    )
    @m.fs.tb_psttrt_H2x.Constraint(["H2O","tds","H_+", "OH_-","O2-v","H2-v"])
    def eq_flow_mass_comp(blk, j):
        if j == "tds":
            pass 
        else: 
            jj = j 
        return (
            blk.properties_in[0].flow_mass_comp[j]
            == blk.properties_out[0].flow_mass_phase_comp[jj]
        )
    
    #H2 Extraction --> Gas Storage
    m.fs.tb_H2x_H2xstor = Translator(
        inlet_property_package=m.fs.prop_od, outlet_property_package=m.fs.prop_H2stor
    )
    @m.fs.tb_H2x_H2xstor.Constraint(["H2-v", "O2-v"])
    def eq_flow_mass_comp(blk, j):
        if j == "H2-v":
            return (
                blk.properties_in[0].flow_mass_phase_comp["Liq", "H2-v"]
                == blk.properties_out[0].flow_mass_comp["H2-v"]
            )
        elif j == "O2-v":
            return (
                blk.properties_in[0].flow_mass_phase_comp["Liq", "O2-v"]
                == blk.properties_out[0].flow_mass_comp["O2-v"]
            )
        else:
            pass

    #Connections
    #Pretreatment
    m.fs.s_feed = Arc(source=m.fs.feed.outlet, destination=prtrt.intake.inlet)
    prtrt.s01 = Arc(sourced=prtrt.intake.outlet, destination=prtrt.ferric_chloride_addition.inlet)
    prtrt.s02 = Arc(source=prtrt.ferric_chloride_addition.outlet, destination=prtrt.chlorination.inlet)
    prtrt.s03 = Arc(source=prtrt.chlorination.outlet , destination=prtrt.static_mixer.inlet)
    prtrt.s04 = Arc(source=prtrt.static_mixer.outlet, destination=prtrt.storage.inlet)
    prtrt.s05 = Arc(source=prtrt.storage.outlet, destination=prtrt.screening.inlet)
    prtrt.s06 = Arc(source=prtrt.screening.treated, destination=prtrt.coag_and_floc.inlet)
    prtrt.s06 = Arc(source=prtrt.coag_and_floc.outlet, destination=prtrt.sedimentation.inlet)
    prtrt.s07 = Arc(source=prtrt.sedimentation.treated, destination=prtrt.flotation.inlet)
    prtrt.s08 = Arc(source=prtrt.flotation.treated, destination=prtrt.gravity_basin.inlet)
    prtrt.s09 = Arc(source=prtrt.gravity_basin.treated, destination=prtrt.mfiltration.inlet)
    prtrt.s10 = Arc(source=prtrt.mfiltration.byproduct, destination=prtrt.mbackwash_pump.inlet)
    prtrt.s11 = Arc(source=prtrt.mfiltration.treated, destination=prtrt.anti_scalant_addition.inlet)
    prtrt.s12 = Arc(source=prtrt.anti_scalant_addition.outlet, destination=prtrt.cfiltration.inlet)
    prtrt.s13 = Arc(source=prtrt.cfiltration.treated, destination=prtrt.gac.inlet)
    prtrt.s14 = Arc(source=prtrt.gac.byproduct, destination=m.fs.gbackwash_pump.inlet)
   
    m.fs.mlandfill = Arc(source=prtrt.mbackwash_pump.byproduct, destination=prtrt.mf_disposal.inlet)
    m.fs.glandfill = Arc(source=prtrt.gbackwash_pump.byproduct, destination=prtrt.gac_disposal.inlet)  

   #Pretreatment --> Lithium extraction
    m.fs.s_prtrt_tb = Arc(source=prtrt.gac.treated, destination=m.fs.tb_prtrt_Lix.inlet)

    #Lithium Extraction
    m.fs.s_tb_Lix = Arc(source=m.fs.tb_prtrt_Lix.outlet, destination=Lix.P1.inlet)
    Lix.s01 = Arc(source=Lix.P1.outlet, destination=Lix.IX.inlet)
    Lix.s02 = Arc(source=Lix.IX.regen, destination=Lix.regen.inlet)

    #Lithium Extraction --> Lithium Storage
    m.fs.s_Lix_tb = Arc(source=Lix.IX.outlet, destination=m.fs.tb_Lix_Lixstor.inlet)

    #Lithium Storage (Theoretical for Costing)
    m.fs.s_tb_Lixstor = Arc(source=m.fs.tb_Lix_Lixstor.outlet, destination=Lixstor.stor.inlet)


    #Lithium Storage to Desalination
    m.fs.s_Lixstor_tb = Arc(source=Lixstor.storage.outlet, destination=m.fs.tb_Lixstor_desal.inlet)

    #First Pass Desalination 
    if erd_type1 == "pressure_exchanger":
       
        m.fs.s_tb_desal = Arc(source=m.fs.tb_Lixstor_desal.outlet, destination=desal.FP_S1.inlet)
        desal.s01 = Arc(source=desal.FP_S1.FP_P1, destination=desal.FP_P1.inlet)
        desal.s02 = Arc(source=desal.FP_P1.outlet, destination=desal.FP_M1.FP_P1)
        desal.s03 = Arc(source=desal.FP_M1.outlet, destination=desal.FP_RO.inlet)
        desal.s04 = Arc(source=desal.FP_RO.retentate, destination=desal.FP_PXR.brine_inlet)
        desal.s05 = Arc(source=desal.FP_S1.FP_PXR, destination=desal.FP_PXR.feed_inlet)
        desal.s06 = Arc(source=desal.FP_PXR.feed_outlet, destination=desal.FP_P2.inlet)
        desal.s07 = Arc(source=desal.FP_P2.outlet, destination=desal.FP_M1.FP_P2)
        m.fs.FP_s_disposal = Arc(
            source=desal.FP_PXR.brine_outlet, destination=desal.FP_disposal.inlet
        )
    elif erd_type1 == "pump_as_turbine":
        m.fs.s_tb_desal = Arc(source=m.fs.tb_Lixstor_desal.outlet, destination=desal.FP_P1.inlet)
        desal.s01 = Arc(source=desal.FP_P1.outlet, destination=desal.FP_RO.inlet)
        desal.s02 = Arc(source=desal.FP_RO.retentate, destination=desal.FP_ERD.inlet)
        m.fs.FP_s_disposal = Arc(
            source=desal.ERD.outlet, destination=desal.FP_disposal.inlet
        )
    
    #Second Pass Desalination
    if erd_type2 == "pressure_exchanger" and erd_type2 == "pressure_exchanger":
        desal.s09 = Arc(source=desal.FP_RO.permeate, destination=desal.SP_S1.inlet)
        desal.s10 = Arc(source=desal.SP_S1.SP_P1, destination=desal.SP_P1.inlet)
        desal.s11 = Arc(source=desal.SP_P1.outlet, destination=desal.SP_M1.SP_P1)
        desal.s12 = Arc(source=desal.SP_M1.outlet, destination=desal.SP_RO.inlet)
        desal.s13 = Arc(source=desal.SP_RO.retentate, destination=desal.SP_PXR.brine_inlet)
        desal.s14 = Arc(source=desal.SP_S1.SP_PXR, destination=desal.SP_PXR.feed_inlet)
        desal.s15 = Arc(source=desal.SP_PXR.feed_outlet, destination=desal.SP_P2.inlet)
        desal.s16 = Arc(source=desal.SP_P2.outlet, destination=desal.SP_M1.SP_P2)
        m.fs.SP_s_disposal = Arc(
            source=desal.SP_PXR.brine_outlet, destination=desal.SP_disposal.inlet
        )
    elif erd_type2 == "pressure_exchanger" and erd_type2 == "pump_as_turbine":
        desal.s09 = Arc(source=desal.FP_RO.permeate, destination=desal.SP_P1.inlet)
        desal.s10 = Arc(source=desal.SP_P1.outlet, destination=desal.SP_RO.inlet)
        desal.s11 = Arc(source=desal.SP_RO.retentate, destination=desal.SP_ERD.inlet)
        m.fs.SP_s_disposal = Arc(
            source=desal.SP_ERD.outlet, destination=desal.SP_disposal.inlet
        )
    elif erd_type2 == "pump_as_turbine" and erd_type2 == "pressure_exchanger":
        desal.s04 = Arc(source=desal.FP_RO.permeate, destination=desal.SP_S1.inlet)
        desal.s05 = Arc(source=desal.SP_S1.SP_P1, destination=desal.SP_P1.inlet)
        desal.s06 = Arc(source=desal.SP_P1.outlet, destination=desal.SP_M1.SP_P1)
        desal.s08 = Arc(source=desal.SP_M1.outlet, destination=desal.SP_RO.inlet)
        desal.s09 = Arc(source=desal.SP_RO.retentate, destination=desal.SP_PXR.brine_inlet)
        desal.s10 = Arc(source=desal.SP_S1.SP_PXR, destination=desal.SP_PXR.feed_inlet)
        desal.s11 = Arc(source=desal.SP_PXR.feed_outlet, destination=desal.SP_P2.inlet)
        desal.s12 = Arc(source=desal.SP_P2.outlet, destination=desal.SP_M1.SP_P2)
        m.fs.SP_s_disposal = Arc (
            source=desal.SP_PXR.brine_outlet, destination=desal.SP_disposal.inlet
        )
    elif erd_type2 == "pump_as_turbine" and erd_type2 == "pump_as_turbine":
       desal.s04 = Arc(source=desal.FP_RO.permeate, destination=desal.SP_P1.inlet)
       desal.s05 = Arc(source=desal.SP_P1.outlet, destination=desal.SP_RO.inlet)
       desal.s06 = Arc(source=desal.SP_RO.retentate, destination=desal.SP_ERD.inlet) 
       m.fs.SP_s_disposal = Arc(
           source=desal.SP_ERD.outlet, destination=desal.SP_disposal.inlet
       )

    #Desalination --> Posttreatment
    m.fs.s_desal_tb = Arc(source=desal.SP_RO.permeate, destination=m.fs.tb_desal_psttrt.inlet)
    
    #Posttreatment
    m.fs.s_tb_psttrt = Arc(source=m.fs.tb_desal_psttrt.outlet, destination=psttrt.IX.inlet)
    psttrt.s01 = Arc(source=psttrt.IX.outlet, destination=psttrt.storage.inlet)

    #Posttreatment to H2 Extraction
    m.fs.s_psttrt_tb = Arc(source=psttrt.storage.outlet, destination=m.fs.tb_psttrt_H2x.inlet)

    #H2 Extraction
    m.fs.s_tb_H2 = Arc(source=m.fs.tb_psttrt_H2x.outlet, destination=H2x.ER.catholyte_inlet)
    m.fs.s_tb_O2= Arc(source=m.fs.tb_psttrt_H2x.outlet, destination=H2x.ER.anolyte_inlet )
    H2x.s01 = Arc(source=H2x.ER.catholyte_outlet, destination=H2x.Compressor_H2.inlet)
    H2x.s02 = Arc(source=H2x.ER.anolyte_outlet, destination=H2x.Compressor_O2.inlet)
    H2x.s03 = Arc(source=H2x.Compressor_H2.outlet, destination=H2x.prod_H2.inlet)
    H2x.s04 = Arc(source=H2x.Compressor_O2.outlet, destination=H2x.prod_O2.inlet)

    #H2 Extraction --> Gas Storage
    m.fs.s_H2_tb = Arc(source=H2x.prod_H2.inlet, destination=m.fs.tb_H2x_H2xstor.inlet)
    m.fs.s_O2_tb = Arc(source=H2x.prod_O2.inlet, destination=m.fs.tb_H2x_H2xstor.inlet)

    #Gas Storage 
    m.fs.s_tb_H2xstor_H2= Arc(source=m.fs.tb_H2x_H2xstor.outlet, destination=H2xstor.H2.inlet)
    m.fs_s_tb_H2xstor_O2 = Arc(source=m.fs.tb_H2x_H2xstor.outlet, destination=H2xstor.O2.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    #scaling
    #Set default property values
    m.fs.prop_od.set_default_scaling("flow_mass_phase_comp", 1e4, index=("Liq", "Li_+"))
    m.fs.prop_od.set_default_scaling("flow_mass_phase_comp", 1e2, index=("Liq", "Ca_2+"))
    m.fs.prop_od.set_default_scaling("flow_mass_phase_comp", 1e3, index=("Liq", "Mg_2+"))
    m.fs.prop_od.set_default_scaling("flow_mass_phase_comp", 1e2, index=("Liq", "Cl_-"))
    m.fs.prop_od.set_default_scaling("flow_mass_phase_comp", 1e3, index=("Liq", "SO4_2-"))
    m.fs.prop_od.set_default_scaling("flow_mass_phase_comp", 1e2, index=("Liq", "Na_+"))
    m.fs.prop_od.set_default_scaling("flow_mass_phase_comp", 1e2, index=("Liq", "H_+"))
    m.fs.prop_od.set_default_scaling("flow_mass_phase_comp", 1e5, index=("Liq", "OH_-"))
    m.fs.prop_od.set_default_scaling("flow_mass_phase_comp", 10, index=("Liq", "O2-v"))
    m.fs.prop_od.set_default_scaling("flow_mass_phase_comp", 10, index=("Liq", "H2-v"))
    m.fs.prop_od.set_default_scaling("flow_mass_phase_comp", 10, index=("Liq", "H2O"))

    #For H2x, may not be able to do this...
    m.fs.prop_od.set_default_scaling("flow_mol_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.prop_od.set_default_scaling("flow_mol_phase_comp", 1, index=("Liq", "OH_-"))
    m.fs.prop_od.set_default_scaling("flow_mol_phase_comp", 1, index=("Liq", "O2-v"))
    m.fs.prop_od.set_default_scaling("flow_mol_phase_comp", 1, index=("Liq", "H2-v"))
    m.fs.prop_od.set_default_scaling("flow_mol_phase_comp", 1, index=("Liq", "H_+"))


    #Set unit model values
    #Lithium extraction pump
    iscale.set_scaling_factor(Lix.P1.control_volume.work, 1e-5) 
    
    #Lithium extraction unit
    m.fs.prop_od.set_default_scaling("flow_mol_phase_comp", 1e-4, index=("Liq", "H2O"))
    m.fs.prop_od.set_default_scaling("flow_mol_phase_comp", 1e3, index=("Liq", "Li_+"))

    #First pass desalination
    iscale.set_scaling_factor(desal.FP_P1.control_volume.work, 1e-5)
    iscale.set_scaling_factor(desal.FP_RO.area, 1e-4)
    if erd_type1 == "pressure_exchanger":
        iscale.set_scaling_factor(desal.FP_P2.control_volume.work, 1e-5)
        iscale.set_scaling_factor(desal.FP_PXR.feed_side.work, 1e-5)
        iscale.set_scaling_factor(desal.FP_PXR.brine_side.work, 1e-5)
    elif erd_type1 == "pump_as_turbine":
        iscale.set_scaling_factor(desal.FP_ERD.control_volume.work, 1e-5)
    
    #Second pass desalination
    iscale.set_scaling_factor(desal.SP_P1.control_volume.work, 1e-5)
    iscale.set_scaling_factor(desal.SP_RO.area, 1e-4)
    if erd_type2 == "pressure_exchanger":
        iscale.set_scaling_factor(desal.SP_P1.control_volume.work, 1e-5)
        iscale.set_scaling_factor(desal.SP_PXR.feed_side.work, 1e-5)
        iscale.set_scaling_factor(desal.SP_PXR.brine_side.work, 1e-5)
    elif erd_type2 == "pump_as_turbine":
        iscale.set_scaling_factor(desal.SP_ERD.control_volume.work, 1e-5)

    #H2 extraction
    #Oxygen Compression
    iscale.set_scaling_factor(H2x.Compressor_O2.feed_side.work, 1e-5)
    iscale.set_scaling_factor(H2x.Compressor_O2.brine_side.work, 1e-7)
    
    #Hydrogen Compression
    iscale.set_scaling_factor(H2x.Compressor_H2.feed_side.work, 1e-5)
    iscale.set_scaling_factor(H2x.Compressor_H2.brine_side.work, 1e-7)
    
    #Calculate and propagate scaling factors
    iscale.calculate_scaling_factors(m)

    return m

def set_operating_conditions(m):
    
    prtrt = m.fs.pretreatment = Block()
    Lix = m.fs.Liextraction = Block()
    Lixstor = m.fs.Listorage = Block()
    desal = m.fs.desalination = Block()
    psttrt = m.fs.posttreatment = Block()
    H2x = m.fs.H2extraction = Block()
    H2xstor = m.fs.H2storage = Block()

    # ---Specifications---
    #Feed

    flow_vol = 0.194 * pyunits.m**3 / pyunits.s
    conc_mass_Li = 0.00017 * pyunits.kg / pyunits.m**3
    conc_mass_Mg = 0.00128  * pyunits.kg / pyunits.m**3
    conc_mass_Na = 0.0108 * pyunits.kg / pyunits.m**3
    conc_mass_Ca = 0.4 * pyunits.kg / pyunits.m**3
    conc_mass_Cl = 0.0194 * pyunits.kg / pyunits.m**3
    conc_mass_SO4 = 0.0027 * pyunits.kg / pyunits.m**3
    conc_mass_H = 0.06 * pyunits.kg / pyunits.m**3
    conc_mass_OH = 0.0000214 *pyunits.kg / pyunits.m**3
    conc_mass_H2 = 0.12 * pyunits.kg / pyunits.m**3
    conc_mass_O2 = 0.477 * pyunits.kg / pyunits.m**3
    conc_mass_tss = 0.03 * pyunits.kg / pyunits.m**3
    temperature = 293.15 * pyunits.K
    pressure = 101325 * pyunits.Pa 

    m.fs.feed.flow_vol[0].fix(flow_vol)
    m.fs.feed.conc_mass_comp[0, "Li_+"].fix(conc_mass_Li)
    m.fs.feed.conc_mass_comp[0, "Mg_2+"].fix(conc_mass_Mg)
    m.fs.feed.conc_mass_comp[0, "Na_+"].fix(conc_mass_Na)
    m.fs.feed.conc_mass_comp[0, "Ca_2+"].fix(conc_mass_Ca)
    m.fs.feed.conc_mass_comp[0, "Cl_-"].fix(conc_mass_Cl)
    m.fs.feed.conc_mass_comp[0, "SO4_2-"].fix(conc_mass_SO4)
    m.fs.feed.conc_mass_comp[0, "H_+"].fix(conc_mass_H)
    m.fs.feed.conc_mass_comp[0, "OH_-"].fix(conc_mass_OH)
    m.fs.feed.conc_mass_comp[0, "H2-v"].fix(conc_mass_H2)
    m.fs.feed.conc_mass_comp[0, "O2-v"].fix(conc_mass_O2)
    m.fs.feed.conc_mass_comp[0, "tss"].fix(conc_mass_tss)
    solve(m.fs.feed, checkpoint="solve feed block")

    m.fs.tb_prtrt_Lix.properties_out[0].temperature.fix(temperature)
    m.fs.tb_prtrt_Lix.properties_out[0].pressure.fix(pressure)

    #Pretreatment

    #Intake
    prtrt.intake.load_parameters_from_database()
    
    #Ferric chloride
    m.db.get_unit_operation_parameters("chemical_addition")
    prtrt.ferric_chloride_addition.load_parameters_from_database()
    prtrt.ferric_chloride_addition.chemical_dosage.fix(20)

    #Chlorination
    m.db.get_unit_operation_parameters("chlorination")
    prtrt.chlorination.load_parameters_from_database(use_default_removal=True)

    #Static mixer
    m.db.get_unit_operation_parameters("static_mixer")
    prtrt.static_mixer.load_parameters_from_database(use_default_removal=True)

    #Storage
    m.db.get_unit_operation_parameters("storage_tank")
    prtrt.storage.load_parameters_from_database(use_default_removal=True)
    prtrt.storage.storage_time.fix(2)

    #Screening
    m.db.get_unit_operation_parameters("screen")
    prtrt.screening.load_parameters_from_database(use_default_removal=True)

    #Coagulation and Flocculation
    m.db.get_unit_operation_parameters("coag_and_floc")
    prtrt.coag_and_floc.load_parameters_from_database(use_default_removal=True)

    #Sedimentation
    m.db.get_unit_operation_parameters("sedimentation")
    prtrt.sedimentation.load_parameters_from_database(use_default_removal=True)

    #Air Flotation
    m.db.get_unit_operation_parameters("air_flotation")
    prtrt.flotation.load_parameters_from_database(use_default_removal=True)

    #Gravity-Basin
    m.db.get_unit_operation_parameters("fixed_bed")
    prtrt.gravity_basin.load_parameters_from_database(use_default_removal=True)

    #Mediafiltration
    m.db.get_unit_operation_parameters("media_filtration")
    prtrt.mfiltration.load_parameters_from_database(use_default_removal=True)

    #Mediafiltration backwash
    m.db.get_unit_operation_parameters("backwash_solids_handling")
    prtrt.mbackwash_pump.load_parameters_from_database(use_default_removal=True)

    #Antiscalant addition
    prtrt.anti_scalant_addition.load_parameters_from_database()
    prtrt.anti_scalant_addition.chemical_dosage.fix(5)

    #Cartridge filtration
    m.db.get_unit_operation_parameters("cartridge_filtration")
    prtrt.cfiltration.load_parameters_from_database(use_default_removal=True)

    #GAC
    m.db.get_unit_operation_parameters("gac")
    prtrt.gac.load_parameters_from_database(use_default_removal=True)

    #GAC backwash
    m.db.get_unit_operation_parameters("backwash_solids_handling")
    prtrt.gbackwash_pump.load_parameters_from_database(use_default_removal=True)

    #Mediafiltration disposal
    m.db.get_unit_operation_parameters("landfill")
    prtrt.mf_disposal.load_parameters_from_database()

    #GAC filtration disposal
    m.db.get_unit_operation_parameters("landfill")
    prtrt.gac_disposal.load_parameters_from_database()

    #Li extraction
    Lix.P1.efficiency_pump.fix(0.80)
    operating_pressure = 101325 * pyunits.Pa
    Lix.P1.control_volume.properties_out[0].pressure.fix(operating_pressure)

    Lix.IX.resin_diam.fix()
    Lix.IX.resin_bulk_dens.fix()
    Lix.IX.bed_porosity.fix()
    Lix.IX.service_flow_rate.fix(15)
    Lix.IX.bed_depth.fix()
    Lix.IX.number_columns.fix()
    Lix.IX.langmuir["Li_+"].fix(0.9932)
    Lix.IX.resin_max_capacity.fix(5.57)
    Lix.IX.dimensionless_time.fix()

    Lixstor.storage.load_parameters_from_database(use_default_removal=True)
    Lixstor.storage.storage_time(1)

    #First pass desalination
    #pump 1 is a high pressure pump, 2 DOF (efficiency and outlet pressure)
    desal.FP_P1.efficiency_pump.fix(0.80)
    operating_pressure = 70e5 * pyunits.Pa
    desal.FP_P1.control_volume.properties_out[0].pressure.fix(operating_pressure)

    #RO unit
    desal.FP_RO.A_comp.fix(4.2e-12) #membrane water permeability coefficient [m/s-Pa]
    desal.FP_RO.B_comp.fix(3.5e-8) #membrane salt permeability coefficient [m/s]
    desal.FP_RO.feed_side.channel_height.fix(1e-3) #channel height in membrane stage [m]
    desal.FP_RO.feed_side.spacer_porosity.fix(0.97) #spacer porosity in membrane stage [-]
    desal.FP_RO.permeate.pressure[0].fix(101325) 
    desal.FP_RO.width.fix(1000) #stage width [m]
    desal.FP_RO.area.fix(flow_vol * 4.5e4 * pyunits.s / pyunits.m) #stage area [m2] 

    if m.erd_type1 == "pressure_exchanger":
        #Splitter (no DOF)
        
        #Pressure exchanger (1 DOF)
        desal.FP_PXR.efficiency_pressure_exchanger.fix(0.95)

        #Pump 2 is a booster pump with (1 DOF, efficiency. Pressure must match the high pressure pump)
        desal.FP_P2.efficiency_pump.fix(0.80)

        #Mixer (no DOF)
    elif m.erd.type1 == "pump_as_turbine":
        #ERD, (2 DOF; efficiency and outlet pressure)
        desal.FP_ERD.efficiency_pump.fix(0.95)
        desal.FP_ERD.control_volume.properties_out[0].pressure.fix(101325)

    #Second pass desalination
    desal.SP_P1.efficiency_pump.fix(0.80)
    desal.SP_P1.control_volume.properties_out[0].pressure.fix(operating_pressure)

    #RO unit
    desal.SP_RO.A_comp.fix(4.2e-12)
    desal.SP_RO.B_comp.fix(3.5e-8)
    desal.SP_RO.feed_side.channel_height.fix(1e-3)
    desal.SP_RO.feed_side.spacer_porosity.fix(0.97)
    desal.SP_RO.permeate.pressure[0].fix(101325)
    desal.SP_RO.width.fix(1000)
    desal.SP_RO.area.fix(flow_vol * 4.5e4 * pyunits.s / pyunits.m)

    if m.erd_type2 == "pressure_exchanger":
        #Splitter (no DOF)

        #Pressure exchanger (1 DOF)
        desal.SP_PXR.efficiency_pressure_exchanger.fix(0.95)

        #Pump 2
        desal.SP_P2.efficiency_pump.fix(0.80)

        #Mixer (no DOF)
    elif m.erd.type2 == "pump_as_turbine":
        desal.SP_ERD.efficiency_pump.fix(0.95)
        desal.SP_ERD.control_volume.properties_out[0].pressure.fix(operating_pressure)

    #Desalination posttreatment

    #Demineralization
    m.db.get_unit_operation_parameters("ion_exchange")
    psttrt.IX.load_parameters_from_database(use_default_removal=True)
    #update resin replacement rate based on AmberLite™ MB20 H/OH Ion Exchange Resin

    #Product water storage
    m.db.get_unit_operation_parameters("storage_tank")
    psttrt.storage.load_parameters_from_database(use_default_removal=True)
    psttrt.storage.storage_time.fix(2)

    #H2 Extraction 
    if m.elec_type == "PEM":
        m.fs.tb_psttrt_H2x.properties_out[0].temperature.fix(273.15 + 70)
        m.fs.tb_psttrt_H2x.properties_out[0].pressure.fix(30e5)

        #Anolyte block
        H2x.ER.anolyte.properties_in[0].temperature.fix(273.15 + 70)
        H2x.ER.anolyte.properties_in[0].pressure.fix(30e5)
        H2x.ER.anolyte.properties_in[0].flow_mol_phase_comp["Liq", "H2O"].fix(5.551)
        H2x.ER.anolyte.properties_in[0].flow_mol_phase_comp["Liq", "H_+"].fix(0)
        H2x.ER.anolyte.properties_in[0].flow_mol_phase_comp["Liq", "O2-v"].fix(0)
        H2x.ER.anolyte.properties_in[0].flow_mol_phase_comp["Liq", "H2-v"].fix(0)

        #Catholyte block
        H2x.ER.catholyte.properties_in[0].temperature.fix(273.15 + 70)
        H2x.ER.catholyte.properties_in[0].pressure.fix(30e5)
        H2x.ER.catholyte.properties_in[0].flow_mol_phase_comp["Liq", "H2O"].fix(5.551)
        H2x.ER.catholyte.properties_in[0].flow_mol_phase_comp["Liq", "H_+"].fix(0)
        H2x.ER.catholyte.properties_in[0].flow_mol_phase_comp["Liq", "O2-v"].fix(0)
        H2x.ER.catholyte.properties_in[0].flow_mol_phase_comp["Liq", "H2-v"].fix(0)

        #PEM reactions
        H2x.ER.membrane_ion_transport_number["Liq", "H_+"].fix(1)
        # H2O --> 0.5 O2 + 2H+ +2e (oxidation reaction)
        H2x.ER.anode_electrochem_potential.fix(1.23)
        H2x.ER.anode_stoich["Liq", "H2O"].fix(-1)
        H2x.ER.anode_stoich["Liq", "O2-v"].fix(0.5) #gaseous 
        H2x.ER.anode_stoich["Liq", "H_+"].fix(2)

        # cathode properties 
        # 2H+ + 2e- --> H2 (reduction reaction)
        H2x.ER.cathode_electrochem_potential.fix(0.000)
        H2x.ER.cathode_stoich["Liq", "H_+"].fix(-2)
        H2x.ER.cathode_stoich["Liq", "H2-v"].fix(1) #gaseous

        #PEM Electrolysis unit properties: 
        H2x.ER.membrane_current_density.fix(3e4) #membrane current density (A/m^2)
        H2x.ER.anode_current_density.fix(3e4) #anode current density (A/m^2) 
        H2x.ER.anode_overpotential.fix(3e-2) #anode overpotential (V)  source:https://www.sciencedirect.com/science/article/pii/S0378775311018131
        H2x.ER.cathode_current_density.fix(3e4) #cathode current density (A/m^2) 
        H2x.ER.cathode_overpotential.fix(3e-2) #cathode overpotential (V) source:https://www.sciencedirect.com/science/article/pii/S0378775311018131
        H2x.ER.current.fix(1.875e4) #current (A) source:https://www.sciencedirect.com/science/article/pii/S0378775311018131
        H2x.ER.efficiency_current.fix(0.65) #current efficiency (-) source:https://onlinelibrary.wiley.com/doi/epdf/10.1002/sstr.202200130
        H2x.ER.resistance.fix(1.1e-4) #ohmic resistance (mΩ cm2,unit discrepancy) source: https://iopscience.iop.org/article/10.1149/08613.0695ecst/pdf
        
    elif m.elec_type == "AWE":
        m.fs.tb_psttrt_H2x.properties_out[0].temperature.fix(273.15 + 70)
        m.fs.tb_psttrt_H2x.properties_out[0].pressure.fix(30e5)
        
        #Anolyte Block
        H2x.ER.anolyte.properties_in[0].temperature.fix(273.15 + 70)
        H2x.ER.anolyte.properties_in[0].pressure.fix(30e5)
        H2x.ER.anolyte.properties_in[0].flow_mass_phase_comp["Liq", "H2O"].fix(5.551)
        H2x.ER.anolyte.properties_in[0].flow_mass_phase_comp["Liq", "OH_-"].fix(0)
        H2x.ER.anolyte.properties_in[0].flow_mass_phase_comp["Liq", "O2-v"].fix(2.22)
        H2x.ER.anolyte.properties_in[0].flow_mass_phase_comp["Liq", "H2-v"].fix(0) 

        #Catholyte Block
        H2x.ER.catholyte.properties_in[0].temperature.fix(273.15 + 70)
        H2x.ER.catholyte.properties_in[0].pressure.fix(30e5)
        H2x.ER.catholyte.properties_in[0].flow_mass_phase_comp["Liq", "H2O"].fix(5.551)
        H2x.ER.catholyte.properties_in[0].flow_mass_phase_comp["Liq", "OH_-"].fix(1.5)
        H2x.ER.catholyte.properties_in[0].flow_mass_phase_comp["Liq", "O2-v"].fix(0)
        H2x.ER.catholyte.properties_in[0].flow_mass_phase_comp["Liq", "H2-v"].fix(4.44) 
    
        #Alkaline Water Electrolysis reactions 
        #Assumes 30% weight KOH electrolyte introduced to the anode and cathode
        H2x.ER.dens_mass_const.fix(1063)
        H2x.ER.membrane_ion_transport_number["Liq", "OH_-"].fix(1)

        #anode properties
        # 4OH- --> 2H2O + O2 + 4e- 
        H2x.ER.anode_electrochem_potential.fix(-0.4)
        H2x.ER.anode_stoich["Liq", "OH_-"].fix(-4)
        H2x.ER.anode_stoich["Liq", "H2O"].fix(2)
        H2x.ER.anode_stoich["Liq", "O2-v"].fix(1)

        #cathode properties
        # 4H2O + 4e- --> 2H2 + 4OH-
        H2x.ER.cathode_electrochem_potential.fix(-0.83)
        H2x.ER.cathode_stoich["Liq", "H2O"].fix(-4)
        H2x.ER.cathode_stoich["Liq", "H2-v"].fix(2)
        H2x.ER.cathode_stoich["Liq", "OH_-"].fix(4)

        #Electrolysis unit properties source: https://onlinelibrary.wiley.com/doi/epdf/10.1002/sstr.202200130
        H2x.ER.membrane_current_density.fix(3e3) #membrane current density (A/m^2)
        H2x.ER.anode_current_density.fix(3e3) #anode current density (A/m^2) 
        H2x.ER.anode_overpotential.fix(9.39e-1) #anode overpotential (V)  source:https://www.sciencedirect.com/science/article/pii/S0378775311018131
        H2x.ER.cathode_current_density.fix(3e3) #cathode current density (A/m^2) 
        H2x.ER.cathode_overpotential.fix(9.6e-1) #cathode overpotential (V) source:https://www.sciencedirect.com/science/article/pii/S0378775311018131
        H2x.ER.current.fix(2.25e4) #current (A) source:https://www.sciencedirect.com/science/article/pii/S0378775311018131
        H2x.ER.efficiency_current.fix(0.61) #current efficiency (-) source:https://onlinelibrary.wiley.com/doi/epdf/10.1002/sstr.202200130
        H2x.ER.resistance.fix(8.18e-5) #ohmic resistance (mΩ cm2,unit discrepancy) source: https://iopscience.iop.org/article/10.1149/08613.0695ecst/pdf
    
        #Produced gas compression
        H2x.Compressor_O2.efficiency_pressure_exchanger.fix(0.95)
        H2x.Compressor_H2.efficiency_pressure_exchanger.fix(0.95)

        #Produced gas storage
        H2xstor.O2.load_parameters_from_database(use_default_removal=True)
        H2xstor.O2.storage_time.fix(24)

        H2xstor.H2.load_parameters_from_database(use_default_removal=True)
        H2xstor.H2.storage_time(24)

def initialize_system(m):
        prtrt = m.fs.pretreatment 
        Lix = m.fs.Liextraction 
        Lixstor =m.fs.Listorage 
        desal = m.fs.desalination 
        psttrt = m.fs.posttreatment
        H2x = m.fs.H2extraction 
        H2xstor = m.fs.H2storage 

        #initialize feed
        solve(m.fs.feed, checkpoint="solve flowsheet after initialize feed")

        #initialize pretreatment
        propagate_state(m.fs.s_feed)
        flags = fix_state_vars(prtrt.intake.properties)
        solve(prtrt, checkpoint="solve flowsheet after initializing pre-treatment")
        revert_state_vars(prtrt.intake.properties, flags)

        #initialize Li extraction
        propagate_state(m.fs.s_prtrt_tb)
        m.fs.tb_prtrt_Lix.properties_out[0].flow_mass_phase_comp["Liq", "H2O"] = value(
            m.fs.tb_prtrt_Lix.properties_in[0].flow_mass_comp["H2O"]
        )
        m.fs.tb_prtrt_Lix.properties_out[0].flow_mass_phase_comp["Liq", "Li_+"] = value(
            m.fs.tb_prtrt_Lix.properties_in[0].flow_mass_comp["Li_+"]
        )
        m.fs.tb_prtrt_Lix.properties_out[0].flow_mass_phase_comp["Liq", "Ca_2+"] = value(
            m.fs.tb_prtrt_Lix.properties_in[0].flow_mass_comp["Ca_2+"]
        )
        m.fs.tb_prtrt_Lix.properties_out[0].flow_mass_phase_comp["Liq", "Mg_2+"] = value(
            m.fs.tb_prtrt_Lix.properties_in[0].flow_mass_comp["Mg_2+"]
        )
        m.fs.tb_prtrt_Lix.properties_out[0].flow_mass_phase_comp["Liq", "Cl_-"] = value(
            m.fs.tb_prtrt_Lix.properties_in[0].flow_mass_comp["Cl_-"]
        )
        m.fs.tb_prtrt_Lix.properties_out[0].flow_mass_phase_comp["Liq", "SO4_2-"] = value(
            m.fs.tb_prtrt_Lix.properties_in[0].flow_mass_comp["SO4_2-"]
        )
        m.fs.tb_prtrt_Lix.properties_out[0].flow_mass_phase_comp["Liq", "Na_+"] = value(
            m.fs.tb_prtrt_Lix.properties_in[0].flow_mass_comp["Na_+"]
        )
        m.fs.tb_prtrt_Lix.properties_out[0].flow_mass_phase_comp["Liq", "H_+"] = value(
            m.fs.tb_prtrt_Lix.properties_in[0].flow_mass_comp["H_+"]
        )
        m.fs.tb_prtrt_Lix.properties_out[0].flow_mass_phase_comp["Liq", "OH_-"] = value(
            m.fs.tb_prtrt_Lix.properties_in[0].flow_mass_comp["OH_-"]
        )
        m.fs.tb_prtrt_Lix.properties_out[0].flow_mass_phase_comp["Liq", "O2-v"] = value(
            m.fs.tb_prtrt_Lix.properties_in[0].flow_mass_comp["O2-v"]
        )
        m.fs.tb_prtrt_Lix.properties_out[0].flow_mass_phase_comp["Liq", "H2-v"] = value(
            m.fs.tb_prtrt_Lix.properties_in[0].flow_mass_comp["H2-v"]
        )
        m.fs.tb_prtrt_Lix.properties_out[0].flow_mass_phase_comp["Liq", "tss"] = value(
            m.fs.tb_prtrt_Lix.properties_in[0].flow_mass_comp["tss"]
        )
        Lix.P1.initialize()
        Lix.IX.initialize()
        propagate_state(m.fs.s_tb_Lix)
        
        #initialize Lithium storage 
        propagate_state(m.fs.s_Lix_tb)
        m.fs.tb_Lix_Lixstor.properties_out[0].flow_mass_comp["H2O"] = value(
            m.fs.tb_Lix_Lixstor.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
        )
        m.fs.tb_Lix_Lixstor.properties_out[0].flow_mass_comp["Li_+"] = value(
            m.fs.tb_Lix_Lixstor.properties_in[0].flow_mass_phase_comp["Liq", "Li_+"]
        )
        m.fs.tb_Lix_Lixstor.properties_out[0].flow_mass_comp["Ca_2+"] = value(
            m.fs.tb_Lix_Lixstor.properties_in[0].flow_mass_phase_comp["Liq", "Ca_2+"]
        )
        m.fs.tb_Lix_Lixstor.properties_out[0].flow_mass_comp["Mg_2+"] = value(
            m.fs.tb_Lix_Lixstor.properties_in[0].flow_mass_phase_comp["Liq", "Mg_2+"]
        )
        m.fs.tb_Lix_Lixstor.properties_out[0].flow_mass_comp["Cl_-"] = value(
            m.fs.tb_Lix_Lixstor.properties_in[0].flow_mass_phase_comp["Liq", "Cl_-"]
        )
        m.fs.tb_Lix_Lixstor.properties_out[0].flow_mass_comp["SO4_2-"] = value(
            m.fs.tb_Lix_Lixstor.properties_in[0].flow_mass_phase_comp["Liq", "SO4_2-"]
        )
        m.fs.tb_Lix_Lixstor.properties_out[0].flow_mass_comp["Na_+"] = value(
            m.fs.tb_Lix_Lixstor.properties_in[0].flow_mass_phase_comp["Liq", "Na_+"]
        )
        m.fs.tb_Lix_Lixstor.properties_out[0].flow_mass_comp["H_+"] = value(
            m.fs.tb_Lix_Lixstor.properties_in[0].flow_mass_phase_comp["Liq", "H_+"]
        )
        m.fs.tb_Lix_Lixstor.properties_out[0].flow_mass_comp["OH_-"] = value(
            m.fs.tb_Lix_Lixstor.properties_in[0].flow_mass_phase_comp["Liq", "OH_-"]
        )
        m.fs.tb_Lix_Lixstor.properties_out[0].flow_mass_comp["O2-v"] = value(
            m.fs.tb_Lix_Lixstor.properties_in[0].flow_mass_phase_comp["Liq", "O2-v"]
        )
        m.fs.tb_Lix_Lixstor.properties_out[0].flow_mass_comp["H2-v"] = value(
            m.fs.tb_Lix_Lixstor.properties_in[0].flow_mass_phase_comp["Liq", "H2-v"]
        )
        propagate_state(m.fs.s_tb_Lixstor)
        flags = fix_state_vars(Lixstor.storage.properties)
        solve(Lixstor, checkpoint="solve flowsheet after initializing Lithium Storage")
        revert_state_vars(Lixstor.storage.properties, flags)

        propagate_state(m.fs.s_Lixstor_tb)
        #Translator Block
        m.fs.tb_Lixstor_desal.properties_out[0].flow_mass_phase_comp["Liq", "H2O"] = value(
            m.fs.tb_Lixstor_desal.properties_in[0].flow_mass_comp["H2O"]
        )

        m.fs.tb_Lixstor_desal.properties_out[0].flow_mass_phase_comp["Liq", "Li_+"] = value(
            m.fs.tb_Lixstor_desal.properties_in[0].flow_mass_comp["Li_+"]
        )

        m.fs.tb_Lixstor_desal.properties_out[0].flow_mass_phase_comp["Liq", "Ca_2+"] = value(
            m.fs.tb_Lixstor_desal.properties_in[0].flow_mass_comp["Ca_2+"]
        )

        m.fs.tb_Lixstor_desal.properties_out[0].flow_mass_phase_comp["Liq", "Mg_2+"] = value(
            m.fs.tb_Lixstor_desal.properties_in[0].flow_mass_comp["Mg_2+"]
        )

        m.fs.tb_Lixstor_desal.properties_out[0].flow_mass_phase_comp["Liq", "Cl_-"] = value(
            m.fs.tb_Lixstor_desal.properties_in[0].flow_mass_comp["Cl_-"]
        )

        m.fs.tb_Lixstor_desal.properties_out[0].flow_mass_phase_comp["Liq", "SO4_2-"] = value(
            m.fs.tb_Lixstor_desal.properties_in[0].flow_mass_comp["SO4_2-"]
        )

        m.fs.tb_Lixstor_desal.properties_out[0].flow_mass_phase_comp["Liq", "Na_+"] = value(
            m.fs.tb_Lixstor_desal.properties_in[0].flow_mass_comp["Na_+"]
        )

        m.fs.tb_Lixstor_desal.properties_out[0].flow_mass_phase_comp["Liq", "H_+"] = value(
            m.fs.tb_Lixstor_desal.properties_in[0].flow_mass_comp["H_+"]
        )

        m.fs.tb_Lixstor_desal.properties_out[0].flow_mass_phase_comp["Liq", "OH_-"] = value(
            m.fs.tb_Lixstor_desal.properties_in[0].flow_mass_comp["OH_-"]
        )

        m.fs.tb_Lixstor_desal.properties_out[0].flow_mass_phase_comp["Liq", "O2-v"] = value(
            m.fs.tb_Lixstor_desal.properties_in[0].flow_mass_comp["O2-v"]
        )

        m.fs.tb_Lixstor_desal.properties_out[0].flow_mass_phase_comp["Liq", "H2-v"] = value(
            m.fs.tb_Lixstor_desal.properties_in[0].flow_mass_comp["H2-v"]
        )
        #First Pass RO Unit
        desal.FP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "H2O"] = value(
            m.fs.feed_properties[0].flow_mass_comp["H2O"]
        )
        desal.FP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "Li_+"] = value(
            m.fs.feed_properties[0].flow_mass_comp["Li_+"]
        )
        desal.FP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "Ca_2+"] = value(
            m.fs.feed_properties[0].flow_mass_comp["Ca_2+"]
        )
        desal.FP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "Mg_2+"] = value(
            m.fs.feed_properties[0].flow_mass_comp["Mg_2+"]
        )
        desal.FP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "Cl_-"] = value(
            m.fs.feed_properties[0].flow_mass_comp["Cl_-"]
        )
        desal.FP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "SO4_2-"] = value(
            m.fs.feed_properties[0].flow_mass_comp["SO4_2-"]
        )
        desal.FP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "Na_+"] = value(
            m.fs.feed_properties[0].flow_mass_comp["Na_+"]
        )
        desal.FP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "H_+"] = value(
            m.fs.feed_properties[0].flow_mass_comp["H_+"]
        )
        desal.FP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "OH_-"] = value(
            m.fs.feed_properties[0].flow_mass_comp["OH_-"]
        )
        desal.FP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "O2-v"] = value(
            m.fs.feed_properties[0].flow_mass_comp["O2-v"]
        )
        desal.FP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "H2-v"] = value(
            m.fs.feed_properties[0].flow_mass_comp["H2-v"]
        )
        desal.FP_RO.feed_side.properties_in[0].temperature = value(
            m.fs.tb_prtrt_desal.properties_out[0].temperature
        )
        desal.FP_RO.feed_side.properties_in[0].pressure = value(
            desal.FP_P1.control_volume.properties_out[0].pressure
        )
        desal.FP_RO.initialize()
        
        propagate_state(m.fs.s_tb_desal)
        if m.erd_type1 == "pressure_exchanger":
            flags = fix_state_vars(desal.FP_S1.mixed_state)
            solve(
                desal,
                checkpoint=f"solve flowsheet after initializing {m.erd_type1, m.erd_type2, m.elec_type} system"
            )
            revert_state_vars(desal.FP_S1.mixed_state, flags)
        elif m.erd_type == "pump_as_turbine":
            flags = fix_state_vars(desal.FP_P1.control_volume.properties_in)
            solve(
                desal, 
                checkpoint=f"solve flowsheet after initializing {m.erd_type1, m.erd_type2, m.elec_type} system"
            )
            revert_state_vars(desal.FP_P1.control_volume.properties_in, flags)

        #Second Pass RO Unit
        desal.SP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "H2O"] = value(
            m.fs.feed_properties[0].flow_mass_comp["H2O"]
        )
        desal.SP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "Li_+"] = value(
            m.fs.feed_properties[0].flow_mass_comp["Li_+"]
        )
        desal.SP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "Ca_2+"] = value(
            m.fs.feed_properties[0].flow_mass_comp["Ca_2+"]
        )
        desal.SP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "Mg_2+"] = value(
            m.fs.feed_properties[0].flow_mass_comp["Mg_2+"]
        )
        desal.SP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "Cl_-"] = value(
            m.fs.feed_properties[0].flow_mass_comp["Cl_-"]
        )
        desal.SP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "SO4_2-"] = value(
            m.fs.feed_properties[0].flow_mass_comp["SO4_2-"]
        )
        desal.SP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "Na_+"] = value(
            m.fs.feed_properties[0].flow_mass_comp["Na_+"]
        )
        desal.SP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "H_+"] = value(
            m.fs.feed_properties[0].flow_mass_comp["H_+"]
        )
        desal.SP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "OH_-"] = value(
            m.fs.feed_properties[0].flow_mass_comp["OH_-"]
        )
        desal.SP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "O2-v"] = value(
            m.fs.feed_properties[0].flow_mass_comp["O2-v"]
        )
        desal.SP_RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "H2-v"] = value(
            m.fs.feed_properties[0].flow_mass_comp["H2-v"]
        )
        desal.SP_RO.feed_side.properties_in[0].temperature = value(
            m.fs.tb_prtrt_desal.properties_out[0].temperature
        )
        desal.SP_RO.feed_side.properties_in[0].pressure = value(
            desal.SP_P1.control_volume.properties_out[0].pressure
        )
        desal.SP_RO.initialize()

        if m.erd_type2 == "pressure_exchanger":
            flags = fix_state_vars(desal.FP_S1.mixed_state)
            solve(
                desal,
                checkpoint=f"solve flowsheet after initializing {m.erd_type1, m.erd_type2, m.elec_type} system"
            )
            revert_state_vars(desal.FP_S1.mixed_state, flags)
        elif m.erd_type2 == "pump_as_turbine":
            flags = fix_state_vars(desal.FP_P1.control_volume.properties_in)
            solve(
                desal, 
                checkpoint=f"solve flowsheet after initializing {m.erd_type1, m.erd_type2, m.elec_type} system"
            )
            revert_state_vars(desal.FP_P1.control_volume.properties_in, flags)

        #initialize posttreatment
        propagate_state(m.fs.s_desal_tb)
        m.fs.tb_desal_psttrt.properties_out[0].flow_mass_comp["H2O"] = value(
            m.fs.tb_desal_psttrt.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
        )
        m.fs.tb_desal_psttrt.properties_out[0].flow_mass_comp["tds"] = value(
            m.fs.tb_desal_psttrt.properties_in[0].flow_mass_phase_comp["Liq", "Li_+"]
            + value(m.fs.tb_desal_psttrt.properties_in[0].flow_mass_phase_comp["Liq", "Ca_2+"])
            + value(m.fs.tb_desal_psttrt.properties_in[0].flow_mass_phase_comp["Liq", "Mg_2+"])
            + value(m.fs.tb_desal_psttrt.properties_in[0].flow_mass_phase_comp["Liq", "Cl_-"])
            + value(m.fs.tb_desal_psttrt.properties_in[0].flow_mass_phase_comp["Liq", "SO4_2-"])
            + value(m.fs.tb_desal_psttrt.properties_in[0].flow_mass_phase_comp["Liq", "Na_+"])
        )
        m.fs.tb_desal_psttrt.properties_out[0].flow_mass_comp["H_+"] = value(
            m.fs.tb_desal_psttrt.properties_in[0].flow_mass_phase_comp["Liq", "H_+"]
        )
        m.fs.tb_desal_psttrt.properties_out[0].flow_mass_comp["OH_-"] = value(
            m.fs.tb_desal_psttrt.properties_in[0].flow_mass_phase_comp["Liq", "OH_-"]
        )
        m.fs.tb_desal_psttrt.properties_out[0].flow_mass_comp["O2-v"] = value(
            m.fs.tb_desal_psttrt.properties_in[0].flow_mass_phase_comp["Liq", "O2-v"]
        )
        m.fs.tb_desal_psttrt.properties_out[0].flow_mass_comp["H2-v"] = value(
            m.fs.tb_desal_psttrt.properties_in[0].flow_mass_phase_comp["Liq", "H2-v"]
        )
        propagate_state(m.fs.s_tb_psttrt)
        flags = fix_state_vars(psttrt.IX.properties)
        solve(psttrt, checkpoint="solve flowsheet after initializing post-treatment")
        revert_state_vars(psttrt.IX.properties, flags)


        #initialize H2 extraction
        propagate_state(m.fs.s_psttrt_tb)
        m.fs.tb_psttrt_H2x.properties_out[0].flow_mass_phase_comp["Liq", "H2O"] = value(
            m.fs.tb_psttrt_H2x.properties_in[0].flow_mass_comp["H2O"]
        )
        m.fs.tb_psttrt_H2x.properties_out[0].flow_mass_phase_comp["Liq", "tds"] = value(
            m.fs.tb_psttrt_H2x.properties_in[0].flow_mass_comp["tds"]
        )
        m.fs.tb_psttrt_H2x.properties_out[0].flow_mass_phase_comp["Liq", "H_+"] = value(
            m.fs.tb_psttrt_H2x.properties_in[0].flow_mass_comp["H_+"]
        )
        m.fs.tb_psttrt_H2x.properties_out[0].flow_mass_phase_comp["Liq", "OH_-"] = value(
            m.fs.
            tb_psttrt_H2x.properties_in[0].flow_mass_comp["OH_-"]
        )
        m.fs.tb_psttrt_H2x.properties_out[0].flow_mass_phase_comp["Liq", "O2-v"] = value(
            m.fs.tb_psttrt_H2x.properties_in[0].flow_mass_comp["O2-v"]
        )
        m.fs.tb_psttrt_H2x.properties_out[0].flow_mass_phase_comp["Liq", "H2-v"] = value(
            m.fs.tb_psttrt_H2x.properties_in[0].flow_mass_comp["H2-v"]
        )
        H2x.ER.initialize()
        if m.elec_type == "PEM":
            solve(
                H2x,
                checkpoint=f"solve flowsheet after initializing {m.elec_type} electrolysis"
            )
        elif m.elec_type == "AWE":
            solve(
                H2x,
                checkpoint=f"solve flowsheet after initializing {m.elec_type} electrolysis"
            )
        propagate_state(m.fs.s_tb_H2)
        propagate_state(m.fs.s_tb_O2)

        H2x.Compressor_H2.initialize()
        H2x.Compressor_O2.initialize()

        propagate_state(m.fs.s_H2_tb)
        propagate_state(m.fs.s_O2_tb)
        m.fs.tb_H2x_H2xstor.properties_out[0].flow_mass_comp["H2-v"] = value(
            m.fs.tb_H2x_H2xstor.properties_in[0].flow_mass_phase_comp["Liq", "H2-v"]
        )
        m.fs.tb_H2x_H2xstor.properties_out[0].flow_mass_comp["O2-v"] = value(
            m.fs.tb_H2x_H2xstor.properties_in[0].flow_mass_phase_comp["Liq", "O2-v"]
        )
        propagate_state(m.fs.s_tb_H2xstor_H2)
        propagate_state(m.fs.s_tb_H2xstor_O2)

        flags = fix_state_vars(H2xstor.H2.properties)
        solve(H2xstor, checkpoint="solve flowsheet after initializing gas storage")
        revert_state_vars(H2xstor.H2.properties, flags)
        

def optimize_operation(m): 
        Lix = m.fs.Liextraction  
        desal = m.fs.desalination 
       
        #Li extraction unit 
        Lix.P1.control_volume.properties_out[0].pressure.unfix()
        Lix.P1.control_volume.properties_out[0].pressure.setub(8300000) #pressure vessel burst pressure
        Lix.P1.control_volume.properties_out[0].pressure.setlb(100000)

        ix = Lix.IX
        target_ion = Lix.IX.config.target_ion
        ix.process_flow.properties_out[0].conc_mass_phase_comp["Liq", target_ion].fix(0.017) #mg/l
        propagate_state(m.fs.s_Lix_tb)
        propagate_state(Lix.s02)
        Lix.IX.initialize()
        Lix.regen.initialize()
        ix.dimensionless_time.unfix()
        ix.number_columns_unfix()
        ix.bed_depth.unfix()

        #First pass RO unit
        desal.FP_P2.control_volume.properties_out[0].pressure.unfix()
        desal.FP_P2.control_volume.properties_out[0].pressure.setub(8300000)
        desal.FP_P2.control_volume.properties_out[0].pressure.setlb(100000)

        #First pass RO inlet velocity
        desal.FP_RO.feed_side.velocity[0, 0].unfix()
        desal.FP_RO.feed_side.velocity[0, 0].setub(0.3)
        desal.FP_RO.feed_side.velocity[0, 0].setlb(0.1)

        #First pass RO membrane area
        desal.FP_RO.area.unfix()
        desal.FP_RO.area.setub(5000)
        desal.FP_RO.area.setlb(50)

        #First pass RO recovery
        desal.FP_RO.recovery_vol_phase[0, "Liq"].unfix()
        desal.FP_RO.recovery_vol_phase[0, "Liq"].setub(0.99)
        desal.FP_RO.recovery_vol_phase[0, "Liq"].setlb(0.1)

        #Second pass RO unit
        desal.SP_P2.control_volume.properties_out[0].pressure.unfix()
        desal.SP_P2.control_volume.properties_out[0].pressure.setub(8300000)
        desal.SP_P2.control_volume.properties_out[0].pressure.setlb(100000)

        #Second pass Ro inlet velocity
        desal.SP_RO.feed_side.velocity[0, 0].unfix()
        desal.SP_RO.feed_side.velocity[0, 0].setub(0.3)
        desal.SP_RO.feed_side.velocity[0, 0].setub(0.1)

        #Second pass RO membrane area
        desal.SP_RO.area.unfix()
        desal.SP_RO.area.setub(5000)
        desal.SP_RO.area.setlb(50)

        #Second pass RO recovery
        desal.SP_RO.recovery_vol_phase[0, "Liq"].unfix()
        desal.SP_RO.recovery_vol_phase[0, "Liq"].setub(0.99)
        desal.SP_RO.recovery_vol_phase[0, "Liq"].setlb(0.1)

        m.fs.objective = Objective(expr=m.fs.LCOT)
        return


def solve(blk, solver=None, checkpoint=None, tee=False, fail_flag=True):
        if solver is None:
            solver = get_solver()
        results = solver.solve(blk, tee=tee)
        check_solve(results, checkpoint=checkpoint, logger=_log, fail_flag=fail_flag)
        return results 
    
def display_results(m):
        m.fs.feed.report()
        m.fs.pretreatment.intake.report()
        m.fs.pretreatment.ferric_chloride_addition.report()
        m.fs.pretreatment.chlorination.report()
        m.fs.pretreatment.static_mixer.report()
        m.fs.pretreatment.storage.report()
        m.fs.pretreatment.screening.report()
        m.fs.pretreatment.coag_and_floc.report()
        m.fs.pretreatment.sedimentation.report()
        m.fs.pretreatment.flotation.report()
        m.fs.pretreatment.gravity_basin.report()
        m.fs.pretreatment.mfiltration.report()
        m.fs.pretreatment.mbackwash_pump.report()
        m.fs.pretreatment.anti_scalant_addition.report()
        m.fs.pretreatment.cfiltration.report()
        m.fs.pretreatment.gac.report()
        m.fs.pretreatment.gbackwash_pump.report()
        m.fs.pretreatment.mf_disposal.report()
        m.fs.pretreatment.gac_disposal.report
        m.fs.Liextraction.P1.report()
        m.fs.Liextraction.IX.report()
        m.fs.Liextraction.regn.report()
        m.fs.Listorage.storage.report()
        if m.erd_type1 == "pressure_exchanger":
            m.fs.desalination.FP_S1.report()
            m.fs.desalination.FP_P1.report()
            m.fs.desalination.FP_P2.report()
            m.fs.desalination.FP_M1.report()
            m.fs.desalination.FP_RO.report()
            m.fs.desalination.FP_PXR.report()
        elif m.erd_type1 == "pump_as_turbine":
            m.fs.desalination.FP_P1.report()
            m.fs.desalination.FP_RO.report()
            m.fs.desalination.FP_ERD.report()
        m.fs.desalination.FP_disposal.report()
        if m.erd_type2 == "pressure_exchanger":
            m.fs.desalination.SP_S1.report()
            m.fs.desalination.SP_P1.report()
            m.fs.desalination.SP_P2.report()
            m.fs.desalination.SP_M1.report()
            m.fs.desalination.SP_RO.report()
            m.fs.desalination.SP_PXR.report()
        elif m.erd_type2 == "pump_as_turbine":
            m.fs.desalination.SP_P1.report()
            m.fs.desalination.SP_RO.report()
            m.fs.desalination.SP_ERD.report()
        m.fs.desalination.SP_disposal.report()
        m.fs.posttreatment.IX.report()
        m.fs.posttreatment.storage.report()
        if m.elec_type == "PEM":
            m.fs.H2extraction.ER.report()
            m.fs.H2extraction.Compressor_O2.report()
            m.fs.H2extraction.Compressor_H2.report()
            m.fs.H2extraction.prod_O2.report()
            m.fs.H2extraction.prod_H2.report()
        elif m.elec_type == "AWE":
            m.fs.H2extraction.ER.report()
            m.fs.H2extraction.Compressor_O2.report()
            m.fs.H2extraction.Compressor_H2.report()
            m.fs.H2extraction.prod_O2.report()
            m.fs.H2extraction.prod_H2.report()
        m.fs.H2storage.O2.report()
        m.fs.H2storage.H2.report()

def add_costing(m):
    prtrt = m.fs.pretreatment 
    Lix = m.fs.Liextraction 
    Lixstor =m.fs.Listorage 
    desal = m.fs.desalination 
    psttrt = m.fs.posttreatment
    H2x = m.fs.H2extraction 
    H2xstor = m.fs.H2storage 

    
    #Add costing package for zero-order units
    m.fs.zo_costing = ZeroOrderCosting()
    m.fs.od_costing = WaterTAPCosting()

    #Add costing to zero order units 
    prtrt.intake.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.ferric_chloride_addition = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.chlorination = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.static_mixer = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.storage = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.screening = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.coag_and_floc = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.sedimentation = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.flotation = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.gravity_basin = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.mfiltration = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.mbackwash_pump = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.anti_scalant_addition = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.cfiltration = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.gac = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    prtrt.gbackwash_pump = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)

    #Li extraction unit 
    Lix.P1.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.od_costing,
        costing_method_arguments={"cost_electricity_flow": False},
    )
    Lix.IX.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)
    Lixstor.storage = UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)

    #First Pass RO Train 
    #RO equipment is costed using  more detailed costing package
    desal.FP_P1.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.od_costing,
        costing_method_arguments={"cost_electricity_flow": True},
    )
    desal.FP_RO.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)
    if m.erd_type1 == "pressure_exchanger":
        desal.FP_S1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)
        desal.FP_M1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)
        desal.FP_PXR.costing = UnitModelCostingBlock(flowsheet_costing_blocking=m.fs.od_costing)
        desal.FP_P2.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.od_costing,
            costing_method_arguments={"cost_electricity_flow": True},
        )
    elif m.erd_type1 == "pump_as_turbine":
        desal.ERD.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)
    else: 
        raise ConfigurationError(
            f"erd_type1 was {m.erd_type1} costing only implemented"
            "for pressure_exchanger or pump_as_turbine"
        )
    desal.SP_P1.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.od_costing,
        costing_method_arguments={"cost_electricity_flow": True},
    )
    desal.SP_RO.costing= UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)
    if m.erd_type2 == "pressure_exchanger":
        desal.SP_S1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)
        desal.SP_M1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)
        desal.SP_PXR.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)
        desal.SP_P2.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.od_costing,
            costing_method_arguments={"cost_electricity_flow": True},
        )
    elif m.erd_type2 == "pump_as_turbine":
        desal.SP_ERD.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)
    else:
        raise ConfigurationError(
            f"erd_type2 was {m.erd_type2}, costing only implemented "
            "for pressure_exchanger or pump_as_turbine"

        )
    m.fs.zo_costing.cost_flow(desal.FP_P1.work_mechanical[0], "electricity")
    if m.erd_type1 == "pressure_exchanger":
        m.fs.zo_costing.cost_flow(desal.FP_P2.work_mechanical[0], "electricity")
    elif m.erd_type1 == "pump_as_turbine":
        m.fs.zo_costing.cost_flow(desal.FP_ERD.work_mechanical[0], "electricity")
    else:
        raise ConfigurationError(
            f"erd_type1 was {m.erd_type1}, costing only implemented "
            "for pressure_exchanger or pump_as_turbine"
        )
    m.fs.zo_costing.cost_flow(desal.SP_P1.work_mechanical[0], "electricity")
    if m.erd_type2 == "pressure_exchanger":
        m.fs.zo_costing.cost_flow(desal.SP_P2.work_mechanical[0], "electricity")
    elif m.erd_type2 == "pump_as_turbine":
        m.fs.zo_costing.cost_flow(desal.SP_ERD.work_mechanical[0], "electricity")
    else:
        raise ConfigurationError(
            f"erd_type2 was {m.erd_type2}, costing only implemented "
            "for pressure_exchanger or pump_as_turbine"
        )

    psttrt.IX.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    psttrt.storage.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)

    if m.elec_type == "PEM":
        H2x.ER.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)
        H2x.ER.costing.factor_membrane_replacement.fix(0.22)
        H2x.ER.costing.membrane_unit_cost.fix(29)
        H2x.ER.costing.anode_unit_cost.fix(600)
        H2x.ER.costing.cathode_unit_cost.fix(800)
        H2x.Compressor_O2 = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.od_costing,
            costing_method_arguments={"cost_electricity_flow": True},
        )
        H2x.Compressor_H2 = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.od_costing,
            costing_method_arguments={"cost_electricity_flow": True},
        )
    if m.elec_type == "AWE":
        H2x.ER.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.od_costing)
        H2x.ER.costing.factor_membrane_replacement.fix(0.33)
        H2x.ER.costing.membrane_unit_cost.fix(27)
        H2x.ER.costing.anode_unit_cost.fix(400)
        H2x.ER.costing.cathode_unit_cost.fix(700)
        H2x.Compressor_O2 = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.od_costing,
            costing_method_arguments={"cost_electricity_flow": True},
        )
        H2x.Compressor_H2 = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.od_costing,
            costing_method_arguments={"cost_electricity_flow": True},
        )
    H2xstor.O2.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    H2xstor.H2.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)

    #Aggregate unit level costs and calculate overall process costs
    m.fs.zo_costing.cost_process()
    
    
    feed_flowrate = m.fs.feed.flow_vol[0]
    m.fs.zo_costing.add_electricity_intensity(feed_flowrate)
    m.fs.od_costing.add_specific_energy_consumption(feed_flowrate)
    m.fs.od_costing.electricity_cost = value(m.fs.zo_costing.electricity_cost)
    m.fs.od_costing.base_currency = pyunits.USD_2020
    m.fs.od_costing.utilization_factor.fix(0.85) 
    zo_crf = m.fs.zo_costing.capital_recovery_factor
    m.fs.od_costing.capital_recovery_factor.fix(value(zo_crf))
    m.fs.od_costing.wacc.unfix()

    m.fs.od_costing.cost_process()

    m.fs.specific_energy_intensity = Expression(
        expr=(
            m.fs.zo_costing.electricity_intensity
            + m.fs.od_costing.specific_energy_consumption
        ),
        doc="Specific energy consumption of the treatment configuration on a feed flowrate basis [kWh/m3] "
    )

    #Annual disposal of waste
    m.fs.brine_disposal_cost = Expression(
        expr=(
            m.fs.zo_costing.utilization_factor
            * (
                m.fs.zo_costing.waste_disposal.cost
                * pyunits.convert(
                    desal.FP_disposal.properties[0].flow_vol
                    + desal.SP_disposal.properties[0].flow_vol,
                    to_units=pyunits.m**3 /m.fs.zo_costing.base_period,


                )
            )
        ),
        doc="Cost of disposing of brine waste",
    )
    #Annual disposal of sludge 
    m.fs.sludge_disposal_cost = Expression(
        expr=(
            m.fs.zo_costing.utilization_factor
            * (
                m.fs.zo_costing.waste_disposal_cost
                * pyunits.convert(
                    prtrt.mf_disposal.properties[0].flow_vol
                    + prtrt.gac_disposal.properties[0].flow_vol,
                    to_units=pyunits.m**3 / m.fs.zo_costing.base_period,

                )
            )
        ),
        doc="Cost of disposing pretreatment sludge"
    )
    #Annual water recovery
    m.fs.water_recovery_revenue = Expression(
        expr=(
            m.fs.zo_costing.utilization_factor
            * m.fs.zo_costing.recovered_water_cost
            * pyunits.convert(
                prtrt.gac.treated.properties[0].flow_vol
                + desal.FP_RO.permeate.properties[0].flow_vol
                + desal.SP_RO.permeate.properties[0].flow_vol,
                + psttrt.IX.outlet.properties[0].flow_vol,
                to_units=pyunits.m**3 / m.fs.zo_costing.base_period,
            )
        ),
        doc="Savings from water recovered",
    )
    #Annual lithium recovery
    m.fs.Li_recovery_revenue = Expression(
        expr=(
            m.fs.zo_costing.utilization_factor
            * m.fs.zo_costing.recovered_Li_cost
            *pyunits.convert(
                (Lix.IX.inlet.properties[0].conc_mass_phase_comp["Liq", "Li_+"]
                - Lix.IX.outlet.properties[0].conc_mass_phase_comp["Liq", "Li_+"])
                * Lix.IX.outlet.properties[0].flow_vol,
                pyunits.m**3 / m.fs.zo_costing.base_period,

            )
        ),
        doc="Savings from lithium recovered"
    )
    m.fs.H2_recovery_revenue = Expression(
        expr=(
            m.fs.zo_costing.utilization_factor
            * m.fs.zo_costing.recovered_H2_cost
            * pyunits.convert(
                H2x.prod_H2.inlet.properties[0].flow_vol,
                to_units=pyunits.m**3 / m.fs.zo_costing.base_period,
            )
        ),
        doc="Saving from hydrogen recovered"
    )
    m.fs.O2_recovery_revenue = Expression(
        expr=(
            m.fs.zo_costing.utilization_factor
            * m.fs.zo_costing.recovered_O2_cost
            * pyunits.convert(
                H2x.prod_O2.inlet.properties[0].flow_vol,
                to_units=pyunits.m**3 / m.fs.zo_costing.base_period
            )
        )
    )

    @m.fs.Expression(doc="Total capital cost of the treatment train")
    def total_capital_cost(b):
        return (
            pyunits.convert(
                m.fs.zo_costing.total_capital_cost, to_units=pyunits.USD_2020
            ) + pyunits.convert(
                m.fs.od_costing.total_capital_cost, to_units=pyunits.USD_2020
            )
        )
    
    @m.fs.Expression(doc="Total operating cost of the treatment train")
    def total_operating_cost(b):
        return (
            pyunits.convert(
                m.fs.zo_costing.total_fixed_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year,
            )
            + pyunits.convert(
                m.fs.zo_costing.total_variable_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year,
            )
            + pyunits.convert(
                m.fs.od_costing.total_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year,
            )
        )
    
    @m.fs.Expression(doc="Total cost of water recovery and brine/sludge disposed")
    def water_externalities(b):
        return pyunits.convert(
        m.fs.water_recovery_revenue
        - m.fs.brine_disposal_cost
        - m.fs.sludge_disposal_cost,
        to_units=pyunits.USD_2020 / pyunits.year,
        )
    
    @m.fs.Expression(doc="Total cost of Li recovery and brine/sludge disposed")
    def Li_externalities(b):
        return pyunits.convert(
        m.fs.Li_recovery_revenue
        - m.fs.brine_disposal_cost
        - m.fs.sludge_disposal_cost,
        to_units=pyunits.USD_2020 / pyunits.year,
        )
    
    @m.fs.Expression(doc="Total cost of H2 and O2 recovery and brine/sludge disposed")
    def H2O2_externalities(b):
        return pyunits.convert(
            m.fs.H2_recovery_revenue
            + m.fs.O2_recovery_revenue
            - m.fs.brine_disposal_cost
            - m.fs.sludge_disposal_cost,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    
    @m.fs.Expression(doc="Total cost of Li, H2, and O2 recovery and brine/sludge disposed")
    def LiH2O2_externalities(b):
        return pyunits.convert(
            m.fs.Li_recovery_revenue
            + m.fs.H2_recovery_revenue
            + m.fs.O2_recovery_revenue
            - m.fs.brine_disposal_cost
            - m.fs.sludge_disposal_cost,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    
    @m.fs.Expression(doc="Levelized cost of treatment with respect to volumetric feed flow (water only)")
    def LCOT_water(b): 
        return (
            b.total_capital_cost * b.zo_costing.capital_recovery_factor
            + b.total_operating_cost
            - b.water_externalities

        ) / (
            pyunits.convert(
                b.feed.properties[0].flow_vol,
                to_units=pyunits.m**3 / pyunits.year,
            )
            * b.zo_costing.utilization_factor
        )
    
    @m.fs.Expression(doc="Levelized cost of treatment with respect to volumetric feed flow (water and Li only)")
    def LCOT_Li(b): 
        return (
            b.total_capital_cost * b.zo_costing.capital_recovery_factor
            + b.total_operating_cost 
            - b.water_externalities
            - b.Li_externalities
        ) / (
            pyunits.convert(
                b.feed.properties[0].flow_vol,
                to_units=pyunits.m**3 / pyunits.year,
            )
            * b.zo_costing.utilization_factor
        )
    @m.fs.Expression(doc="Levelized cost of treatment with respect to volumetric feed flow (water, H2 and O2 only)")
    def LCOT_H2O2(b): 
        return (
            b.total_capital_cost * b.zo_costing_capital_recovery_factor
            + b.total_operating_cost
            - b.water_externalities
            - b.H2O2_externalities
        ) / (
            pyunits.convert(
                b.feed.properties[0].flow_vol,
                to_units=pyunits.m**3 / pyunits.year,
            )
            * b.zo_costing.utilization_factor
        )
    @m.fs.Expression(doc="Levelized cost of treatment with respect to volumetric feed flow (water, Li, H2 and O2)")
    def LCOT_LiH2O2(b): 
        return (
            b.total_capital_cost * b.zo_costing_capital_recovery_factor
            + b.total_operating_cost
            - b.water_externalities
            - b.Li_externalities
            - b.H2O2_externalities
        ) / (
            pyunits.convert(
                b.feed.properties[0].flow_vol,
                to_units=pyunits.m**3 / pyunits.year,
            )
            * b.zo_costing.utilization_factor
        )
    
    @m.fs.Expression(doc="Levelized cost of water with respect to volumetric permeate flow (water only)")
    def LCOW_water(b):
        return ( 
            b.total_capital_cost * b.zo_costing.capital_recovery_factor
            + b.total_operating_cost 
            + m.fs.brine_disposal_cost
            + m.fs.sludge_disposal_cost
        ) / (
            pyunits.convert(
                prtrt.gac.treated.properties[0].flow_vol,
                + desal.FP_RO.permeate.properties[0].flow_vol,
                + desal.SP_RO.permeate.properties[0].flow_vol,
                + psttrt.IX.outlet.properties[0].flow_vol,
                to_units=pyunits.m**3 / pyunits.year,

            )
            * b.zo_costing.utilization_factor
        )
    @m.fs.Expression(doc="Levelized cost of water with respect to volumetric permeate flow (Li extraction)")
    def LCOW_Li(b):
        return (
            b.total_capital_cost * b.zo_costing.capital_recovery_factor
            + b.total_operating_cost
            + m.fs.brine_disposal_cost
            + m.fs.sludge_disposal_cost
        ) / (
            pyunits.convert(
                prtrt.gac.treated.properties[0].flow_vol,
                + Lix.IX.outlet.properties[0].flow_vol,
                + desal.FP_RO.permeate.properties[0].flow_vol,
                + desal.SP_RO.permeate.properties[0].flow_vol,
                + psttrt.IX.outlet.properties[0].flow_vol,
                to_units=pyunits.m**3 / pyunits.year,
            )
            * b.zo_costing.utilization_factor
        )
    
    assert_units_consistent(m)

    #Set costing scalar factors
    iscale.set_scaling_factor(m.fs.zo_costing.total_capital_cost, 1e-4)
    iscale.set_scaling_factor(m.fs.od_costing.total_capital_cost, 1e-6)

    iscale.set_scaling_factor(m.fs.zo_costing.total_operating_cost, 1e-4)
    iscale.set_scaling_factor(m.fs.zo_costing.total_operating_cost, 1e-6)

    for block in m.fs.component_objects(Block, descend_into=True):
        if isinstance(block, UnitModelBlockData) and hasattr(block, "costing"):
            iscale.set_scaling_factor(block.costing.capital_cost, 1e-4)
    return
    
         
def initialize_costing(m):
        m.fs.zo_costing.initialize()
        m.fs.od_costing.initialize()

def display_costing(m):
    capex = value(pyunits.convert(m.fs.total_capital_cost, to_units=pyunits.MUSD_2020))

    prtrt_capex = value(
        pyunits.convert(
            m.fs.pretreatment.intake.costing.capital_cost,
            + m.fs.pretreatment.ferric_chloride_addition.costing.capital_cost,
            + m.fs.pretreatment.chlorination.costing.capital_cost,
            + m.fs.pretreatment.static_mixer.costing.capital_cost,
            + m.fs.pretreatment.storage.costing.capital_cost,
            + m.fs.pretreatment.screening.costing.capital_cost, 
            + m.fs.pretreatment.coag_and_floc.costing.capital_cost,
            + m.fs.pretreatment.sedimentation.costing.capital_cost,
            + m.fs.pretreatment.flotation.costing.capital_cost,
            + m.fs.pretreatment.gravity_basin.costing.capital_cost,
            + m.fs.pretreatment.mfiltration.costing.capital_cost,
            + m.fs.pretreatment.mbackwash_pump.costing.capital_cost,
            + m.fs.pretreatment.anti_scalant_addition.costing.capital_cost,
            + m.fs.pretreatment.cfiltration.costing.capital_cost,
            + m.fs.pretreatment.gac.costing.capital_cost,
            + m.fs.pretreatment.gbackwash_pump.costing.capital_cost,
            + m.fs.pretreatment.mf_disposal.costing.capital_cost,
            + m.fs.pretreatment.gac_disposal.costing.capital_cost,
            to_units=pyunits.MUSD_2020,
        )
    )

    Li_capex = value(
        pyunits.convert( 
            m.fs.Liextraction.P1.costing.capital_cost,
            + m.fs.Liextraction.IX.costing.capital_cost,
            + m.fs.Listorage.storage.costing.capital_cost,
            to_units=pyunits.MUSD_2020,
        )
    )
    if m.erd_type1 == "pressure_exchanger" and m.erd_type2 == "pressure_exchanger":
        desal_capex = value(
            pyunits.convert(
                m.fs.desalination.FP_S1.costing.capital_cost,
                + m.fs.desalination.FP_P1.costing.capital_cost,
                + m.fs.desalination.FP_P2.costing.capital_cost,
                + m.fs.desalination.FP_M1.costing.capital_cost,
                + m.fs.desalination.FP_RO.costing.capital_cost,
                + m.fs.desalination.FP_PXR.costing.capital_cost,
                + m.fs.desalination.SP_S1.costing.capital_cost,
                + m.fs.desalination.SP_P1.costing.capital_cost,
                + m.fs.desalination.SP_P2.costing.capital_cost,
                + m.fs.desalination.SP_M1.costing.capital_cost,
                + m.fs.desalination.SP_RO.costing.capital_cost,
                + m.fs.desalination.SP_PXR.costing.capital_cost,
                to_units=pyunits.MUSD_2020,
            )   
        )
    elif m.erd_type1 == "pressure_exchanger" and m.erd_type2 == "pump_as_turbine":
        desal_capex = value(
            pyunits.convert(
                m.fs.desalination.FP_S1.costing.capital_cost,
                + m.fs.desalination.FP_P1.costing.capital_cost,
                + m.fs.desalination.FP_P2.costing.capital_cost,
                + m.fs.desalination.FP_M1.costing.capital_cost,
                + m.fs.desalination.FP_RO.costing.capital_cost,
                + m.fs.desalination.FP_PXR.costing.capital_cost,
                + m.fs.desalination.SP_P1.costing.capital_cost,
                + m.fs.desalination.SP_RO.costing.capital_cost,
                + m.fs.desalination.SP_ERD.costing.capital_cost,
                to_units=pyunits.MUSD_2020,
            )
        )
    elif m.erd_type1 == "pump_as_turbine" and m.erd_type2 == "pressure_exchanger":
        desal_capex = value(
            pyunits.convert(
                m.fs.desalination.FP_P1.costing.capital_cost,
                + m.fs.desalination.FP_RO.costing.capital_cost,
                + m.fs.desalination.FP_ERD.costing.capital_cost,
                + m.fs.desalination.SP_S1.costing.capital_cost,
                + m.fs.desalination.SP_P1.costing.capital_cost,
                + m.fs.desalination.SP_P2.costing.capital_cost,
                + m.fs.desalination.SP_M1.costing.capital_cost,
                + m.fs.desalination.SP_RO.costing.capital_cost,
                + m.fs.desalination.SP_PXR.costing.capital_cost,
                to_units=pyunits.MUSD_2020,
            )
        )
    elif m.erd_type1 == "pump_as_turbine" and m.erd_type2 == "pump_as_turbine":
        desal_capex = value(
            pyunits.convert(
                m.fs.desalination.FP_P1.costing.capital_cost,
                + m.fs.desalination.FP_RO.costing.capital_cost,
                + m.fs.desalination.FP_ERD.costing.capital_cost,
                + m.fs.desalination.SP_P1.costing.capital_cost,
                + m.fs.desalination.SP_RO.costing.capital_cost,
                + m.fs.desalination.SP_ERD.costing.capital_cost,
                to_units=pyunits.MUSD_2020,
            )
        )
    else: 
        pass

    if m.elec_type == "AWE":
        H2_capex = value(
            pyunits.convert(
                m.fs.H2extraction.ER.costing.capital_cost, 
                + m.fs.H2extraction.Compressor_O2.costing.capital_cost,
                + m.fs.H2extraction.Compressor_H2.costing.capital_cost,
                + m.fs.H2storage.O2.costing.capital_cost,
                + m.fs.H2storage.H2.costing.capital_cost,
                to_units=pyunits.MUSD_2020,              
            )
        )

    if m.elec_type == "PEM":
        H2_capex = value(
            pyunits.convert(
                m.fs.H2extraction.ER.costing.capital_cost,
                + m.fs.H2extraction.Compressor_O2.costing.capital_cost,
                + m.fs.H2extraction.Compressor_H2.costing.capital_cost,
                + m.fs.H2storage.O2.costing.capital_cost,
                + m.fs.H2storage.H2.costing.capital_cost,
                to_units=pyunits.MUSD_2020,
            )
        )
    psttrt_capex = value(
        pyunits.convert(
            m.fs.posttreatment.IX.costing.capital_cost,
            to_units=pyunits.USD_2020,
        )
    )

    opex = value(pyunits.convert(m.fs.total_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
    
    prtrt_opex = value(
        pyunits.convert(
            m.fs.pretreatment.intake.costing.total_operating_cost,
            + m.fs.pretreatment.ferric_chloride_addition.costing.total_operating_cost,
            + m.fs.pretreatment.chlorination.costing.total_operating_cost,
            + m.fs.pretreatment.static_mixer.costing.total_operating_cost,
            + m.fs.pretreatment.storage.costing.total_operating_cost,
            + m.fs.pretreatment.screening.costing.total_operating_cost,
            + m.fs.pretreatment.coag_and_flock.costing.total_operating_cost,
            + m.fs.pretreatment.sedimentation.costing.total_operating_cost,
            + m.fs.pretreatment.flotation.costing.total_operating_cost,
            + m.fs.pretreatment.gravity_basin.costing.total_operating_cost,
            + m.fs.pretreatment.mfiltration.costing.total_operating_cost,
            + m.fs.pretreatment.mbackwash_pump.costing.total_operating_cost,
            + m.fs.pretreatment.anti_scalant_addition.costing.total_operating_cost,
            + m.fs.pretreatment.cfiltration.costing.total_operating_cost, 
            + m.fs.pretreatment.gac.costing.total_operating_cost, 
            + m.fs.pretreatment.gbackwash_pump.costing.total_operating_cost,
            + m.fs.pretreatment.mf_disposal.costing.total_operating_cost, 
            + m.fs.pretreatment.gac_disposal.costing.total_operating_cost, 
            to_units=pyunits.MUSD_2020 / pyunits.year,
        )
    )

    Li_opex=  value(
        pyunits.convert( 
            m.fs.Liextraction.P1.costing.total_operating_cost,
            + m.fs.Liextraction.IX.costing.total_operating_cost,
            + m.fs.Listorage.storage.costing.total_operating_cost,
            to_units=pyunits.MUSD_2020 / pyunits.year,
        )
    )

    if m.erd_type1 == "pressure_exchanger" and m.erd_type2 == "pressure_exchanger":
        desal_opex = value(
            pyunits.convert(
                m.fs.desalination.FP_S1.costing.total_operating_cost,
                + m.fs.desalination.FP_P1.costing.total_operating_cost,
                + m.fs.desalination.FP_P2.costing.total_operating_cost,
                + m.fs.desalination.FP_M1.costing.total_operating_cost,
                + m.fs.desalination.FP_RO.costing.total_operating_cost,
                + m.fs.desalination.FP_PXR.costing.total_operating_cost,
                + m.fs.desalination.SP_S1.costing.total_operating_cost,
                + m.fs.desalination.SP_P1.costing.total_operating_cost,
                + m.fs.desalination.SP_P2.costing.total_operating_cost,
                + m.fs.desalination.SP_M1.costing.total_operating_cost,
                + m.fs.desalination.SP_RO.costing.total_operating_cost,
                + m.fs.desalination.SP_PXR.costing.total_operating_cost,
                to_units=pyunits.MUSD_2020 / pyunits.year,
            )   
        )
    elif m.erd_type1 == "pressure_exchanger" and m.erd_type2 == "pump_as_turbine":
        desal_opex = value(
            pyunits.convert(
                m.fs.desalination.FP_S1.costing.total_operating_cost,
                + m.fs.desalination.FP_P1.costing.total_operating_cost,
                + m.fs.desalination.FP_P2.costing.total_operating_cost,
                + m.fs.desalination.FP_M1.costing.total_operating_cost,
                + m.fs.desalination.FP_RO.costing.total_operating_cost,
                + m.fs.desalination.FP_PXR.costing.total_operating_cost,
                + m.fs.desalination.SP_P1.costing.total_operating_cost,
                + m.fs.desalination.SP_RO.costing.total_operating_cost,
                + m.fs.desalination.SP_ERD.costing.total_operating_cost,
                to_units=pyunits.MUSD_2020 / pyunits.year,
            )
        )
    elif m.erd_type1 == "pump_as_turbine" and m.erd_type2 == "pressure_exchanger":
        desal_opex = value(
            pyunits.convert(
                m.fs.desalination.FP_P1.costing.total_operating_cost,
                + m.fs.desalination.FP_RO.costing.total_operating_cost,
                + m.fs.desalination.FP_ERD.costing.total_operating_cost,
                + m.fs.desalination.SP_S1.costing.total_operating_cost,
                + m.fs.desalination.SP_P1.costing.total_operating_cost,
                + m.fs.desalination.SP_P2.costing.total_operating_cost,
                + m.fs.desalination.SP_M1.costing.total_operating_cost,
                + m.fs.desalination.SP_RO.costing.total_operating_cost,
                + m.fs.desalination.SP_PXR.costing.total_operating_cost,
                to_units=pyunits.MUSD_2020 / pyunits.year,
            )
        )
    elif m.erd_type1 == "pump_as_turbine" and m.erd_type2 == "pump_as_turbine":
        desal_opex = value(
            pyunits.convert(
                m.fs.desalination.FP_P1.costing.total_operating_cos,
                + m.fs.desalination.FP_RO.costing.total_operating_cost,
                + m.fs.desalination.FP_ERD.costing.total_operating_cost,
                + m.fs.desalination.SP_P1.costing.total_operating_cost,
                + m.fs.desalination.SP_RO.costing.total_operating_cost,
                + m.fs.desalination.SP_ERD.costing.total_operating_cost,
                to_units=pyunits.MUSD_2020 / pyunits.year,
            )
        )
    else: 
        pass

    if m.elec_type == "AWE":
        H2_opex = value(
            pyunits.convert(
                m.fs.H2extraction.ER.costing.total_operating_cost, 
                + m.fs.H2extraction.Compressor_O2.costing.total_operating_cost,
                + m.fs.H2extraction.Compressor_H2.costing.total_operating_cost,
                + m.fs.H2storage.O2.costing.total_operating_cost,
                + m.fs.H2storage.H2.costing.total_operating_cost,
                to_units=pyunits.MUSD_2020 / pyunits.year,              
            )
        )

    if m.elec_type == "PEM":
        H2_opex = value(
            pyunits.convert(
                m.fs.H2extraction.ER.costing.total_operating_cost,
                + m.fs.H2extraction.Compressor_O2.costing.total_operating_cost,
                + m.fs.H2extraction.Compressor_H2.costing.total_operating_cost,
                + m.fs.H2storage.O2.costing.total_operating_cost,
                + m.fs.H2storage.H2.costing.total_operating_cost,
                to_units=pyunits.MUSD_2020 / pyunits.year,
            )
        )

    psttrt_opex = value(
        pyunits.convert(
            m.fs.posttreatment.IX.costing.total_operating_cost,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    H2O_prod_externalities = value(
        pyunits.convert(
            m.fs.water_externalities, to_units=pyunits.MUSD_2020 / pyunits.year
        )
    )

    H2OLi_prod_externalities = value(
        pyunits.convert(
            m.fs.Li_externalities, to_units=pyunits.MUSD_2020 / pyunits.year
        )
    )
    H2O2_prod_externalities = value(
        pyunits.convert(
            m.fs.H2O2_externalities, to_units=pyunits.MUSD_2020 / pyunits.year
        )
    )

    LiH2O2_prod_externalities = value(
        pyunits.convert(
            m.fs.LiH2O2_externalities, to_units=pyunits.MUSD_2020 / pyunits.year
        )
    )
    wrr = value(
        pyunits.convert(
            m.fs.water_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year
        )
    )
    Lirr = value(
        pyunits.convert(
            m.fs.Li_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year
        )
    )
    H2rr = value(
        pyunits.convert(
            m.fs.H2_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year
        )
    )
    O2rr = value(
        pyunits.convert(
            m.fs.O2_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year
        )
    )
    sdc = value(
        pyunits.convert(m.fs.sludge_disposal_cost, to_units=pyunits.USD_2020 / pyunits.year)
    )

    bdc = value(
        pyunits.convert(m.fs.brine_disposal_cost, to_units=pyunits.USD_2020 / pyunits.year)
    )

    feed_flowrate = value(
        pyunits.convert(m.fs.feed.properties[0].flow_vol, to_units=m**3 / pyunits.hr)
    )

    capex_norm = (
        value(pyunits.convert(m.fs.total_capital_cost, to_units=pyunits.USD_2020))
        / feed_flowrate
    )

    annual_investment = value(
        pyunits.convert(
            m.fs.total_capital_cost*m.fs.zo_costing.capital_recovery_factor
            + m.fs.total_operating_cost,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    opex_fraction = (
        100
        * value(
            pyunits.convert(
                m.fs.total_operating_cost, to_units=pyunits.USD_2020 / pyunits.year
            )
        )
        / annual_investment
    )

    lcot_water = value(pyunits.convert(m.fs.LCOT_water, to_units=pyunits.USD_2020 / pyunits.m**3))
    lcot_Li = value(pyunits.convert(m.fs.LCOT_Li, to_units=pyunits.USD_2020 / pyunits.m**3))
    lcot_h2o2 = value(pyunits.convert(m.fs.LCOT_H2O2, to_units=pyunits.USD_2020 / pyunits.m**3))
    lcot_Lih2o2 = value(pyunits.convert(m.fs.LCOT_LiH2O2, to_units=pyunits.USD_2020 / pyunits.m**3))

    lcow_water = value(pyunits.convert(m.fs.LCOT_water, to_units=pyunits.USD_2020 / pyunits.m**3))
    lcow_Li = value(pyunits.convert(m.fs.LCOT_Li, to_units=pyunits.USD_2020 / pyunits.m**3))

    sec=m.fs.specific_energy_intensity()

    print("\n System Costing Metrics:")
    print(f"\nTotal Capital Cost: {capex:.4f} M$")
    print(f"Pretreatment Capital Cost: {prtrt_capex:.4f} $")
    print(f"Lithium Extraction Capital Cost: {Li_capex: 4f} $")
    print(f"Desalination Capital Cost: {desal_capex: 4f} $")
    print(f"Ion Exchange Posttreatment Capital Cost: {psttrt_capex: 4f} $")
    print(f"Hydrogen & Oxygen Extraction Capital Cost: {H2_capex: 4f} $")

    print("\n--------------Unit Capital Costs----------------\n")
    for u in m.fs.zo_costing._registered_unit_costing:
        print(
            u.name,
            " : {price:0.3f} $".format(
                price=value(pyunits.convert(u.capital_cost, to_units=pyunits.USD_2020))
            )
        )
    for z in m.fs.od_costing._registered_unit_costing:
        print(
            z.name,
            " : {price:0.3f} $".format(
                price=value(pyunits.convert(z.capital_cost, to_units=pyunits.USD_2020))
            )
        )
    
    print(f"\nTotal Operating Cost: {opex:.4f} M$/year")
    print(f"Pretreatment Operational Cost: {prtrt_opex: .4f} M$/year")
    print(f"Lithium Extraction Operational Cost: {Li_opex: .4f} M$/year")
    print(f"Desalination Operational Cost: {desal_opex: .4f} M$/year")
    print(f"Ion Exchange Posttreatment Operational Cost: {psttrt_opex: .4f} M$/year")
    print(f"Hydrogen & Oxygen Extraction Operational Cost: {H2_opex: .4f} M$/year")

    print(f"\nH2O Production Externalities: {H2O_prod_externalities: .4f} M$/year")
    print(f"H2O and Li Production Externalities: {H2OLi_prod_externalities: .4f} M$/year")
    print(f"Hydrogen and Oxygen Production Externalities: {H2O2_prod_externalities: .4f} M$/year")
    print(f"Lithium, Hydrogen, & Oxygen Production Externalities: {LiH2O2_prod_externalities: 4f} M$/year")

    print(f"\nWater Recovery Revenue: {wrr:.4f} USD/year")
    print(f"Lithium Recovery Revenue: {Lirr: .4f} USD/year")
    print(f"Hydrogen Recovery Revenue: {H2rr: .4f} USD/year")
    print(f"Oxygen Recovery Revenue: {O2rr: .4f} USD/year")

    print(f"\nDesalination Brine Disposal Cost: {bdc: .4f} USD/year")
    print(f"Pretreatment Waste Disposal Cost: {sdc: .4f} USD/year")

    print(f"\nTotal Annual Cost: {annual_investment: .4f} $/year")
    print(f"Normalized Capital Cost: {capex_norm: .4f} $/m3feed/hr")
    print(f"Opex Fraction of Annual Cost: {opex_fraction: .4f} %")

    print(f"\nClean Water Production Levelized Cost of Treatment w/ Externalities: {lcot_water: .4f} $/m3 feed")
    print(f"Water & Li Production Levelized Cost of Treatment w/ Externalities: {lcot_Li: .4f} $/m3 feed")
    print(f"Hydrogen & Oxygen Production Levelized Cost of Treatment w/ Externalities: {lcot_h2o2: .4f} $/m3 feed")
    print(f"Li, Hydrogen & Oxygen Production Levelized Cost of Treatment w/ Externalities: {lcot_Lih2o2: .4f} $/m3 feed")

    print(f"\nClean Water Production Levelized Cost of Water w/ Externalities: {lcow_water: .4f} $/m3 permeate")
    print(f"Li Production Levelized Cost of Water w/ Externalities: {lcow_Li: .4f} $/m3 permeate")

    print(f"\nSpecific Energy Intensity: {sec: .4f} kWh/m3 feed")

    
def export_to_ui():
        from watertap.ui.fsapi import FlowsheetInterface

        def noop(*args, **kwargs):
            return

        return FlowsheetInterface(
            name="Configuration 1",
            description="C1 Passive Li and H2 Extraction From Seawater",
            do_export=noop,
            do_build=noop,
            do_solve=noop,
         )


if __name__ == "__main__":
        m = main(erd_type1="pressure_exchanger", erd_type2="pressure_exchanger", elec_type="PEM")



        
    

    


    

        



    











    




    

