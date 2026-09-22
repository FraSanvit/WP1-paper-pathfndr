"""Run the plotting and postprocessing script."""

# %%
import json
import os
from pathlib import Path

import config as cnf
import helper as helper
import plot as plot
import postprocess as postprocess

# %% Create results

base_path = Path(cnf.NEXUS_RESULTS_PATH)
postprocess.create_results(base_path)
postprocess.friendly_secmod_results()

# %%

scenario_list = [
    "EP2050+ Zero Basis",
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]
filename = "highntc"
plot.electricity_balance_grid_split(filename, scenario_list)

scenario_list = [
    "EP2050+ Zero Basis",
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s5_bothflex_lowntc_noexp",
    "Nexus_s8_noflex_lowntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]
filename = "all"
plot.electricity_balance_grid_split(filename, scenario_list)

scenario_list = [
    "EP2050+ Zero Basis",
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s1_bothflex_highntc_exp",
    "Nexus_s4_noflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_exp",
]
filename = "exp_vs_noexp"
plot.electricity_balance_grid_split(filename, scenario_list)

# %% Winter / Summer net imports
scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]
filename = "highntc"
plot.winter_summer_import(filename, scenario_list)

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s1_bothflex_highntc_exp",
    "Nexus_s4_noflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_exp",
]
filename = "exp"
plot.winter_summer_import(scenario_list, filename)


# %% Generation, capacity, curtailment per tech
ref_scen = "Nexus_s4_noflex_highntc_noexp"

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

filename = "gen_cap_curt_highntc"

plot.generation_capacity_curt_per_tech(filename, ref_scen, scenario_list)


plot_name = "_high_vs_low"
scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
    "Nexus_s5_bothflex_lowntc_noexp",
    "Nexus_s8_noflex_lowntc_noexp",
]

ref_scen = "Nexus_s4_noflex_highntc_noexp"

filename = "gen_cap_curt_highntc" + plot_name

plot.generation_capacity_curt_per_tech(filename, ref_scen, scenario_list)

# %% Seasonal plots

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

filename = "monthly_dispatch_flex"
plot.monthly_dispatch_grid(scenario_list, filename)

# %% Costs

ref_scen = "Nexus_s4_noflex_highntc_noexp"

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

filename = "total_cost"
plot.system_cost_horizontal(scenario_list, ref_scen, filename)

ref_scen = "Nexus_s1_bothflex_highntc_noexp"

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
    "Nexus_s1_bothflex_highntc_exp",
    "Nexus_s4_noflex_highntc_exp",
]

filename = "total_cost_exp"
plot.system_cost_horizontal(scenario_list, ref_scen, filename)

ref_scen = "Nexus_s8_noflex_lowntc_noexp"

scenario_list = [
    "Nexus_s5_bothflex_lowntc_noexp",
    "Nexus_s6_hpflex_lowntc_noexp",
    "Nexus_s7_hpflex_lowntc_noexp",
    "Nexus_s8_noflex_lowntc_noexp",
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

filename = "total_cost_ntc"
plot.system_cost_horizontal(scenario_list, ref_scen, filename)

# %% Season days hourly plot

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

filename = "season_day_hourly_plot"
plot.season_days_with_total(scenario_list, filename + "_with_total_new_color")
plot.season_days(scenario_list, filename + "_new_color")


# %% Flexibility scheduling

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

filename = "shift_stacked_newcolor"
plot.dispatch(scenario_list, filename)

# %% SecMOD


scenario_list = [
    "S1_high_EV_HP",
    "S2_high_noEV_HP",
    "S3_high_EV_noHP",
    "S4_high_noEV_noHP",
]

plot.end_use_carrier_mix(scenario_list, "end_use_secmod_high", min_share=0.0012)

scenario_list = ["S5_low_EV_HP", "S6_low_noEV_HP", "S7_low_EV_noHP", "S8_low_noEV_noHP"]

plot.end_use_carrier_mix(scenario_list, "end_use_secmod_low", min_share=0.0012)

scenario_list = ["S1_high_EV_HP", "S5_low_EV_HP", "S4_high_noEV_noHP", "S8_low_noEV_noHP"]

plot.end_use_carrier_mix(scenario_list, "end_use_secmod_flex_no_flex", min_share=0.0012)

# CO2

scenario_list = [
    "S1_high_EV_HP",
    "S2_high_noEV_HP",
    "S3_high_EV_noHP",
    "S4_high_noEV_noHP",
]
filename = "co2_balance_high"
plot.co2_balance(scenario_list, filename)

scenario_list = ["S1_high_EV_HP", "S5_low_EV_HP", "S4_high_noEV_noHP", "S8_low_noEV_noHP"]
filename = "co2_balance_all"
plot.co2_balance(scenario_list, filename)


# %% nexus-e specific change wrt shifted demand

# curtailment in MWh/TWh
# capacity GW/TWh
# generation + imports (x month?)
# system costs


# duals? - savings from HP and Ev flexibility
# Trades
# load curves
# price duration curve?
# power dispatch

ref_scen = "Nexus_s4_noflex_highntc_noexp"

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

filename = "gen_cap_curt_flex_highntc"
plot.generation_capacity_curt_per_unit_flex(filename, ref_scen, scenario_list)


ref_scen = "Nexus_s4_noflex_highntc_noexp"

plot.system_cost_horizontal_unit_flex(scenario_list, ref_scen, "total_cost_flex_highntc")


# Load duration curves

ref_scen = "Nexus_s4_noflex_highntc_noexp"

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

filename = "noEXP"

plot.vres_vs_shift_subplots(scenario_list, ref_scen, filename)
plot.load_duration_curves_subplot(scenario_list, filename)

# %% Dispatch

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

periods = {
    "winter": ["2019-01-05 00:00:00", "2019-01-10 00:00:00"],
    "summer": ["2019-08-05 00:00:00", "2019-08-10 00:00:00"],
}

filename = "noExp"

plot.hourly_dispatch(
    scenario_list,
    periods,
    filename,
)

# Cumulative flexibility
plot.flexibility_3rows_updown(
    scenario_list,
    "cumu_shift_total",
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[0.5, 3, 2.5],
    hspace=0.5,
)

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
]

plot.flexibility_4cols_seasonal(
    scenario_list,
    "cumu_season_high",
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[1, 3, 2],
    hspace=0.5,
)

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s5_bothflex_lowntc_noexp",
]

plot.flexibility_4cols_seasonal(
    scenario_list,
    "cumu_season_high_vs_low",
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[1, 3, 2],
    hspace=0.5,
)

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s1_bothflex_highntc_exp",
]

plot.flexibility_4cols_seasonal(
    scenario_list,
    "cumu_season_noexp_vs_exp",
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[1, 3, 2],
    hspace=0.5,
)

# %% Import and Export by season
scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
    "Nexus_s1_bothflex_highntc_exp",
    "Nexus_s4_noflex_highntc_exp",
    "Nexus_s5_bothflex_lowntc_noexp",
    "Nexus_s8_noflex_lowntc_noexp",
]

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
    "Nexus_s5_bothflex_lowntc_noexp",
    "Nexus_s8_noflex_lowntc_noexp",
]

plot_name = "_high"
plot_name = "_high_vs_low"
plot_name = "_exp_vs_noexp"

plot.seasonal_balance(scenario_list, ["PV Alpine", "PV Roof", "Wind Onshore"], "VRES" + plot_name)
plot.seasonal_balance(scenario_list, ["Import", "Export"], "import_export" + plot_name)
plot.seasonal_balance(
    scenario_list, ["Hydro Dam", "Hydro RoR", "Pump (Gen)", "Pump (Load)"], "hydro" + plot_name
)
plot.seasonal_balance(
    scenario_list,
    ["Battery (Gen)", "Battery (Load)", "Gas CC-Syn", "Waste"],
    "flex_techs" + plot_name,
)
plot.seasonal_balance(
    scenario_list, ["DSM (Down)", "EMob Shift (Down)", "HP Shift (Down)"], "flex" + plot_name
)

# %% Seasonal hourly percentile flexibility

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

filename_prefix = "flexibility_seasonal_percentile"

plot.flexibility_seasonal_percentile(
    scenario_list,
    filename_prefix,
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[1, 3, 2],
    hspace=0.5,
    min_abs_y=None,
)

# Change in dispatch of supply/demand technologies

tech_list = ["PV Roof", "PV Alpine"]
tech_label = "PV"
name_file = "tech_percentile_{tech_label}"
ref_scenario = "Nexus_s4_noflex_highntc_noexp"

plot.tech_dispatch_seasonal_percentile(
    scenario_list,
    name_file,
    ref_scenario,
    tech_list=[
        "PV Roof",
        "PV Alpine",
    ],
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[1, 3, 2],
    hspace=0.5,
    min_abs_y=None,
)


# %% costs - shadow prices

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

ref_scenario = "Nexus_s4_noflex_highntc_noexp"

filename_prefix = "noexp"

plot.duals_seasonal_percentile(
    scenario_list,
    ref_scenario,
    filename_prefix,
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[1, 3, 2],
    hspace=0.5,
    min_abs_y=None,
)

# %%
scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

ref_scenario = "Nexus_s4_noflex_highntc_noexp"

filename = "noexp"

plot.electricity_savings_seasonal(scenario_list, ref_scenario, filename)


scenario_list = [
    "Nexus_s1_bothflex_highntc_exp",
    "Nexus_s3_evflex_highntc_exp",
    "Nexus_s2_hpflex_highntc_exp",
    "Nexus_s4_noflex_highntc_exp",
]

ref_scenario = "Nexus_s4_noflex_highntc_exp"

filename = "exp"

plot.electricity_savings_seasonal(scenario_list, ref_scenario, filename)

scenario_list = [
    "Nexus_s5_bothflex_lowntc_noexp",
    "Nexus_s7_evflex_lowntc_noexp",
    "Nexus_s6_hpflex_lowntc_noexp",
    "Nexus_s8_noflex_lowntc_noexp",
]

ref_scenario = "Nexus_s8_noflex_lowntc_noexp"

filename = "ntc"

plot.electricity_savings_seasonal(scenario_list, ref_scenario, filename)

# %% EV flexibility days of shift

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

plot.emobility_duration_shift(scenario_list, "high_ntc")

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s1_bothflex_highntc_exp",
    "Nexus_s5_bothflex_lowntc_noexp",
]

plot.emobility_duration_shift(scenario_list, "all")

# %% Max shift capapcity

scenario_list = [
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
]

plot.shift_capacity_boxplots(scenario_list, "high_ntc")

