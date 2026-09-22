"""Configuration."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_PATHS_FILE = REPO_ROOT / "config" / "paths.yaml"

with _PATHS_FILE.open("r", encoding="utf-8") as _f:
    _PATHS = yaml.safe_load(_f)


def _resolve_path(key: str) -> Path:
    """Resolve a path from config/paths.yaml, relative to the repo root if not absolute."""
    path = Path(_PATHS[key])
    return path if path.is_absolute() else REPO_ROOT / path


NEXUS_RESULTS_PATH = _resolve_path("nexus_results_path")
SECMOD_RESULTS_PATH = _resolve_path("secmod_results_path")

DATA_DIR = _resolve_path("data_dir")
RESULTS_DIR = _resolve_path("results_dir")
FIGURES_DIR = _resolve_path("figures_dir")

RESULTS_NEXUS_DIR = RESULTS_DIR / "nexus-e"
RESULTS_SECMOD_DIR = RESULTS_DIR / "secmod"
DATA_EP2050_DIR = DATA_DIR / "ep2050+"

SCENARIO_NAME = "EV_HP_flexibility_visualisation"

UNIT_DICT = {
    "elec_price": "EUR/MWh",
    "system_cost_cost": "Mln EUR",
    "system_cost_gen": "Mln EUR",
    "cap": "GW",
    "demand_h": "GWh",
    "generation_h": "GWh",
    "curt_h": "GWh",
    "import_h": "GWh",
}

SCENARIO_NAME_MAPPING = {
    "bothflex": "EV + HP Flex",
    "evflex": "EV Flex",
    "hpflex": "HP Flex",
    "noflex": "noFlex",
    "highntc": "High INT",
    "lowntc": "Low INT",
    "noexp": "noExp",
    "exp": "Exp",
}


SCENARIO_NAME_MAPPING_SECMOD = {
    "S1_high_EV_HP": "EV + HP Flex | High INT | noExp",
    "S2_high_noEV_HP": "HP Flex | High INT | noExp",
    "S3_high_EV_noHP": "EV Flex | High INT | noExp",
    "S4_high_noEV_noHP": "noFlex | High INT | noExp",
    "S5_low_EV_HP": "EV + HP Flex | Low INT | noExp",
    "S6_low_noEV_HP": "HP Flex | Low INT | noExp",
    "S7_low_EV_noHP": "EV Flex | Low INT | noExp",
    "S8_low_noEV_noHP": "noFlex | Low INT | noExp",
}

TECH_COLOR_MAPPING = {
    "Battery (Gen)": "#964796",
    "Battery (Load)": "#964796",
    "Hydro RoR": "#222c7b",
    "Hydro Dam": "#3146be",
    "Pump (Gen)": "#728edb",
    "Pump (Load)": "#728edb",
    "PV": "#fce072",
    "PV Alpine": "#e0bd31",
    "PV Roof": "#fce072",
    "Wind Onshore": "#9af290",
    "HtP": "#79CACA",
    "HP Shift (Down)": "#ff355e",
    "EMob Shift (Down)": "#FF00FF",
    "DSM (Down)": "#E0089E",
    "Load Shed": "#424949",
    "EMob Shift (Up)": "#CD00CD",
    "HP Shift (Up)": "#fa8072",
    "DSM (Up)": "#E0A3CE",
    "Waste": "#3ea95a",
    "Biomass": "#728b36",
    "Biogas": "#638654",
    "Geothermal": "#b45f06",
    "Fossil": "#434344",
    "Nuclear": "#f13535",
    "Conventional": "#F5B76C",
    "Electrolysis": "#518888",
    "Heat Pump": "#da6868",
    "e-Mobility": "#dfdf78",
    "Gas CC-Syn": "#755943",
    "Wind Offshore": "#448733",
    "Grid Expansion": "#c4a7fd",
    "Import": "#D8D8D8",
    "Export": "#9A9A9A",
    "Import (Net)": "#D8D8D8",
}

SCENARIO_COLOR_DICT = {
    "noflex": "#9C9C9C",
    "evflex": "#84D5A3",
    "hpflex": "#EE675D",
    "bothflex": "#7C73F7",
}

SHIFT_HATCH_MAPPING = {
    "EMob Shift (Down)": "//",
    "HP Shift (Down)": "//",
    "DSM (Down)": "//",
}

SHIFT_COLOR_MAPPING = {
    "HP Shift (Down)": "#da6868",
    "HP Shift (Up)": "#d23232",
    "Heat Pump": "#da6868",
    "Heat Pump-Down": "#da6868",
    "EMob Shift (Up)": "#34836EFF",
    "EMob Shift (Down)": "#6BC9B0FF",
    "e-Mobility": "#6BC9B0FF",
    "e-Mobility-Down": "#6BC9B0FF",
    "DSM (Down)": "#E2D692FF",
    "DSM (Up)": "#D1BB3EFF",
    "Conventional": "#E2D692FF",
    "Conventional-Down": "#E2D692FF",
    "Variable load": "#8FA1EBFF",
    "Electrolysis": "#39CFDA",
}

UNIT_MAPPING_SECMOD = {
    "capacities": "MW",
    "co2_balance": "ktonCO2",
    "cost_invest": "billionCHF/year",
    "cost_operation": "billionCHF/year",
    "cost_nsd": "billionCHF/year",
    "yearly_elec_production": "TWh",
    "mobility": "MMvkm",
}

SCENARIO_MAPPING_SECMOD = {
    "Nexus_s1_bothflex_highntc_noexp",
    "Nexus_s2_hpflex_highntc_noexp",
    "Nexus_s3_evflex_highntc_noexp",
    "Nexus_s4_noflex_highntc_noexp",
    "Nexus_s5_bothflex_lowntc_noexp",
    "Nexus_s6_hpflex_lowntc_noexp",
    "Nexus_s7_evflex_lowntc_noexp",
    "Nexus_s8_noflex_lowntc_noexp",
}
