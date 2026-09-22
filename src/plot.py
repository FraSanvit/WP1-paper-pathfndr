"""Plotting functions."""

import math
import os
import string
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import PercentFormatter
import seaborn as sns

import config as cnf
import helper as helper
import postprocess as postprocess

TRESHOLD = 1.0  # in GWh


SEASON_DAY_DICT = {
    "Winter": "2019-01-15",
    "Spring": "2019-04-15",
    "Summer": "2019-07-15",
    "Autumn": "2019-10-15",
}


def darken_color(color, amount=0.3):
    """
    Darkens the given color by multiplying (1 - amount) to the RGB channels.
    amount < 1 = darker, amount > 1 = lighter
    """
    c = np.array(mcolors.to_rgb(color))
    return np.clip(c * amount, 0, 1)


def auto_text_color(color, threshold=0.5):
    """Returns 'white' or 'black' depending on background color luminance."""
    r, g, b = mcolors.to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < threshold else "black"


def format_scenario_name(scenario):
    if not scenario.startswith("Nexus"):
        return scenario

    parts = scenario.split("_")

    # drop "Nexus" and scenario index (e.g. s1)
    parts = parts[2:]

    mapped_parts = []
    for p in parts:
        mapped_parts.append(cnf.SCENARIO_NAME_MAPPING.get(p, p))

    # first part on first line, rest on second line
    if len(mapped_parts) > 1:
        return mapped_parts[0] + "\n" + " | ".join(mapped_parts[1:])
    else:
        return mapped_parts[0]


def format_scenario_secmod(df):
    """Format scenario names for SecMOD results."""
    df = df.copy()
    df["scenario"] = (
        df["scenario"].map(cnf.SCENARIO_NAME_MAPPING_SECMOD).str.replace("|", "\n", n=1)
    )

    return df


def map_tech_names_secmod(df, col):
    """Mapping techs in SecMOD from dataframe."""
    uppercase_terms = {"dac", "pv", "ccgt", "ccs", "lt"}

    def _map_string(s):
        if not isinstance(s, str):
            return s

        parts = s.split(" ")
        parts = [p.upper() if p.lower() in uppercase_terms else p for p in parts]

        out = " ".join(parts)

        # Capitalize only the first character
        return out[0].upper() + out[1:] if out else out

    df = df.copy()
    df[col] = df[col].apply(_map_string)
    return df

def map_tech_string(s):
    uppercase_terms = {"dac", "pv", "ccgt", "ccs", "lt"}

    if not isinstance(s, str):
        return s

    parts = s.split(" ")
    parts = [p.upper() if p.lower() in uppercase_terms else p for p in parts]
    out = " ".join(parts)

    return out[0].upper() + out[1:] if out else out


def format_scenario_name_inline(scenario: str) -> str:
    """
    Format a scenario name for inline display.

    - Drops the prefix "Nexus" and scenario index (e.g., "s1") if present.
    - Maps remaining parts using cnf.SCENARIO_NAME_MAPPING.
    - Joins parts with " | ".

    Example:
        "Nexus_s1_bothflex_highntc_noexp" -> "bothflex | highntc | noexp"
    """
    if not scenario.startswith("Nexus"):
        return scenario.strip()

    parts = scenario.split("_")

    # Safety: if fewer than 3 parts, just skip "Nexus"
    parts = parts[2:] if len(parts) > 2 else parts[1:]

    mapped_parts = [cnf.SCENARIO_NAME_MAPPING.get(p, p).strip() for p in parts]

    return " | ".join(mapped_parts)


def split_scenario(s):
    """
    Expected format.

    Nexus_s1_bothflex_highntc_exp
    """
    parts = s.split("_")

    if len(parts) < 5:
        return pd.Series([None, None, None], index=["flex", "NTC", "gridExp"])

    return pd.Series(
        parts[2:5],  # discard Nexus, s1
        index=["flex", "NTC", "gridExp"],
    )


def order_scenarios(scenarios: pd.Series) -> pd.Categorical:
    # Split into non-Nexus and Nexus
    non_nexus = sorted([s for s in scenarios if not s.lower().startswith("nexus")])
    nexus = [s for s in scenarios if s.lower().startswith("nexus")]

    # Ordering dictionaries
    flex_order = {"bothflex": 1, "evflex": 2, "hpflex": 3, "noflex": 4}
    NTC_order = {"highntc": 1, "lowntc": 2}
    gridExp_order = {"noexp": 1, "exp": 2}

    def nexus_key(s):
        parts = s.split("_")
        if len(parts) < 5:
            flex, ntc, gexp = "", "", ""
        else:
            flex, ntc, gexp = parts[2:5]
        return (
            NTC_order.get(ntc.lower(), 99),
            gridExp_order.get(gexp.lower(), 99),
            flex_order.get(flex.lower(), 99),
            s,
        )

    # Sort nexus scenarios using the key
    nexus_sorted = sorted(nexus, key=nexus_key)

    # Combine non-nexus first, then nexus
    ordered = non_nexus + nexus_sorted

    # Deduplicate just in case
    seen = set()
    ordered_unique = []
    for s in ordered:
        if s not in seen:
            ordered_unique.append(s)
            seen.add(s)

    return pd.Categorical(scenarios, categories=ordered_unique, ordered=True)


def get_percentiles(arr):
    return {
        "p0": np.percentile(arr, 5, axis=0),
        "p10": np.percentile(arr, 10, axis=0),
        "p25": np.percentile(arr, 25, axis=0),
        "p50": np.percentile(arr, 50, axis=0),
        "p75": np.percentile(arr, 75, axis=0),
        "p90": np.percentile(arr, 90, axis=0),
        "p100": np.percentile(arr, 95, axis=0),
        "mean": arr.mean(axis=0),
    }


def get_stats(arr):
    if arr.size == 0:
        return None  # or raise ValueError("Empty array")
    return {
        "p0": np.percentile(arr, 5, axis=0),
        "p10": np.percentile(arr, 10, axis=0),
        "p25": np.percentile(arr, 25, axis=0),
        "p50": np.percentile(arr, 50, axis=0),
        "p75": np.percentile(arr, 75, axis=0),
        "p90": np.percentile(arr, 90, axis=0),
        "p100": np.percentile(arr, 95, axis=0),
    }


def electricity_balance(scenario_list=None):
    """Plot electricity yearly balance."""
    # ---- Read EP2050+ Zero Basis results ----
    ep2050_file = cnf.DATA_EP2050_DIR / "electricity_balance.csv"
    if not ep2050_file.exists():
        raise FileNotFoundError("electricity_balance.csv not found")

    ep2050_df = pd.read_csv(ep2050_file).fillna(0)

    # ---- Read Nexus-e results ----
    nexus_df = postprocess.electricity_yearly_balance()

    yearly_elec_balance = pd.concat([ep2050_df, nexus_df], ignore_index=True)

    yearly_elec_balance["scenario"] = order_scenarios(yearly_elec_balance["scenario"])
    yearly_elec_balance = yearly_elec_balance.sort_values("scenario")
    # ---- Remove unwanted techs ----
    tech_to_remove = [
        "Import (Net)",
        "Load (Net)",
        "Load (Total)",
        "Load Shed",
        "DSM (Down)",
        "HP Shift (Down)",
        "EMob Shift (Down)",
        "DSM (Up)",
        "HP Shift (Up)",
        "EMob Shift (Up)",
    ]
    yearly_elec_balance = yearly_elec_balance[~yearly_elec_balance["tech"].isin(tech_to_remove)]

    if scenario_list is not None:
        yearly_elec_balance = yearly_elec_balance[
            yearly_elec_balance["scenario"].isin(scenario_list)
        ]

    # ---- Reverse scenario order ----
    scenarios = yearly_elec_balance["scenario"].unique()[::-1]

    # ---- Tech ordering ----
    ordered_techs = [
        t for t in list(cnf.TECH_COLOR_MAPPING.keys()) if t in yearly_elec_balance["tech"].unique()
    ]

    fig, ax = plt.subplots(figsize=(16, 0.5 * len(scenarios) + 2))

    # ---- Consistent colors per tech ----

    supply_handles = {}
    demand_handles = {}

    # ---- Plot ----
    for i, scenario in enumerate(scenarios):
        df_s = yearly_elec_balance[yearly_elec_balance["scenario"] == scenario]

        # ---- Filter small values (|value| < 1) ----
        df_s = df_s[df_s["value"].abs() >= TRESHOLD]

        # ---- Enforce tech order AFTER filtering ----
        df_s = df_s.set_index("tech").reindex(ordered_techs).dropna().reset_index()

        left_pos = 0.0
        left_neg = 0.0

        for _, row in df_s.iterrows():
            tech = row["tech"]
            value = row["value"]
            color = cnf.TECH_COLOR_MAPPING.get(tech, "#cccccc")

            is_demand = (value < 0) or ("Load" in tech) or ("Up" in tech)

            if not is_demand:
                bar = ax.barh(i, value, left=left_pos, color=color, edgecolor="none")
                if abs(value) > 7:
                    ax.text(
                        left_pos + value / 2,
                        i,
                        f"{int(round(value))}",
                        va="center",
                        ha="center",
                        fontsize=10,
                        color=auto_text_color(color),
                    )
                left_pos += value
                supply_handles.setdefault(tech, bar[0])
            else:
                bar = ax.barh(i, -abs(value), left=left_neg, color=color, edgecolor="none")
                if abs(value) > 7:
                    ax.text(
                        left_neg - abs(value) / 2,
                        i,
                        f"{int(round(abs(value)))}",
                        va="center",
                        ha="center",
                        fontsize=10,
                        color=auto_text_color(color),
                    )
                left_neg -= abs(value)
                demand_handles.setdefault(tech, bar[0])

    # ---- Axes formatting ----
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels([format_scenario_name(s) for s in scenarios])
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("")
    ax.set_xlabel("Electricity supply (+) and demand (-) [TWh]", fontsize=14)
    ax.axvline(0, linewidth=0.8, color="black")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # ---- Remove vertical padding ----
    ax.set_ylim(-0.5, len(scenarios) - 0.5)

    # ---- Ordered legends ----
    supply_handles_ordered = [supply_handles[t] for t in ordered_techs if t in supply_handles]
    supply_labels_ordered = [t for t in ordered_techs if t in supply_handles]

    demand_handles_ordered = [demand_handles[t] for t in ordered_techs if t in demand_handles]
    demand_labels_ordered = [t for t in ordered_techs if t in demand_handles]

    supply_legend = ax.legend(
        supply_handles_ordered,
        supply_labels_ordered,
        title="Supply (+)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        title_fontsize=12,
        fontsize=12,
    )

    demand_legend = ax.legend(
        demand_handles_ordered,
        demand_labels_ordered,
        title="Demand (-)",
        loc="upper left",
        bbox_to_anchor=(1.3, 1.0),
        frameon=False,
        title_fontsize=12,
        fontsize=12,
    )

    supply_legend.get_title().set_ha("left")
    demand_legend.get_title().set_ha("left")
    supply_legend._legend_box.align = "left"
    demand_legend._legend_box.align = "left"

    ax.add_artist(supply_legend)

    plt.tight_layout()
    figures_dir = cnf.FIGURES_DIR
    figures_dir.mkdir(exist_ok=True)

    fig_path = figures_dir / "electricity_balance.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def electricity_balance_grid_split(filename, scenario_list=None):
    """Grid plot.

    Row 0 (spanning): absolute electricity balance
    Row 1 left: delta demand vs EP2050+ Zero Basis
    Row 1 right: delta supply vs EP2050+ Zero Basis
    """
    # ------------------------------------------------------------------
    # Read data
    # ------------------------------------------------------------------
    ep2050_file = cnf.DATA_EP2050_DIR / "electricity_balance.csv"
    if not ep2050_file.exists():
        raise FileNotFoundError("electricity_balance.csv not found")

    ep2050_df = pd.read_csv(ep2050_file).fillna(0)
    nexus_df = postprocess.electricity_yearly_balance()

    df = pd.concat([ep2050_df, nexus_df], ignore_index=True)

    df["scenario"] = order_scenarios(df["scenario"])
    df = df.sort_values("scenario")

    # ------------------------------------------------------------------
    # Compute net import: sum Import (Net) + Export (Net) → Import
    # ------------------------------------------------------------------
    import_export = df[df["tech"].isin(["Import", "Export"])]
    if not import_export.empty:
        # Sum import and export → new Import row
        import_sum = import_export.groupby(["scenario"], as_index=False)["value"].sum()
        import_sum["tech"] = "Import"

        # Remove old Import, keep Export but set to 0
        df = df[df["tech"] != "Import"]
        df.loc[df["tech"] == "Export", "value"] = 0

        # Append new Import
        df = pd.concat([df, import_sum], ignore_index=True)

    # ------------------------------------------------------------------
    # Remove other unwanted techs
    # ------------------------------------------------------------------
    tech_to_remove = [
        "Load (Net)",
        "Load (Total)",
        "Load Shed",
        "DSM (Down)",
        "HP Shift (Down)",
        "EMob Shift (Down)",
        "DSM (Up)",
        "HP Shift (Up)",
        "EMob Shift (Up)",
    ]
    df = df[~df["tech"].isin(tech_to_remove)]

    if scenario_list is not None:
        df = df[df["scenario"].isin(scenario_list)]

    scenarios = df["scenario"].unique()[::-1]
    all_techs = df["tech"].unique()

    ordered_techs = [
        t for t in cnf.TECH_COLOR_MAPPING.keys() if t in df["tech"].unique() and t != "Import (Net)"
    ]

    # ------------------------------------------------------------------
    # Assign type based on original sign
    # ------------------------------------------------------------------
    df["type"] = df["value"].apply(lambda x: "supply" if x >= 0 else "demand")

    # ------------------------------------------------------------------
    # Split into supply and demand for the delta plots
    # ------------------------------------------------------------------
    supply_df = df[df["type"] == "supply"].copy()
    demand_df = df[df["type"] == "demand"].copy()  # keep negative values
    supply_tech = supply_df.tech.unique()
    demand_tech = demand_df.tech.unique()

    # ------------------------------------------------------------------
    # Ensure all combinations exist
    # ------------------------------------------------------------------
    def complete_df(input_df, supply_tech, demand_tech):
        # Ensure all combinations of scenario x tech exist
        idx = pd.MultiIndex.from_product([scenarios, all_techs], names=["scenario", "tech"])
        input_df = input_df.set_index(["scenario", "tech"])
        input_df = input_df.reindex(idx, fill_value=0).reset_index()

        # Assign 'type' based on tech membership
        def assign_type(tech):
            if tech in supply_tech:
                return "supply"
            elif tech in demand_tech:
                return "demand"
            else:
                return "unknown"  # optional, if there are leftover techs

        input_df["type"] = input_df["tech"].apply(assign_type)

        input_df = input_df[["scenario", "tech", "value", "type"]]
        input_df["unit"] = "TWh"
        return input_df

    supply_df = complete_df(supply_df, supply_tech, demand_tech)
    demand_df = complete_df(demand_df, supply_tech, demand_tech)
    df = complete_df(df, supply_tech, demand_tech)

    # ------------------------------------------------------------------
    # Compute delta vs EP2050+ Zero Basis
    # ------------------------------------------------------------------
    ref_df = df[df["scenario"].str.lower() == "ep2050+ zero basis"]
    if ref_df.empty:
        raise ValueError("EP2050+ Zero Basis reference scenario not found")

    ref_supply = (
        ref_df[ref_df["type"] == "supply"]
        .groupby("tech", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "ref_value"})
    )
    ref_demand = (
        ref_df[ref_df["type"] == "demand"]
        .groupby("tech", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "ref_value"})
    )

    delta_supply = supply_df.merge(ref_supply, on="tech", how="left")
    delta_supply["value"] = delta_supply["value"] - delta_supply["ref_value"]

    delta_demand = demand_df.merge(ref_demand, on="tech", how="left")
    delta_demand["value"] = (
        -delta_demand["value"] + delta_demand["ref_value"]
    )  # negative values preserved

    df = df[~df["tech"].isin(["Import", "Export"])]
    import_export["type"] = import_export["value"].apply(lambda x: "supply" if x >= 0 else "demand")
    df = pd.concat([df, import_export], ignore_index=True)

    # ------------------------------------------------------------------
    # Helper plotting function
    # ------------------------------------------------------------------
    def plot_balance(ax, plot_df, annotate=True, treshold=9):
        supply_handles = {}
        demand_handles = {}

        for i, scenario in enumerate(scenarios):
            df_s = plot_df[plot_df["scenario"] == scenario]

            df_s = df_s.set_index("tech").reindex(ordered_techs).dropna().reset_index()

            left_pos = 0.0
            left_neg = 0.0

            for _, row in df_s.iterrows():
                tech = row["tech"]
                value = row["value"]
                color = cnf.TECH_COLOR_MAPPING.get(tech, "#cccccc")

                if value >= 0:
                    bar = ax.barh(i, value, left=left_pos, color=color, edgecolor="none")
                    if annotate and (abs(value) >= treshold):
                        ax.text(
                            left_pos + value / 2,
                            i,
                            f"{int(round(value))}",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color=auto_text_color(color),
                        )
                    left_pos += value
                    supply_handles.setdefault(tech, bar[0])
                else:
                    bar = ax.barh(i, -value, left=left_neg + value, color=color, edgecolor="none")
                    if annotate and (abs(value) >= treshold):
                        ax.text(
                            left_neg + value / 2,
                            i,
                            f"{int(round(abs(value)))}",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color=auto_text_color(color),
                        )
                    left_neg += value
                    demand_handles.setdefault(tech, bar[0])

        ax.axvline(0, linewidth=0.8, color="black")
        ax.set_ylim(-0.5, len(scenarios) - 0.5)
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        return supply_handles, demand_handles

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 0.9 * len(scenarios) + 4))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 2], hspace=0.4, wspace=0.15)

    title_letters = iter(string.ascii_lowercase)

    ax_abs = fig.add_subplot(gs[0, :])
    ax_dem = fig.add_subplot(gs[1, 0])
    ax_sup = fig.add_subplot(gs[1, 1], sharey=ax_dem, sharex=ax_dem)

    # Absolute plot
    supply_handles, demand_handles = plot_balance(ax_abs, df)
    ax_abs.set_yticks(np.arange(len(scenarios)))
    ax_abs.set_yticklabels([format_scenario_name(s) for s in scenarios])
    ax_abs.tick_params(axis="y", labelsize=12)
    ax_abs.set_xlabel("Electricity supply (+) and demand (-) [TWh]", fontsize=14)
    letter = next(title_letters)
    ax_abs.set_title(f"{letter}. Electricity balance", fontsize=15, x=0, ha="left")
    scenario_sums = df.groupby("scenario")["value"].sum()
    ax_abs.tick_params(axis="both", labelsize=12)

    # For stacked bars, sum positives and sum negatives separately
    pos_sums = df[df["value"] > 0].groupby("scenario")["value"].sum()
    neg_sums = df[df["value"] < 0].groupby("scenario")["value"].sum()

    # Find max absolute total across scenarios
    max_val = max(pos_sums.max(), -neg_sums.min())  # negative values -> take -min

    # Set symmetric x-limits with a small padding
    ax_abs.set_xlim(-max_val * 1.05, max_val * 1.05)

    # Delta demand (annotate all non-zero)
    plot_balance(ax_dem, delta_demand, annotate=True, treshold=1)
    ax_dem.set_yticks(np.arange(len(scenarios)))
    ax_dem.set_yticklabels([format_scenario_name(s) for s in scenarios])
    ax_dem.tick_params(axis="both", labelsize=12)
    ax_dem.set_xlabel("Demand change [TWh]", fontsize=13)
    letter = next(title_letters)
    ax_dem.set_title(
        f"{letter}. Electricity demand vs EP2050+ Zero Basis", fontsize=14, x=0, ha="left"
    )
    pos_sums = delta_demand[delta_demand["value"] > 0].groupby("scenario")["value"].sum()
    neg_sums = delta_demand[delta_demand["value"] < 0].groupby("scenario")["value"].sum()
    max_val = max(pos_sums.max(), -neg_sums.min())  # negative values -> take -min
    ax_dem.set_xlim(-max_val * 1.05, max_val * 1.05)

    # Delta supply (annotate all non-zero)
    plot_balance(ax_sup, delta_supply, annotate=True, treshold=1)
    ax_sup.set_yticks(np.arange(len(scenarios)))
    ax_sup.set_yticklabels([format_scenario_name(s) for s in scenarios])
    ax_sup.tick_params(axis="both", labelsize=12)
    ax_sup.set_xlabel("Supply change [TWh]", fontsize=13)
    letter = next(title_letters)
    ax_sup.set_title(
        f"{letter}. Electricity supply vs EP2050+ Zero Basis", fontsize=14, x=0, ha="left"
    )
    ax_sup.tick_params(axis="y", labelleft=False)
    pos_sums = delta_supply[delta_supply["value"] > 0].groupby("scenario")["value"].sum()
    neg_sums = delta_supply[delta_supply["value"] < 0].groupby("scenario")["value"].sum()
    max_val = max(pos_sums.max(), -neg_sums.min())  # negative values -> take -min
    ax_sup.set_xlim(-max_val * 1.05, max_val * 1.05)

    # Legends
    supply_handles_ordered = [
        supply_handles[t] for t in ordered_techs if t in supply_handles and t != "Import (Net)"
    ]
    demand_handles_ordered = [
        demand_handles[t] for t in ordered_techs if t in demand_handles and t != "Import (Net)"
    ]

    supply_labels = [t for t in ordered_techs if t in supply_handles]
    demand_labels = [t for t in ordered_techs if t in demand_handles]

    supply_legend = ax_abs.legend(
        supply_handles_ordered,
        supply_labels,
        title="Supply (+)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=12,
        title_fontsize=12,
    )

    demand_legend = ax_abs.legend(
        demand_handles_ordered,
        demand_labels,
        title="Demand (-)",
        loc="lower left",
        bbox_to_anchor=(1.02, -1.45),
        frameon=False,
        fontsize=12,
        title_fontsize=12,
    )

    supply_legend.get_title().set_ha("left")
    demand_legend.get_title().set_ha("left")
    supply_legend._legend_box.align = "left"
    demand_legend._legend_box.align = "left"

    ax_abs.add_artist(supply_legend)

    # Save
    plt.tight_layout()
    figures_dir = cnf.FIGURES_DIR
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(
        figures_dir / f"electricity_balance_grid_{filename}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def generation_capacity_curt_per_tech(filename, ref_scen, scenario_list=None):
    """Grid plot:
    Row 0 (spanning): absolute electricity balance
    Row 1 left: delta demand vs EP2050+ Zero Basis
    Row 1 right: delta supply vs EP2050+ Zero Basis
    """
    # ------------------------------------------------------------------
    # Remove other unwanted techs
    # ------------------------------------------------------------------
    tech_to_remove = [
        "Load (Net)",
        "Load (Total)",
        "Load Shed",
        "DSM (Down)",
        "HP Shift (Down)",
        "EMob Shift (Down)",
        "DSM (Up)",
        "HP Shift (Up)",
        "EMob Shift (Up)",
        "Pump (Load)",
        "Battery (Load)",
        "Import",
        "Import (Net)",
        "Export",
        "e-Mobility",
        "Conventional",
        "Heat Pump",
        "Electrolysis",
    ]
    # ------------------------------------------------------------------
    # Electricity balance
    # ------------------------------------------------------------------
    balance_df = postprocess.electricity_yearly_balance()
    balance_df["scenario"] = order_scenarios(balance_df["scenario"])
    balance_df = balance_df.sort_values("scenario")

    balance_df = balance_df[~balance_df["tech"].isin(tech_to_remove)]
    balance_ref = balance_df[balance_df["scenario"] == ref_scen]

    balance_df = helper.fill_dataframe(balance_df, "scenario", "tech", "value")

    balance_df = balance_df.merge(
        balance_ref[["tech", "value"]].rename(columns={"value": "ref_value"}), on="tech", how="left"
    )
    balance_df["delta_value"] = np.where(
        balance_df["ref_value"] != 0,
        (balance_df["value"] - balance_df["ref_value"]) / balance_df["ref_value"] * 100,
        0
    )

    balance_df["unit"] = "%"
    if scenario_list is not None:
        balance_df = balance_df[balance_df["scenario"].isin(scenario_list)]
    balance_df = balance_df[balance_df["scenario"] != ref_scen]
    # ------------------------------------------------------------------
    # Electricity capacities
    # ------------------------------------------------------------------
    cap_path = cnf.RESULTS_NEXUS_DIR
    cap_file = cap_path / "cap.csv"

    cap_df = pd.read_csv(cap_file).fillna(0)
    cap_df = cap_df[~cap_df["tech"].isin(tech_to_remove)]
    cap_df["capacity"] = cap_df["capacity"].abs()
    cap_df = helper.fill_dataframe(cap_df, "scenario", "tech", "capacity")

    cap_ref = cap_df[cap_df["scenario"] == ref_scen]
    cap_df = cap_df.merge(
        cap_ref[["tech", "capacity"]].rename(columns={"capacity": "ref_value"}),
        on="tech",
        how="left",   # <-- key change
    )

    cap_df[["capacity", "ref_value"]] = cap_df[["capacity", "ref_value"]].fillna(0)

    cap_df["delta_value"] = np.where(
        cap_df["ref_value"] != 0,
        (cap_df["capacity"] - cap_df["ref_value"]) / cap_df["ref_value"] * 100,
        0
    )
    cap_df["unit"] = "%"

    if scenario_list is not None:
        cap_df = cap_df[cap_df["scenario"].isin(scenario_list)]

    # ------------------------------------------------------------------
    # Curtailment
    # ------------------------------------------------------------------

    curt_h = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "curt_h.csv").fillna(0)
    curt_df = postprocess.timeseries_to_yearly_total(curt_h)
    curt_df = curt_df[~curt_df["tech"].isin(tech_to_remove)]

    curt_ref = curt_df[curt_df["scenario"] == ref_scen]
    curt_df = curt_df.merge(
        curt_ref[["tech", "value"]].rename(columns={"value": "ref_value"}), on="tech", how="left"
    )
    curt_df["delta_value"] = np.where(
        curt_df["ref_value"] != 0,
        (curt_df["value"] - curt_df["ref_value"]) / curt_df["ref_value"] * 100,
        0
    )

    curt_df["unit"] = "%"

    if scenario_list is not None:
        curt_df = curt_df[curt_df["scenario"].isin(scenario_list)]

    # remove ref

    cap_df = cap_df[cap_df["scenario"] != ref_scen]
    curt_df = curt_df[curt_df["scenario"] != ref_scen]

    plot_generation_and_capacity_delta(balance_df, cap_df, curt_df, filename)


def clean_tech_name(tech):
    tech = tech.replace("(", "").replace(")", "")
    tech = tech.replace(" ", "\n")
    return tech


def scenario_color(scenario):
    for key, color in cnf.SCENARIO_COLOR_DICT.items():
        if key in scenario:
            return color
    return "black"


def ordered_scenarios(df):
    ordered = []
    for key in cnf.SCENARIO_COLOR_DICT.keys():
        ordered += [s for s in df["scenario"].unique() if key in s]
    return ordered


def scenario_label(scenario):
    parts = scenario.split("_")
    if len(parts) >= 3:
        key = parts[2]
        return cnf.SCENARIO_NAME_MAPPING.get(key, key)
    return scenario


def plot_generation_and_capacity_delta(balance_df, cap_df, curt_df, filename):
    # Harmonised fontsize
    font = 12

    # Clean tech names
    for df in [balance_df, cap_df, curt_df]:
        df["tech"] = df["tech"].apply(clean_tech_name)

    techs = balance_df["tech"].unique()
    scenarios = ordered_scenarios(balance_df)

    n_scen = len(scenarios)
    n_tech = len(techs)

    x = np.arange(n_tech)
    bar_width = 0.7 / n_scen

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(0.5 * n_tech + 5, 8),
        # sharex=False,
        # sharey=True
    )

    title_letters = iter(string.ascii_lowercase)

    for ax, df, title in zip(
        axes,
        [balance_df, cap_df, curt_df],
        [
            "Electricity generation",
            "Capacity",
            "Curtailment",
        ],
    ):
        for i, scen in enumerate(scenarios):
            df_s = df[df["scenario"] == scen].set_index("tech").reindex(techs)

            ax.bar(
                x + i * bar_width - (n_scen - 1) * bar_width / 2,
                df_s["delta_value"],
                width=bar_width,
                color=scenario_color(scen),
                label=scenario_label(scen),  # if ax is axes[0] else None,
            )

        unit = df["unit"].iloc[0] if not df.empty else ""
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(
            f"Variation compared to\nnoFlex scenario\n[{unit}]", fontsize=font, labelpad=5
        )
        letter = next(title_letters)
        ax.set_title(f"{letter}. {title}", fontsize=font, x=0, ha="left")
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(techs)
        ax.tick_params(axis="x", labelsize=font)
        ax.tick_params(axis="y", labelsize=font)

        ax.set_xlim(-0.5, n_tech - 0.5)

    # Legend to the right
    legend = axes[1].legend(
        # title="Scenario",
        loc="center left",
        bbox_to_anchor=(0.75, 0.45),
        ncol=1,
        frameon=False,
        fontsize=font,
        title_fontsize=font,
    )

    legend.get_title().set_ha("left")
    legend._legend_box.align = "left"

    fig.subplots_adjust(hspace=0.7)

    fig.align_ylabels(axes)
    # plt.tight_layout()
    fig_path = cnf.FIGURES_DIR / f"{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def seasonal_balance(scenario_list, tech_list, filename):
    """Plot the import and export (or generic trade technologies) by season."""
    monthly_balance = postprocess.electricity_monthly_balance()

    season_dict = {
        "winter": ["Jan", "Feb", "Dec"],
        "spring": ["Mar", "Apr", "May"],
        "summer": ["Jun", "Jul", "Aug"],
        "autumn": ["Sep", "Oct", "Nov"],
    }

    font = 12

    # Assign seasons
    season_balance = monthly_balance.copy()
    season_balance["season"] = season_balance["month_str"].apply(
        lambda m: next((s for s, months in season_dict.items() if m in months), "Unknown")
    )

    # Aggregate
    df_all = (
        season_balance.groupby(["scenario", "tech", "unit", "season"])["value"].sum().reset_index()
    )

    # Check whether net import exists in original data
    net_import_available = "Import (Net)" in df_all["tech"].unique()

    # Reverse scenario order
    scenario_list = list(reversed(scenario_list))

    # Filter scenarios
    df_all = df_all[df_all["scenario"].isin(scenario_list)].copy()
    df_all["scenario"] = pd.Categorical(df_all["scenario"], categories=scenario_list, ordered=True)

    # Seasons present
    season_order = list(season_dict.keys())
    seasons_present = [s for s in season_order if s in df_all["season"].unique()]

    # ---- Validate tech_list against available techs ----
    available_techs = set(df_all["tech"].unique())
    requested_techs = set(tech_list)

    missing_techs = requested_techs - available_techs

    if missing_techs:
        print(f"The following technologies were not found in the data: {sorted(missing_techs)}")

    # Optionally: remove missing techs to avoid empty plotting
    tech_list = [tech for tech in tech_list if tech in available_techs]

    if not tech_list:
        raise ValueError("None of the requested technologies are available in the data.")

    # Filter to plotting techs only
    df = df_all[df_all["tech"].isin(tech_list)].copy()

    unit = df["unit"].iloc[0]

    # ---------- Symmetric x limit ----------
    df_trade = df.copy()

    pos_sum = df_trade[df_trade["value"] > 0].groupby(["scenario", "season"])["value"].sum()

    neg_sum = df_trade[df_trade["value"] < 0].groupby(["scenario", "season"])["value"].sum()

    max_ = pos_sum.max() if not pos_sum.empty else 0
    min_ = neg_sum.min() if not neg_sum.empty else 0

    if min_ < 0:
        min_value = -1.05 * max(abs(min_), max_)
        max_value = 1.05 * max(abs(min_), max_)
    else:
        min_value = min_
        max_value = 1.05 * max_

    # ---------- Figure ----------
    n_seasons = len(seasons_present)
    n_scenarios = len(scenario_list)

    fig, axes = plt.subplots(
        1, n_seasons, figsize=(3.0 * n_seasons, 0.55 * n_scenarios + 1.8), sharey=True
    )

    if n_seasons == 1:
        axes = [axes]

    y_pos = np.arange(n_scenarios)

    # ---------- Plotting ----------
    for ax, season in zip(axes, seasons_present):
        df_season = df[df["season"] == season]

        for i, scenario in enumerate(scenario_list):
            df_s = df_season[df_season["scenario"] == scenario]

            pos_base = 0
            neg_base = 0

            # stack technologies
            for tech in tech_list:
                val_series = df_s.loc[df_s["tech"] == tech, "value"]
                if val_series.empty:
                    continue

                v = val_series.values[0]
                color = cnf.TECH_COLOR_MAPPING.get(tech, "gray")

                if v >= 0:
                    ax.barh(i, v, left=pos_base, color=color)
                    pos_base += v
                else:
                    ax.barh(i, v, left=neg_base, color=color)
                    neg_base += v

            # ---------- Optional Net Import Marker ----------

            # Net = sum of all tech values for this scenario & season
            net_val = df_s["value"].sum()
            ratio = (net_val / max_value) if max_value != 0 else 0

            if ratio >= 0.35:
                alignment = "right"
                offset = -0.08 * max_value

            elif 0 <= ratio < 0.35:
                alignment = "left"
                offset = 0.08 * max_value

            elif -0.35 < ratio < 0:
                alignment = "right"
                offset = -0.08 * max_value

            else:  # ratio <= -0.35
                alignment = "left"
                offset = 0.08 * max_value

            ax.plot(
                net_val,
                i,
                marker="D",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=6,
                linestyle="None",
                zorder=10,
            )

            ax.text(
                net_val + offset,
                i,
                f"{net_val:.1f} {unit}",
                va="center",
                ha=alignment,
                fontsize=font - 1,
            )

        ax.set_title(season.capitalize(), fontsize=font)
        ax.set_xlim(min_value, max_value)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"Supply (+) / Demand (-)\n[{unit}]", fontsize=font)
        ax.set_ylim(-0.5, n_scenarios - 0.5)
        ax.grid(axis="both", linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

    # ---------- Y Labels ----------
    formatted_labels = [format_scenario_name(s) for s in scenario_list]
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(formatted_labels, fontsize=font)

    # ---------- Legend ----------
    legend_elements = [
        Patch(facecolor=cnf.TECH_COLOR_MAPPING.get(tech, "gray"), label=tech) for tech in tech_list
    ]

    if min_value < 0:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="black",
                markerfacecolor="white",
                linestyle="None",
                label="Net balance",
            )
        )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=font,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    fig_path = cnf.FIGURES_DIR / f"seasonal_{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

def winter_summer_import(scenario_list, filename):
    """Plot net import for winter and summer."""
    monthly_balance = postprocess.electricity_monthly_balance()
    winter = ['Jan', 'Feb', 'Mar','Oct', 'Nov', 'Dec']
    summer = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

    winter_df = monthly_balance[monthly_balance["month_str"].isin(winter)]
    summer_df = monthly_balance[monthly_balance["month_str"].isin(summer)]

    winter_net_imp = winter_df[winter_df["tech"].isin(["Import", "Export"])].groupby("scenario")["value"].sum().reset_index()
    winter_net_imp["season"] = "winter"

    summer_net_imp = summer_df[summer_df["tech"].isin(["Import", "Export"])].groupby("scenario")["value"].sum().reset_index()
    summer_net_imp["season"] = "summer"

    net_import = pd.concat([winter_net_imp, summer_net_imp])
    net_import = net_import[net_import["scenario"].isin(scenario_list)]

    # Pivot for easy plotting
    df_plot = net_import.pivot(index="scenario", columns="season", values="value")
    scenario_order = net_import["scenario"].unique()
    df_plot = df_plot.loc[scenario_order]

    scenarios = df_plot.index
    y_pos = np.arange(len(scenarios))

    winter_vals = df_plot["winter"].values
    summer_vals = df_plot["summer"].values

    # Determine symmetric limits based on max absolute value
    max_abs = max(abs(np.concatenate([winter_vals, summer_vals, [8.79]])))
    xmin, xmax = -max_abs, max_abs

    fig, ax = plt.subplots(figsize=(9, 3.5))

    # Plot bars
    bars_winter = ax.barh(y_pos, winter_vals, left=0, color="lightblue", label="Winter")
    bars_summer = ax.barh(y_pos, summer_vals, left=0, color="gold", label="Summer")

    # Annotate numbers at the end of each bar
    for bar in bars_winter:
        width = bar.get_width()
        ax.text(width + 0.02 * max_abs, bar.get_y() + bar.get_height()/2,
                f"{width:.1f}", va="center", ha="left", color="black", fontsize=9)

    for bar in bars_summer:
        width = bar.get_width()
        ax.text(width - 0.02 * max_abs, bar.get_y() + bar.get_height()/2,
                f"{width:.1f}", va="center", ha="right", color="black", fontsize=9)

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([format_scenario_name(s) for s in scenarios])

    # Symmetric x-limits
    ax.set_xlim(xmin*1.2, xmax*1.2)

    # Grid
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Vertical lines
    ep2050_import_w = 8.79269200140646 # winter
    ep2050_import_s = -9.156626 # summer
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(ep2050_import_w, color="blue", linestyle="--", linewidth=1.5)  # EP2050+ line
    ax.axvline(ep2050_import_s, color="#F44D00", linestyle="--", linewidth=1.5)  # EP2050+ line

    # Annotation
    ax.annotate("EP2050+ Zero Basis\nWinter import",
                xy=(ep2050_import_w, y_pos[-1]),
                xytext=(ep2050_import_w - 0.05 * max_abs, y_pos[-1]),
                color="blue",
                va="center",
                ha="right",
                rotation=0)

    ax.annotate("EP2050+ Zero Basis\nSummer import",
                xy=(ep2050_import_s, y_pos[-1]),
                xytext=(ep2050_import_s + 0.05 * max_abs, y_pos[-1]),
                color="#F44D00",
                va="center",
                ha="left",
                rotation=0)

    # Legend on the right, no frame
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 0.9))
    ax.set_title("Winter and Summer net imports")
    ax.set_xlabel("Import (+) / Export (-) [TWh]")

    plt.tight_layout()
    fig_path = cnf.FIGURES_DIR / f"net_import_winter_summer_{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()



def monthly_dispatch_grid(scenario_list, filename, font=12):
    """Monthly plot of electricity balance and shifted demand.

    Row 0: monthly generation balance (stacked bars)
    Row 1: monthly flexibility balance (stacked bars)
    """
    monthly_balance = postprocess.electricity_monthly_balance()

    # Filter generation and net_import_flexibility
    tech_to_remove = [
        "Load (Net)",
        "Load (Total)",
        "Load Shed",
        "DSM (Down)",
        "HP Shift (Down)",
        "EMob Shift (Down)",
        "DSM (Up)",
        "HP Shift (Up)",
        "EMob Shift (Up)",
        "e-Mobility",
        "Conventional",
        "Heat Pump",
        "Electrolysis",
        "Import",
        "Export",
    ]
    gen_df = monthly_balance[~monthly_balance["tech"].isin(tech_to_remove)]

    tech_to_keep = [
        "DSM (Down)",
        "HP Shift (Down)",
        "EMob Shift (Down)",
        "DSM (Up)",
        "HP Shift (Up)",
        "EMob Shift (Up)",  # "Import (Net)"
    ]
    flex_df = monthly_balance[monthly_balance["tech"].isin(tech_to_keep)]

    if scenario_list is not None:
        gen_df = gen_df[gen_df["scenario"].isin(scenario_list)]
        flex_df = flex_df[flex_df["scenario"].isin(scenario_list)]

    # ------------------------------------------------------------------
    # Scenario filtering & ordering
    # ------------------------------------------------------------------
    if scenario_list is not None:
        gen_df = gen_df[gen_df["scenario"].isin(scenario_list)]
        flex_df = flex_df[flex_df["scenario"].isin(scenario_list)]

    scenario_cat = order_scenarios(gen_df["scenario"])
    scenarios = scenario_cat.categories.tolist()

    # ------------------------------------------------------------------
    # Tech ordering (Import Net last)
    # ------------------------------------------------------------------
    ordered_techs = [
        t for t in cnf.TECH_COLOR_MAPPING.keys() if t in monthly_balance["tech"].unique()
    ]

    if "Import (Net)" in ordered_techs:
        ordered_techs = [t for t in ordered_techs if t != "Import (Net)"] + ["Import (Net)"]

    # Generation: max positive / negative across all scenarios
    gen_max = gen_df.groupby(["scenario", "month"])["value"].sum().abs().max() * 1.5
    gen_min = gen_df.groupby(["scenario", "month"])["value"].sum().abs().min() * 1.5

    # Flexibility: same
    flex_max = max(
        flex_df[flex_df["value"] >= 0].groupby(["scenario", "month"])["value"].sum().abs().max()
        * 1.1,
        1,
    )

    n_scenarios = len(scenarios)
    title_letters = iter(string.ascii_lowercase)

    height_ratios = []
    for _ in scenarios:
        height_ratios.extend([1.5, 1.1, 0.7])

    fig, axes = plt.subplots(
        nrows=3 * n_scenarios,
        ncols=1,
        figsize=(8, 4 * n_scenarios),
        # sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    for i in range(n_scenarios):
        axes[3 * i + 2].axis("off")

    # Identify shared y-axes
    gen_axes = axes[0::3]
    flex_axes = axes[1::3]

    for i, scenario in enumerate(scenarios):
        ax_gen = axes[3 * i]
        ax_flex = axes[3 * i + 1]

        gen_s = gen_df[gen_df["scenario"] == scenario]
        flex_s = flex_df[flex_df["scenario"] == scenario]

        def plot_monthly_stacked_bars(ax, df, ordered_techs):
            # Ensure month order
            df = df.sort_values("month")

            months = df["month"].drop_duplicates()
            x = range(len(months))

            pos_bottom = np.zeros(len(months))
            neg_bottom = np.zeros(len(months))

            for tech in ordered_techs:
                tech_df = df[df["tech"] == tech]
                if tech_df.empty:
                    continue

                values = tech_df.set_index("month").reindex(months)["value"].to_numpy()
                color = cnf.TECH_COLOR_MAPPING.get(tech, "grey")

                pos = np.where(values > 0, values, 0)
                neg = np.where(values < 0, values, 0)

                ax.bar(x, pos, bottom=pos_bottom, color=color, width=0.7)
                ax.bar(x, neg, bottom=neg_bottom, color=color, width=0.7)

                pos_bottom += pos
                neg_bottom += neg

            ax.set_xticks(x)
            ax.set_xticklabels([m.strftime("%b") for m in months], rotation=0)

        # --- Generation ---
        plot_monthly_stacked_bars(ax_gen, gen_s, ordered_techs)
        letter = next(title_letters)
        title = format_scenario_name_inline(scenario)
        ax_gen.set_title(f"{letter}. {title}", fontsize=font, x=0, ha="left")
        ax_gen.axhline(0, color="black", linewidth=0.8)
        ax_gen.set_ylim(-gen_min, gen_max)
        ax_gen.set_ylabel("Generation\n[TWh]", fontsize=font)
        ax_gen.set_axisbelow(True)
        ax_gen.grid(axis="y", linestyle="--", alpha=0.7)
        ax_gen.set_xticks([])
        ax_gen.set_xticklabels([])

        # --- Flexibility ---
        plot_monthly_stacked_bars(ax_flex, flex_s, ordered_techs)
        ax_flex.axhline(0, color="black", linewidth=0.8)
        ax_flex.set_ylim(-flex_max, flex_max)
        ax_flex.set_ylabel("Flexibility\n[TWh]", fontsize=font)
        ax_flex.set_axisbelow(True)
        ax_flex.grid(axis="y", linestyle="--", alpha=0.7)

    tech_sign = monthly_balance.groupby("tech")["value"].apply(lambda s: np.sign(s.mean()))

    tech_list = [
        t
        for t in ordered_techs
        if t in pd.concat([gen_df, flex_df], ignore_index=True).tech.unique()
    ]

    supply_techs = [t for t in tech_list if tech_sign.get(t, 0) >= 0]
    demand_techs = [t for t in tech_list if tech_sign.get(t, 0) < 0]

    def make_handles(techs):
        return [
            Patch(facecolor=cnf.TECH_COLOR_MAPPING[t], label=t)
            for t in techs
            if t in cnf.TECH_COLOR_MAPPING
        ]

    supply_handles = make_handles(supply_techs)
    demand_handles = make_handles(demand_techs)

    supply_legend = fig.legend(
        handles=supply_handles,
        title="Supply (+):",
        loc="upper left",
        bbox_to_anchor=(0.93, 0.85),
        frameon=False,
        fontsize=font - 1,
        title_fontsize=font,
    )

    demand_legend = fig.legend(
        handles=demand_handles,
        title="Demand (-):",
        loc="upper left",
        bbox_to_anchor=(0.93, 0.55),
        frameon=False,
        fontsize=font - 1,
        title_fontsize=font,
    )

    supply_legend.get_title().set_ha("left")
    demand_legend.get_title().set_ha("left")
    supply_legend._legend_box.align = "left"
    demand_legend._legend_box.align = "left"

    plt.subplots_adjust(hspace=0.15)
    # plt.tight_layout()
    fig_path = cnf.FIGURES_DIR / f"{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    plt.show()


def import_export():
    """Plot import export per country and month."""
    import_h = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "import_h.csv").fillna(0)
    import_total = postprocess.timeseries_to_yearly_total(import_h)
    import_total = helper.split_tech_and_country(import_total)

    TECH_COLOR_DICT = {
        "Exports": "#8F5587",
        "Imports": "#427A49",
        "Transit to": "#F292E5FF",
        "Transit from": "#78E386FF",
    }


def system_cost(scenario_list, ref_scen):
    """Plot system costs per cost type and per generation technology."""
    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    cost_cost_path = cnf.RESULTS_NEXUS_DIR / "system_cost_cost.csv"
    if not cost_cost_path.exists():
        raise FileNotFoundError("system_cost_cost.csv not found")

    cost_cost_df = pd.read_csv(cost_cost_path).fillna(0)
    cost_cost_df["cost_type"] = cost_cost_df["cost_type"].str.replace(" 2050", "", regex=False)

    cost_gen_path = cnf.RESULTS_NEXUS_DIR / "system_cost_gen.csv"
    if not cost_gen_path.exists():
        raise FileNotFoundError("system_cost_gen.csv not found")

    cost_gen_df = pd.read_csv(cost_gen_path).fillna(0)

    # -------------------------------------------------
    # Filter scenarios
    # -------------------------------------------------
    if scenario_list is not None:
        cost_cost_df = cost_cost_df[cost_cost_df["scenario"].isin(scenario_list)]
        cost_gen_df = cost_gen_df[cost_gen_df["scenario"].isin(scenario_list)]

    # -------------------------------------------------
    # Prepare pivots
    # -------------------------------------------------
    cost_type_pivot = cost_cost_df.pivot(
        index="scenario", columns="cost_type", values="cost"
    ).fillna(0)

    tech_pivot = cost_gen_df.pivot(index="scenario", columns="tech", values="cost").fillna(0)

    # Reference scenarios
    ref_cost_type = cost_type_pivot.loc[ref_scen]
    ref_tech = tech_pivot.loc[ref_scen]

    cost_type_diff = cost_type_pivot.subtract(ref_cost_type, axis=1)
    tech_diff = tech_pivot.subtract(ref_tech, axis=1)

    # -------------------------------------------------
    # Colors
    # -------------------------------------------------
    colors_cost_type = {ct: f"C{i}" for i, ct in enumerate(cost_type_pivot.columns)}

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---------- (0,0) Absolute cost by type ----------
    ax = axes[0, 0]
    bottom = np.zeros(len(cost_type_pivot))
    x = np.arange(len(cost_type_pivot))

    for ct in cost_type_pivot.columns:
        ax.bar(x, cost_type_pivot[ct], bottom=bottom, color=colors_cost_type[ct])
        bottom += cost_type_pivot[ct]

    ax.set_title("Total system cost")
    ax.set_ylabel("Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_scenario_name_inline(s) for s in cost_type_pivot.index], rotation=45, ha="right"
    )

    # ---------- (0,1) Difference cost by type ----------
    ax = axes[0, 1]
    bottom = np.zeros(len(cost_type_diff))

    for ct in cost_type_diff.columns:
        ax.bar(x, cost_type_diff[ct], bottom=bottom, color=colors_cost_type[ct])
        bottom += cost_type_diff[ct]

    ax.set_title("Difference to reference")
    ax.set_ylabel(f"Δ cost vs {format_scenario_name_inline(ref_scen)}")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_scenario_name_inline(s) for s in cost_type_diff.index], rotation=45, ha="right"
    )

    # ---------- (1,0) Absolute generation cost ----------
    ax = axes[1, 0]
    bottom = np.zeros(len(tech_pivot))

    ordered_techs = [t for t in cnf.TECH_COLOR_MAPPING.keys() if t in tech_pivot.columns]

    for t in ordered_techs:
        ax.bar(x, tech_pivot[t], bottom=bottom, color=cnf.TECH_COLOR_MAPPING.get(t, "#cccccc"))
        bottom += tech_pivot[t]

    ax.set_title("Generation cost by technology")
    ax.set_ylabel("Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_scenario_name_inline(s) for s in tech_pivot.index], rotation=45, ha="right"
    )

    # ---------- (1,1) Difference generation cost ----------
    ax = axes[1, 1]
    bottom = np.zeros(len(tech_diff))

    for t in ordered_techs:
        ax.bar(x, tech_diff[t], bottom=bottom, color=cnf.TECH_COLOR_MAPPING.get(t, "#cccccc"))
        bottom += tech_diff[t]

    ax.set_title("Difference to reference")
    ax.set_ylabel(f"Δ cost vs {format_scenario_name_inline(ref_scen)}")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_scenario_name_inline(s) for s in tech_diff.index], rotation=45, ha="right"
    )

    # -------------------------------------------------
    # Legends (one per row)
    # -------------------------------------------------
    cost_type_handles = [
        Patch(facecolor=colors_cost_type[ct], label=ct) for ct in cost_type_pivot.columns
    ]

    tech_handles = [
        Patch(facecolor=cnf.TECH_COLOR_MAPPING.get(t, "#cccccc"), label=t) for t in ordered_techs
    ]

    cost_type_legend = axes[0, 1].legend(
        handles=cost_type_handles,
        title="Cost type",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
    )

    cost_gen_legend = axes[1, 1].legend(
        handles=tech_handles,
        title="Technology",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
    )

    cost_type_legend.get_title().set_ha("left")
    cost_gen_legend.get_title().set_ha("left")
    cost_type_legend._legend_box.align = "left"
    cost_gen_legend._legend_box.align = "left"

    # -------------------------------------------------
    # Layout
    # -------------------------------------------------
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()


def system_cost_horizontal(scenario_list, ref_scen, filename):
    """Horizontal stacked system cost plots with net totals and legends including diamond markers."""
    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    cost_cost_path = cnf.RESULTS_NEXUS_DIR / "system_cost_cost.csv"
    if not cost_cost_path.exists():
        raise FileNotFoundError("system_cost_cost.csv not found")

    cost_cost_df = pd.read_csv(cost_cost_path).fillna(0)
    cost_cost_df["cost_type"] = cost_cost_df["cost_type"].str.replace(" 2050", "", regex=False)

    cost_gen_path = cnf.RESULTS_NEXUS_DIR / "system_cost_gen.csv"
    if not cost_gen_path.exists():
        raise FileNotFoundError("system_cost_gen.csv not found")

    cost_gen_df = pd.read_csv(cost_gen_path).fillna(0)

    # -------------------------------------------------
    # Filter scenarios
    # -------------------------------------------------
    if scenario_list is not None:
        cost_cost_df = cost_cost_df[cost_cost_df["scenario"].isin(scenario_list)]
        cost_gen_df = cost_gen_df[cost_gen_df["scenario"].isin(scenario_list)]

    # -------------------------------------------------
    # Order scenarios
    # -------------------------------------------------
    if scenario_list is not None:
        cost_cost_df["scenario"] = pd.Categorical(
            cost_cost_df["scenario"],
            categories=scenario_list,
            ordered=True,
        )
        cost_gen_df["scenario"] = pd.Categorical(
            cost_gen_df["scenario"],
            categories=scenario_list,
            ordered=True,
        )

    # -------------------------------------------------
    # Prepare pivots
    # -------------------------------------------------
    cost_type_pivot = cost_cost_df.pivot(
        index="scenario", columns="cost_type", values="cost"
    ).fillna(0)

    tech_pivot = cost_gen_df.pivot(index="scenario", columns="tech", values="cost").fillna(0)

    ref_cost_type = cost_type_pivot.loc[ref_scen]
    ref_tech = tech_pivot.loc[ref_scen]

    cost_type_diff = cost_type_pivot.subtract(ref_cost_type, axis=1)
    tech_diff = tech_pivot.subtract(ref_tech, axis=1)

    # -------------------------------------------------
    # Colors
    # -------------------------------------------------
    colors_cost_type = {ct: f"C{i}" for i, ct in enumerate(cost_type_pivot.columns)}

    custom_colors = {
        "Fixed Operation": "#51b6ff",
        "Investments": "#feb27b",
        "Trading": "#8fc778",
        "Variable Operation": "#e56868",
        "Grid Expansion": "#c4a7fd",
    }

    colors_cost_type = {
        ct: custom_colors.get(ct, f"C{i}") for i, ct in enumerate(cost_type_pivot.columns)
    }

    ordered_techs = [t for t in cnf.TECH_COLOR_MAPPING.keys() if t in tech_pivot.columns]
    tech_colors = {t: cnf.TECH_COLOR_MAPPING.get(t, "#cccccc") for t in ordered_techs}

    # -------------------------------------------------
    # Helper function
    # -------------------------------------------------
    def stacked_diverging_barh(ax, df, color_map, ref_totals=None):
        y = np.arange(len(df))
        left_pos = np.zeros(len(df))
        left_neg = np.zeros(len(df))

        for col in df.columns:
            values = df[col].values
            pos = np.clip(values, 0, None)
            neg = np.clip(values, None, 0)

            ax.barh(y, pos, left=left_pos, color=color_map[col])
            ax.barh(y, neg, left=left_neg, color=color_map[col])

            left_pos += pos
            left_neg += neg

        totals = df.sum(axis=1).values
        totals[np.abs(totals) < 0.005] = np.nan

        # Diamond markers for totals
        ax.plot(
            totals,
            y,
            marker="D",
            linestyle="None",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,  # optional, but looks good
            zorder=5,
        )

        # Annotate % reduction only if ref_totals is provided (second column)
        if ref_totals is not None:
            pct_change = -(totals) / ref_totals * 100
            for xi, yi, p in zip(totals, y, pct_change):
                if abs(p) >= 0.002:
                    ax.text(
                        xi,
                        yi + 0.3,
                        f"{-p:.1f}%",
                        va="center",
                        ha="center",
                        fontsize=11,
                        color="black",
                        fontweight="bold",
                    )

        ax.axvline(0, color="black", linewidth=0.8)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        return y, totals

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 1.75 * len(scenario_list)),
        sharey="row",
        gridspec_kw={"width_ratios": [1, 0.7], "height_ratios": [1, 1]},
    )

    font = 14
    title_letters = iter(string.ascii_lowercase)

    # ---------- Row 0: cost by type ----------
    y, totals_cost_abs = stacked_diverging_barh(
        axes[0, 0], cost_type_pivot, colors_cost_type, ref_totals=None
    )
    letter = next(title_letters)
    axes[0, 0].set_title(f"{letter}. Total system cost by cost type", fontsize=font, x=0, ha="left")
    axes[0, 0].set_xlabel("Total system cost [Million €]")
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels(
        [format_scenario_name(s) for s in cost_type_pivot.index], fontsize=font
    )

    ref_totals_cost = cost_type_pivot.loc[ref_scen].sum()
    y, totals_cost_diff = stacked_diverging_barh(
        axes[0, 1], cost_type_diff, colors_cost_type, ref_totals=ref_totals_cost
    )
    letter = next(title_letters)
    axes[0, 1].set_title(f"{letter}. Cost difference by cost type", fontsize=font, x=0, ha="left")
    axes[0, 1].set_xlabel("Cost difference [Million €]")
    axes[0, 1].set_yticks(y)
    axes[0, 1].tick_params(axis="y", labelleft=False)

    # ---------- Row 1: generation cost ----------
    y, totals_gen_abs = stacked_diverging_barh(
        axes[1, 0], tech_pivot[ordered_techs], tech_colors, ref_totals=None
    )
    letter = next(title_letters)
    axes[1, 0].set_title(
        f"{letter}. Total system cost by technology", fontsize=font, x=0, ha="left"
    )
    axes[1, 0].set_xlabel("Total system cost [Million €]")
    axes[1, 0].set_yticks(y)
    axes[1, 0].set_yticklabels([format_scenario_name(s) for s in tech_pivot.index], fontsize=font)

    ref_totals_gen = tech_pivot.loc[ref_scen, ordered_techs].sum()
    y, totals_gen_diff = stacked_diverging_barh(
        axes[1, 1], tech_diff[ordered_techs], tech_colors, ref_totals=ref_totals_gen
    )
    letter = next(title_letters)
    axes[1, 1].set_title(f"{letter}. Cost difference by technology", fontsize=font, x=0, ha="left")
    axes[1, 1].set_xlabel("Cost difference [Million €]")
    axes[1, 1].set_yticks(y)
    axes[1, 1].tick_params(axis="y", labelleft=False)

    # -------------------------------------------------
    # Fontsize
    # -------------------------------------------------
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            ax.tick_params(axis="y", labelsize=font)
            ax.tick_params(axis="x", labelsize=font)
            if ax.get_ylabel():
                ax.yaxis.label.set_size(font)
            if ax.get_xlabel():
                ax.xaxis.label.set_size(font)

    # -------------------------------------------------
    # Harmonise x limits per column
    # -------------------------------------------------
    factor = 1.1
    xmin_col0 = min(axes[0, 0].get_xlim()[0], axes[1, 0].get_xlim()[0]) * factor
    xmax_col0 = max(axes[0, 0].get_xlim()[1], axes[1, 0].get_xlim()[1]) * factor
    axes[0, 0].set_xlim(xmin_col0, xmax_col0)
    axes[1, 0].set_xlim(xmin_col0, xmax_col0)

    xmin_col1 = min(axes[0, 1].get_xlim()[0], axes[1, 1].get_xlim()[0]) * factor
    xmax_col1 = max(axes[0, 1].get_xlim()[1], axes[1, 1].get_xlim()[1]) * factor
    axes[0, 1].set_xlim(xmin_col1, xmax_col1)
    axes[1, 1].set_xlim(xmin_col1, xmax_col1)

    # -------------------------------------------------
    # Legends (including diamond marker)
    # -------------------------------------------------

    diamond_handle = Line2D(
        [0],
        [0],
        marker="D",
        linestyle="None",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
        label="Total cost /\nTotal cost difference",
    )

    cost_type_handles = [
        Patch(facecolor=colors_cost_type[ct], label=ct) for ct in cost_type_pivot.columns
    ]
    tech_handles = [Patch(facecolor=tech_colors[t], label=t) for t in ordered_techs]

    cost_type_legend = fig.legend(
        handles=[*cost_type_handles, diamond_handle],
        title="Cost type:",
        loc="upper left",
        bbox_to_anchor=(0.82, 0.95),
        frameon=False,
        fontsize=font,
        title_fontsize=font,
        alignment="left",
    )

    cost_gen_legend = fig.legend(
        handles=[*tech_handles, diamond_handle],
        title="Technology:",
        loc="upper left",
        bbox_to_anchor=(0.82, 0.60),
        frameon=False,
        fontsize=font,
        title_fontsize=font,
        alignment="left",
    )

    cost_type_legend.get_title().set_ha("left")
    cost_gen_legend.get_title().set_ha("left")
    cost_type_legend._legend_box.align = "left"
    cost_gen_legend._legend_box.align = "left"

    # -------------------------------------------------
    # Layout
    # -------------------------------------------------
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    fig_path = cnf.FIGURES_DIR / f"{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def season_days_with_total(scenario_list, filename):  # noqa: PLR0912, PLR0915
    """Plot one representative day per season for each scenario with Up/Down + Electrolysis columns."""
    # -------------------------------------------------
    # Read data
    # -------------------------------------------------
    supply = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "generation_h.csv").fillna(0)

    demand = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "demand_h.csv").fillna(0)

    demand = demand.drop(columns=["Electrolysis"], errors="ignore")

    # Merge hourly data
    hourly_df = supply.merge(
        demand,
        on=["timestep", "scenario"],
        suffixes=("_supply", "_demand"),
    )
    hourly_df["timestep"] = pd.to_datetime(hourly_df["timestep"])

    # -------------------------------------------------
    # Filter & order scenarios
    # -------------------------------------------------
    if scenario_list is not None:
        hourly_df = hourly_df[hourly_df["scenario"].isin(scenario_list)]
        hourly_df["scenario"] = pd.Categorical(
            hourly_df["scenario"],
            categories=scenario_list,
            ordered=True,
        )

    # -------------------------------------------------
    # Add shifted demand
    # -------------------------------------------------
    def safe_col(df, col):
        return df[col] if col in df.columns else 0.0

    hourly_df["Shifted e-Mobility"] = (
        safe_col(hourly_df, "e-Mobility")
        - safe_col(hourly_df, "EMob Shift (Up)")
        - safe_col(hourly_df, "EMob Shift (Down)")
    )
    hourly_df["Shifted Heat Pump"] = (
        safe_col(hourly_df, "Heat Pump")
        - safe_col(hourly_df, "HP Shift (Up)")
        - safe_col(hourly_df, "HP Shift (Down)")
    )
    hourly_df["Shifted Conventional"] = (
        safe_col(hourly_df, "Conventional")
        - safe_col(hourly_df, "DSM (Up)")
        - safe_col(hourly_df, "DSM (Down)")
    )

    hourly_df["PV"] = safe_col(hourly_df, "PV Alpine") + safe_col(hourly_df, "PV Roof")

    max_pv = hourly_df["PV"].max()
    min_pv = 0

    # Columns for plotting: all Up/Down + Electrolysis
    plot_cols = [c for c in hourly_df.columns if ("Up" in c or "Down" in c)] # or c == "Electrolysis"

    font = 12
    title_letters = iter(string.ascii_lowercase)

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    n_rows = len(scenario_list)
    n_cols = len(SEASON_DAY_DICT) + 1  # last column = yearly sums

    width_ratios = [1, 1, 1, 1, 0.3]  # example: last column slightly wider
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.1 * n_cols, 2.4 * n_rows),
        gridspec_kw=dict(
            wspace=0.04,
            width_ratios=width_ratios,
            hspace=0.5,
        ),
        sharex=False,
        sharey=False,
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    y_mins, y_maxs = [], []
    y_mins_year, y_maxs_year = [], []

    for i, scenario in enumerate(scenario_list):
        df_scen = hourly_df[hourly_df["scenario"] == scenario]

        for j in range(n_cols):
            ax = axes[i, j]
            ax2 = ax.twinx()

            if j < n_cols - 1:
                # Seasonal columns
                season, day_str = list(SEASON_DAY_DICT.items())[j]
                day = pd.to_datetime(day_str)
                day_df = df_scen[
                    (df_scen["timestep"] >= day)
                    & (df_scen["timestep"] < day + pd.Timedelta(days=1))
                ]

                if day_df.empty:
                    continue

                x = day_df["timestep"].dt.hour.values
                y = day_df[plot_cols].values.T

                # Separate positive and negative
                y_pos = np.clip(y, 0, None)
                y_neg = np.clip(y, None, 0)

                colors = [cnf.SHIFT_COLOR_MAPPING.get(col, "#cccccc") for col in plot_cols]

                if y_pos.any():
                    ax.stackplot(x, -y_pos, colors=colors)
                if y_neg.any():
                    ax.stackplot(x, -y_neg, colors=colors)

                # Annotate season
                ax.text(
                    0.5,
                    0.9,
                    season,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=font,
                )

                ax.set_xlim(0, 23)
                ymin, ymax = ax.get_ylim()
                y_mins.append(ymin)
                y_maxs.append(ymax)

                # x-ticks
                ax.set_xticks(range(6, 24, 6))
                ax.set_xticklabels([f"{h:02d}:00" for h in range(6, 24, 6)], fontsize=font)
                ax.tick_params(axis="x", labelsize=font - 1)

            else:
                # Exclude Electrolysis
                plot_cols_filtered = [c for c in plot_cols if c != "Electrolysis"]

                yearly_sum = df_scen[plot_cols_filtered].sum().values / -1000  # from GWh to TWh
                x = np.array([0])  # single x position
                width = 0.6  # narrow bar
                colors = [cnf.SHIFT_COLOR_MAPPING.get(c, "#cccccc") for c in plot_cols_filtered]

                # Separate positive and negative
                y_pos = np.clip(yearly_sum, 0, None)
                y_neg = np.clip(yearly_sum, None, 0)

                # Positive stacking
                bottom_pos = 0
                for val, c in zip(y_pos, colors):
                    if val > 0:
                        ax.bar(x, val, bottom=bottom_pos, width=width, color=c, edgecolor="none")
                        bottom_pos += val

                # Negative stacking
                bottom_neg = 0
                for val, c in zip(y_neg, colors):
                    if val < 0:
                        ax.bar(x, val, bottom=bottom_neg, width=width, color=c, edgecolor="none")
                        bottom_neg += val

                # ---- Add total label ----
                total_pos = y_pos.sum()

                # Place label above positive stack (or above 0 if all negative)
                y_text = bottom_pos if bottom_pos != 0 else 0
                ax.text(
                    x[0],
                    y_text + 0.1,
                    f"{total_pos:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=font-1,
                )

                ax.set_xlim(-0.5, 0.5)  # <-- this makes the bar appear narrow
                ax.set_xticks([])

                # y-limits independent
                ymin, ymax = ax.get_ylim()
                y_mins_year.append(ymin)
                y_maxs_year.append(ymax)
                ax.set_xticks([])
                ax.set_xlabel("Year", fontsize=font)
                # y-axis label on right
                ax.yaxis.tick_right()
                ymin, ymax = ax.get_ylim()
                y_mins_year.append(ymin)
                y_maxs_year.append(ymax)

            # Spine adjustments
            if j == 0:
                ax.spines["left"].set_visible(True)
                ax.spines["right"].set_visible(False)
            elif j == n_cols - 2:
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(True)
            elif j == n_cols - 1:
                ax.spines["left"].set_visible(True)
                ax.spines["right"].set_visible(True)
                # Move y-ticks to the right
                ax.yaxis.tick_right()
                ax.tick_params(axis="y", left=False)  # optionally hide left-side ticks
                ax.set_ylabel("Flexibility\n[TWh]", fontsize=font, labelpad=10)
                ax.yaxis.set_label_position("right")  # move label to right side
                ax.yaxis.tick_right()  # move ticks to right
                ax.tick_params(axis="y", labelsize=font)

            else:
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)

            # y-axis ticks
            if j == 0 or j == n_cols - 1:
                ax.tick_params(axis="y", labelsize=font, colors="black")
            else:
                ax.tick_params(
                    axis="y",
                    which="both",
                    labelleft=False,  # remove tick labels
                    left=True,  # keep tick marks
                    colors="white",  # make tick marks white
                )
            # Only first column gets scenario title
            if j == 0:
                scenario_title = f"{next(title_letters)}. {format_scenario_name_inline(scenario)}"
                ax.set_title(scenario_title, fontsize=font, x=0, ha="left")
                ax.set_ylabel("Shift up (+) / down (-)\n[GW]", fontsize=font, labelpad=10)

            ax.grid(True, axis="y", linestyle="--", alpha=0.6)
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)
            ax.set_axisbelow(True)

            if j < n_cols - 1:
                ax2.plot(
                    day_df["timestep"].dt.hour.values,
                    day_df["PV"].values,
                    color="#151514",
                    linewidth=2,
                    linestyle=":",
                    zorder=10,
                    label="PV generation",
                )

            # Set ax2 limits for all subplots (use precomputed min/max)
            ax2.set_ylim(min_pv * 0.95, max_pv * 1.05)

            # Hide all spines for ax2 everywhere
            for spine in ax2.spines.values():
                spine.set_visible(False)

            # Hide all ticks/labels by default
            ax2.tick_params(
                axis="y", which="both", labelleft=False, labelright=False, left=False, right=False
            )

            # Only for last column, show y-axis label and ticks
            if j == n_cols - 2:
                ax2.set_ylabel("PV generation\n[GW]", fontsize=font)
                ax2.tick_params(axis="y", labelright=True, right=True)

            if j == n_cols - 1:
                # Get current position of the axes
                pos = ax.get_position()  # Bbox(x0, y0, x1, y1)

                # Shift it to the right by e.g. 0.05 in figure coordinates
                new_pos = [pos.x0 + 0.07, pos.y0, pos.width, pos.height]
                ax.set_position(new_pos)

                # If you have ax2, move it exactly the same
                ax2.set_position(new_pos)

    # Harmonise y-limits for seasonal columns
    global_ymin = min(y_mins) * 1.2
    global_ymax = max(y_maxs) * 1.2
    for i in range(n_rows):
        for j in range(n_cols):
            if j < n_cols - 1:  # seasonal
                axes[i, j].set_ylim(global_ymin, global_ymax)
            else:  # last column: yearly stacked
                axes[i, j].set_ylim(0, max(y_maxs_year) * 1.1)

    # -------------------------------------------------
    # Legends
    # -------------------------------------------------
    shift_up_cols = [c for c in plot_cols if "Up" in c or c == "Electrolysis"]
    shift_down_cols = [c for c in plot_cols if "Down" in c]

    handles_up = [
        Patch(facecolor=cnf.SHIFT_COLOR_MAPPING.get(c, "#cccccc"), label=c) for c in shift_up_cols
    ]
    handles_down = [
        Patch(facecolor=cnf.SHIFT_COLOR_MAPPING.get(c, "#cccccc"), label=c) for c in shift_down_cols
    ]

    handles_pv = [
        Line2D([0], [0], color="#191919", linestyle=":", linewidth=2.5, label="PV generation")
    ]

    up = fig.legend(
        handles_up,
        [h.get_label() for h in handles_up],
        title="Shift Up (+):",
        loc="upper left",
        fontsize=font,
        title_fontsize=font,
        frameon=False,
        bbox_to_anchor=(1.07, 0.65),
    )
    down = fig.legend(
        handles_down,
        [h.get_label() for h in handles_down],
        title="Shift Down (-):",
        loc="upper left",
        fontsize=font,
        title_fontsize=font,
        frameon=False,
        bbox_to_anchor=(1.07, 0.5),
    )

    pv = fig.legend(
        handles_pv,
        [h.get_label() for h in handles_pv],
        title="",
        loc="upper left",
        fontsize=font,
        title_fontsize=font,
        frameon=False,
        bbox_to_anchor=(1.07, 0.35),
    )

    up.get_title().set_ha("left")
    down.get_title().set_ha("left")
    pv.get_title().set_ha("left")
    up._legend_box.align = "left"
    down._legend_box.align = "left"
    pv._legend_box.align = "left"

    plt.tight_layout(rect=[0, 0, 0.92, 1], h_pad=1.7, w_pad=0.0)

    fig_path = cnf.FIGURES_DIR / f"{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def season_days(scenario_list, filename):
    """Plot one representative day per season for each scenario with Up/Down + Electrolysis columns."""
    # -------------------------------------------------
    # Read data
    # -------------------------------------------------
    supply = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "generation_h.csv").fillna(0)

    demand = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "demand_h.csv").fillna(0)

    demand = demand.drop(columns=["Electrolysis"], errors="ignore")

    # Merge hourly data
    hourly_df = supply.merge(
        demand,
        on=["timestep", "scenario"],
        suffixes=("_supply", "_demand"),
    )
    hourly_df["timestep"] = pd.to_datetime(hourly_df["timestep"])

    # -------------------------------------------------
    # Filter & order scenarios
    # -------------------------------------------------
    if scenario_list is not None:
        hourly_df = hourly_df[hourly_df["scenario"].isin(scenario_list)]
        hourly_df["scenario"] = pd.Categorical(
            hourly_df["scenario"],
            categories=scenario_list,
            ordered=True,
        )

    # -------------------------------------------------
    # Add shifted demand
    # -------------------------------------------------
    def safe_col(df, col):
        return df[col] if col in df.columns else 0.0

    hourly_df["Shifted e-Mobility"] = (
        safe_col(hourly_df, "e-Mobility")
        - safe_col(hourly_df, "EMob Shift (Up)")
        - safe_col(hourly_df, "EMob Shift (Down)")
    )
    hourly_df["Shifted Heat Pump"] = (
        safe_col(hourly_df, "Heat Pump")
        - safe_col(hourly_df, "HP Shift (Up)")
        - safe_col(hourly_df, "HP Shift (Down)")
    )
    hourly_df["Shifted Conventional"] = (
        safe_col(hourly_df, "Conventional")
        - safe_col(hourly_df, "DSM (Up)")
        - safe_col(hourly_df, "DSM (Down)")
    )

    hourly_df["PV"] = safe_col(hourly_df, "PV Alpine") + safe_col(hourly_df, "PV Roof")

    max_pv = hourly_df["PV"].max()
    min_pv = 0

    # Columns for plotting: all Up/Down + Electrolysis
    plot_cols = [c for c in hourly_df.columns if ("Up" in c or "Down" in c)] # or c == "Electrolysis"

    font = 12
    title_letters = iter(string.ascii_lowercase)

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    n_rows = len(scenario_list)
    n_cols = len(SEASON_DAY_DICT)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols, 2.3 * n_rows),
        sharex=False,  # show x-ticks for all subplots
        sharey="row",
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    y_mins, y_maxs = [], []

    for i, scenario in enumerate(scenario_list):
        df_scen = hourly_df[hourly_df["scenario"] == scenario]

        for j, (season, day_str) in enumerate(SEASON_DAY_DICT.items()):
            ax = axes[i, j]
            ax2 = ax.twinx()

            day = pd.to_datetime(day_str)
            day_df = df_scen[
                (df_scen["timestep"] >= day) & (df_scen["timestep"] < day + pd.Timedelta(days=1))
            ]

            if day_df.empty:
                continue

            x = day_df["timestep"].dt.hour.values
            y = day_df[plot_cols].values.T

            # Separate positive and negative for stacking
            y_pos = np.clip(y, 0, None)
            y_neg = np.clip(y, None, 0)

            # Colors from TECH_COLOR_MAPPING
            colors = [cnf.SHIFT_COLOR_MAPPING.get(col, "#cccccc") for col in plot_cols]

            if y_pos.any():
                ax.stackplot(x, -y_pos, colors=colors)
            if y_neg.any():
                ax.stackplot(x, -y_neg, colors=colors)

            # Titles & labels
            # if i == 0:
            #     ax.set_title(season, fontsize=font)

            # x-ticks for all subplots
            ax.set_xticks(range(6, 24, 6))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(6, 24, 6)], fontsize=font)
            ax.tick_params(axis="y", labelsize=font)
            ax.tick_params(axis="x", labelsize=font - 1)

            if j == 0:
                # Only first column gets scenario name as "title"
                ax.set_ylabel("Shift up (+) / down (-)\n[GW]", fontsize=font)
                letter = next(title_letters)
                scenario_title = f"{letter}. {format_scenario_name_inline(scenario)}"
                ax.set_title(scenario_title, fontsize=font, x=0, ha="left")
            else:
                # Remove y-label for middle/right columns
                ax.set_ylabel("")

            # Annotate season inside each subplot (top-left corner)
            ax.text(
                0.5,
                0.9,  # relative coordinates: 0.5 = center x, 0.9 = near top
                season,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=font,
                # fontweight="bold"
            )

            ax.grid(True, axis="y", linestyle="--", alpha=0.6)
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)
            ax.set_axisbelow(True)

            # spine adjustments
            if j == 0:  # first column
                ax.spines["left"].set_visible(True)
                ax.spines["right"].set_visible(False)
            elif j == n_cols - 1:  # last column
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(True)
            else:  # middle columns
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)

            # y-axis ticks
            if j != 0:
                ax.tick_params(axis="y", colors="white")  # tick lines

            # Collect y-limits
            ymin, ymax = ax.get_ylim()
            y_mins.append(ymin)
            y_maxs.append(ymax)

            # Set x-limits exactly 0–23
            ax.set_xlim(0, 23)

            if j < n_cols - 1:
                ax2.plot(
                    day_df["timestep"].dt.hour.values,
                    day_df["PV"].values,
                    color="#0D0D0C",
                    linewidth=2,
                    linestyle=":",
                    zorder=10,
                    label="PV generation",
                )

            # Set ax2 limits for all subplots (use precomputed min/max)
            ax2.set_ylim(min_pv * 0.95, max_pv * 1.05)

            # Hide all spines for ax2 everywhere
            for spine in ax2.spines.values():
                spine.set_visible(False)

            # Hide all ticks/labels by default
            ax2.tick_params(
                axis="y", which="both", labelleft=False, labelright=False, left=False, right=False
            )

            # Only for last column, show y-axis label and ticks
            if j == n_cols - 1:
                ax2.set_ylabel("PV generation\n[GW]", fontsize=font)
                ax2.tick_params(axis="y", labelright=True, right=True)

    # -------------------------------------------------
    # Harmonise y-limits
    # -------------------------------------------------
    global_ymin = min(y_mins) * 1.2
    global_ymax = max(y_maxs) * 1.2

    for ax in axes.flat:
        ax.set_ylim(global_ymin, global_ymax)

    # -------------------------------------------------
    # Legend (global at right)
    # -------------------------------------------------
    shift_up_cols = [c for c in plot_cols if "Up" in c or c == "Electrolysis"]
    shift_down_cols = [c for c in plot_cols if "Down" in c]

    # Create handles
    handles_up = [
        Patch(facecolor=cnf.SHIFT_COLOR_MAPPING.get(c, "#cccccc"), label=c) for c in shift_up_cols
    ]
    handles_down = [
        Patch(facecolor=cnf.SHIFT_COLOR_MAPPING.get(c, "#cccccc"), label=c) for c in shift_down_cols
    ]
    handles_pv = [
        Line2D([0], [0], color="#141413", linestyle=":", linewidth=2.5, label="PV generation")
    ]

    up = fig.legend(
        handles_up,
        [h.get_label() for h in handles_up],
        title="Shift Up (+):",
        loc="upper left",
        fontsize=font,
        title_fontsize=font,
        frameon=False,
        bbox_to_anchor=(0.93, 0.65),  # slightly right of axes
    )

    down = fig.legend(
        handles_down,
        [h.get_label() for h in handles_down],
        title="Shift Down (-):",
        loc="upper left",
        fontsize=font,
        title_fontsize=font,
        frameon=False,
        bbox_to_anchor=(0.93, 0.5),
    )

    pv = fig.legend(
        handles_pv,
        [h.get_label() for h in handles_pv],
        title="",
        loc="upper left",
        fontsize=font,
        title_fontsize=font,
        frameon=False,
        bbox_to_anchor=(0.93, 0.35),
    )

    # Force title alignment
    up.get_title().set_ha("left")
    down.get_title().set_ha("left")
    pv.get_title().set_ha("left")

    # Force entry alignment
    up._legend_box.align = "left"
    down._legend_box.align = "left"
    pv._legend_box.align = "left"

    plt.tight_layout(rect=[0, 0, 0.92, 1], h_pad=1.4, w_pad=0.5)
    fig_path = cnf.FIGURES_DIR / f"{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def dispatch(scenario_list, filename):
    """Plot dispatch per season"""
    # -------------------------------------------------
    # Read data
    # -------------------------------------------------

    # Prepare Supply
    supply = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "generation_h.csv").fillna(0)
    supply = supply.drop(columns=["Electrolysis"], errors="ignore")

    demand = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "demand_h.csv").fillna(0)

    # Merge hourly data
    hourly_df = supply.merge(
        demand,
        on=["timestep", "scenario"],
        suffixes=("_supply", "_demand"),
    )
    hourly_df["timestep"] = pd.to_datetime(hourly_df["timestep"])

    # -------------------------------------------------
    # Filter & order scenarios
    # -------------------------------------------------
    if scenario_list is not None:
        hourly_df = hourly_df[hourly_df["scenario"].isin(scenario_list)]
        hourly_df["scenario"] = pd.Categorical(
            hourly_df["scenario"],
            categories=scenario_list,
            ordered=True,
        )

    ordered_techs = [t for t in list(cnf.TECH_COLOR_MAPPING.keys()) if t in supply.columns]

    # -------------------------------------------------
    # Add shifted demand
    # -------------------------------------------------
    def safe_col(df, col):
        return df[col] if col in df.columns else 0.0

    hourly_df["Shifted e-Mobility"] = (
        safe_col(hourly_df, "e-Mobility")
        - safe_col(hourly_df, "EMob Shift (Up)")
        - safe_col(hourly_df, "EMob Shift (Down)")
    )
    hourly_df["Shifted Heat Pump"] = (
        safe_col(hourly_df, "Heat Pump")
        - safe_col(hourly_df, "HP Shift (Up)")
        - safe_col(hourly_df, "HP Shift (Down)")
    )
    hourly_df["Shifted Conventional"] = (
        safe_col(hourly_df, "Conventional")
        - safe_col(hourly_df, "DSM (Up)")
        - safe_col(hourly_df, "DSM (Down)")
    )

    hourly_df["e-Mobility-Down"] = safe_col(hourly_df, "e-Mobility") - safe_col(
        hourly_df, "EMob Shift (Down)"
    )
    hourly_df["Heat Pump-Down"] = safe_col(hourly_df, "Heat Pump") - safe_col(
        hourly_df, "HP Shift (Down)"
    )
    hourly_df["Conventional-Down"] = safe_col(hourly_df, "Conventional") - safe_col(
        hourly_df, "DSM (Down)"
    )

    hourly_df["Variable load"] = (
        safe_col(hourly_df, "Electrolysis")
        + safe_col(hourly_df, "Pump (Load)")
        + safe_col(hourly_df, "Battery (Load)")
    )

    hourly_df["Shifted total load"] = (
        safe_col(hourly_df, "Shifted Conventional")
        - safe_col(hourly_df, "Variable load")
        + safe_col(hourly_df, "Shifted Heat Pump")
        + safe_col(hourly_df, "Shifted e-Mobility")
    )

    hourly_df["Unshifted total load"] = (
        safe_col(hourly_df, "Conventional")
        - safe_col(hourly_df, "Variable load")
        + safe_col(hourly_df, "Heat Pump")
        + safe_col(hourly_df, "e-Mobility")
    )

    filtered_columns = [
        "Conventional-Down",
        "DSM (Up)",
        "Variable load",
        "e-Mobility-Down",
        "EMob Shift (Up)",
        "Heat Pump-Down",
        "HP Shift (Up)",
        "DSM (Down)",
        "EMob Shift (Down)",
        "HP Shift (Down)",
    ]

    # Columns for plotting: all Up/Down + Electrolysis
    plot_cols = [c for c in filtered_columns if c in hourly_df.columns]
    hourly_df[plot_cols] = hourly_df[plot_cols].abs()
    hourly_df = hourly_df.loc[
        :,
        list(["scenario", "timestep", "Shifted total load", "Unshifted total load"]) + (plot_cols),
    ]

    font = 12
    title_letters = iter(string.ascii_lowercase)

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    n_rows = len(scenario_list)
    n_cols = len(SEASON_DAY_DICT)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols, 2.3 * n_rows),
        sharex=False,  # show x-ticks for all subplots
        sharey="row",
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    y_mins, y_maxs = [], []

    for i, scenario in enumerate(scenario_list):
        df_scen = hourly_df[hourly_df["scenario"] == scenario]

        for j, (season, day_str) in enumerate(SEASON_DAY_DICT.items()):
            ax = axes[i, j]
            day = pd.to_datetime(day_str)
            day_df = df_scen[
                (df_scen["timestep"] >= day) & (df_scen["timestep"] < day + pd.Timedelta(days=1))
            ]

            if day_df.empty:
                continue

            x = day_df["timestep"].dt.hour.values
            y = day_df[plot_cols].values.T

            # Separate positive and negative for stacking
            y_pos = np.clip(y, 0, None)
            y_neg = np.clip(y, None, 0)

            # Colors from TECH_COLOR_MAPPING
            colors = [cnf.SHIFT_COLOR_MAPPING.get(col, "#cccccc") for col in plot_cols]
            hatches = [cnf.SHIFT_HATCH_MAPPING.get(col, None) for col in plot_cols]

            if y_pos.any():
                polys_pos = ax.stackplot(x, y_pos, colors=colors)
            if y_neg.any():
                polys_neg = ax.stackplot(x, y_neg, colors=colors)

            if y_pos.any():
                for poly, col in zip(polys_pos, plot_cols):
                    hatch = cnf.SHIFT_HATCH_MAPPING.get(col)
                    if hatch:
                        poly.set_hatch(hatch)
                        poly.set_edgecolor("white")
                        poly.set_linewidth(0.3)

            if y_neg.any():
                for poly, col in zip(polys_neg, plot_cols):
                    hatch = cnf.SHIFT_HATCH_MAPPING.get(col)
                    if hatch:
                        poly.set_hatch(hatch)
                        poly.set_edgecolor("white")
                        poly.set_linewidth(0.3)

            ax.plot(
                day_df["timestep"].dt.hour.values,
                day_df["Shifted total load"].values,
                color="black",
                linewidth=1.2,
                linestyle="-",
                zorder=10,
                label="Shifted total load",
            )

            ax.plot(
                day_df["timestep"].dt.hour.values,
                day_df["Unshifted total load"].values,
                color="black",
                linewidth=1.2,
                linestyle=":",
                zorder=10,
                label="Unshifted total load",
            )
            # Titles & labels
            # if i == 0:
            #     ax.set_title(season, fontsize=font)

            # x-ticks for all subplots
            ax.set_xticks(range(6, 24, 6))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(6, 24, 6)], fontsize=font)
            ax.tick_params(axis="y", labelsize=font)
            ax.tick_params(axis="x", labelsize=font - 1)

            if j == 0:
                # Only first column gets scenario name as "title"
                ax.set_ylabel("Electricity demand\n[GW]", fontsize=font)
                letter = next(title_letters)
                scenario_title = f"{letter}. {format_scenario_name_inline(scenario)}"
                ax.set_title(scenario_title, fontsize=font, x=0, ha="left")
            else:
                # Remove y-label for middle/right columns
                ax.set_ylabel("")

            # Annotate season inside each subplot (top-left corner)
            ax.text(
                0.5,
                0.9,  # relative coordinates: 0.5 = center x, 0.9 = near top
                season,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=font,
                # fontweight="bold"
            )

            ax.grid(True, axis="y", linestyle="--", alpha=0.6)
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)
            ax.set_axisbelow(True)

            # spine adjustments
            if j == 0:  # first column
                ax.spines["left"].set_visible(True)
                ax.spines["right"].set_visible(False)
            elif j == n_cols - 1:  # last column
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(True)
            else:  # middle columns
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)

            # y-axis ticks
            if j != 0:
                ax.tick_params(axis="y", colors="white")  # tick lines

            # Collect y-limits
            ymin, ymax = ax.get_ylim()
            y_mins.append(ymin)
            y_maxs.append(ymax)

            # Set x-limits exactly 0–23
            ax.set_xlim(0, 23)

    # -------------------------------------------------
    # Harmonise y-limits
    # -------------------------------------------------
    global_ymin = min(y_mins) * 1.2
    global_ymax = max(y_maxs) * 1.2

    for ax in axes.flat:
        ax.set_ylim(global_ymin, global_ymax)

    # -------------------------------------------------
    # Legend (global at right)
    # -------------------------------------------------
    line_handles = [
        Line2D([0], [0], color="black", lw=1.2, linestyle="-", label="Shifted total load"),
        Line2D([0], [0], color="black", lw=1.2, linestyle=":", label="Unshifted total load"),
    ]

    legend_items = [c.replace("-Down", "") for c in plot_cols]

    handles_general = [
        Patch(
            facecolor=cnf.SHIFT_COLOR_MAPPING.get(c, "#cccccc"),
            hatch=cnf.SHIFT_HATCH_MAPPING.get(c, None),
            edgecolor="white",  # important so hatch is visible
            linewidth=0.5,
            label=c,
        )
        for c in legend_items
        if "(" not in c
    ] + line_handles

    handles_shift = [
        Patch(
            facecolor=cnf.SHIFT_COLOR_MAPPING.get(c, "#cccccc"),
            hatch=cnf.SHIFT_HATCH_MAPPING.get(c, None),
            edgecolor="white",  # important so hatch is visible
            linewidth=0.5,
            label=c,
        )
        for c in legend_items
        if "(" in c
    ] + line_handles

    legend_general = fig.legend(
        handles_general,
        [h.get_label() for h in handles_general],
        title="Load:",
        loc="upper left",
        fontsize=font,
        title_fontsize=font,
        frameon=False,
        bbox_to_anchor=(0.93, 0.8),  # slightly right of axes
    )

    legend_shift = fig.legend(
        handles_shift,
        [h.get_label() for h in handles_shift],
        title="Shift Up and Down:",
        loc="lower left",
        fontsize=font,
        title_fontsize=font,
        frameon=False,
        bbox_to_anchor=(0.93, 0.25),  # slightly right of axes
    )

    # Force title alignment
    legend_general.get_title().set_ha("left")
    legend_shift.get_title().set_ha("left")

    # Force entry alignment
    legend_general._legend_box.align = "left"
    legend_shift._legend_box.align = "left"

    plt.tight_layout(rect=[0, 0, 0.92, 1], h_pad=1.4, w_pad=0.5)
    fig_path = cnf.FIGURES_DIR / f"{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def end_use_carrier_mix(scenario_list, filename, min_share=0.005):
    """Horizontal normalized stacked bar chart per carrier (one column).

    Filtering out tech contributions below min_share (default 0.5%),
    reversing y-axis so first scenario is on top,
    with subplot titles as letters, x-axis % shown, and simplified legends.
    Numbers are displayed in the middle of each stacked bar piece if >5%.
    Legend items are capitalized.
    """
    output_path = cnf.RESULTS_SECMOD_DIR
    supply = pd.read_csv(
        os.path.join(output_path, "combined_yearly_flows_generation_high_and_low.csv"),
        index_col=False,
    )

    # End-use electrification
    transport = supply[(supply["carrier"].str.contains("mobility"))]
    transport["units"] = "billion vkm"
    transport["values"] *= 1000 # convert 10^12 to 10^9 vkm

    heat = supply[(supply["carrier"].str.contains("heat"))]
    # merging low temperature heat carriers
    heat["carrier"] = heat["carrier"].str.replace("heat low temperature DAC", "heat low temperature", regex=False)

    end_use_df = pd.concat([transport, heat], ignore_index=True)

    end_use_df = end_use_df[end_use_df["scenario"].isin(scenario_list)]
    end_use_df = format_scenario_secmod(end_use_df)

    carriers = end_use_df["carrier"].unique()
    techs = end_use_df["techs"].unique()

    # Assign colors to techs
    cmap = plt.get_cmap("tab20")
    colors = {tech: cmap(i % 20) for i, tech in enumerate(techs)}

    n_carriers = len(carriers)
    fig, axes = plt.subplots(n_carriers, 1, figsize=(10, 2.5 * n_carriers), sharex=False)

    if n_carriers == 1:
        axes = [axes]

    letters = string.ascii_lowercase

    for i, (ax, carrier) in enumerate(zip(axes, carriers)):
        df_carrier = end_use_df[end_use_df["carrier"] == carrier]
        unit = df_carrier["units"].iloc[0]

        # Pivot: techs as columns, scenarios as index
        pivot_df = df_carrier.pivot_table(
            index="scenario", columns="techs", values="values", aggfunc="sum", fill_value=0
        )

        # Normalize rows to sum to 1 (percentage)
        pivot_norm = pivot_df.div(pivot_df.sum(axis=1), axis=0)

        # Filter out techs below min_share in all scenarios
        pivot_norm = pivot_norm.loc[:, (pivot_norm >= min_share).any(axis=0)]

        # Reverse the order of scenarios
        # pivot_norm = pivot_norm.iloc[::-1]

        # Horizontal stacked bars
        left = pd.Series([0] * pivot_norm.shape[0], index=pivot_norm.index)
        for tech in pivot_norm.columns:
            values = pivot_norm[tech]
            abs_values = pivot_df[tech]
            ax.barh(pivot_norm.index, values, left=left, color=colors[tech], label=tech)

            # Annotate numbers in the middle if >5%
            for j, scenario in enumerate(pivot_norm.index):
                share = values.loc[scenario]
                abs_value = abs_values.loc[scenario]
                if share > 0.05:  # Only display if >5%
                    x_pos = left.loc[scenario] + share / 2
                    y_pos = j
                    ax.text(
                        x_pos,
                        y_pos,
                        # f"{int(share * 100)}",
                        f"{round(abs_value,1)}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=10,
                        fontweight="bold",
                    )

            left += values

        # X-axis
        ax.set_xlim(0, 1)
        if i == n_carriers - 1:
            ax.set_xlabel("End use carrier mix (%)", fontsize=12)
        else:
            ax.set_xlabel("")

        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

        # Remove y-axis label
        ax.set_ylabel("")

        # Subplot title: capitalized + letter prefix
        ax.set_title(f"{letters[i]}. {carrier.capitalize()} [{unit}]", loc="left", fontsize=12)

        # Legend to the right, capitalized labels, no title, no frame
        handles, labels = ax.get_legend_handles_labels()
        labels = [map_tech_string(label) for label in labels]
        ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)

    plt.tight_layout()
    fig_path = cnf.FIGURES_DIR / f"{filename}_abs.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


# %%
import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D


def co2_balance(scenario_list, filename, min_share=0.005):

    output_path = cnf.RESULTS_SECMOD_DIR
    output = pd.read_csv(
        os.path.join(output_path, "combined_output_high_and_low.csv"),
        index_col=False,
    )

    df = output[
        (output["variable"] == "co2_balance") & (output["scenario"].isin(scenario_list))
    ].copy()

    # Format scenario names
    df = format_scenario_secmod(df)
    df = map_tech_names_secmod(df, "techs")

    # FIX: convert literal "\n" to real line breaks (LaTeX-safe)
    df["scenario"] = df["scenario"].str.replace(r"\s*\\n\s*", "\n", regex=True)

    # Convert to Mton
    df["values"] = df["values"] / 1000.0

    scenarios = df["scenario"].unique()
    techs = df["techs"].unique()

    max_pos = df[df["values"] > 0].groupby("scenario")["values"].sum().max()

    min_neg = df[df["values"] < 0].groupby("scenario")["values"].sum().min()
    scale = np.max([abs(min_neg), max_pos]) * 1.15

    # Color mapping
    cmaps = ["tab20", "tab20b", "tab20c"]

    palette = []
    for name in cmaps:
        cmap = plt.get_cmap(name)
        palette.extend([cmap(i) for i in range(cmap.N)])

    colors = {tech: palette[i] for i, tech in enumerate(techs)}

    fig, ax = plt.subplots(figsize=(8, 1 * len(scenarios)))

    # Reverse scenario order
    scenarios = scenarios[::-1]
    y_pos = np.arange(len(scenarios))

    pos_left = np.zeros(len(scenarios))
    neg_left = np.zeros(len(scenarios))

    # --- STACKED BARS ---
    for tech in techs:
        tech_vals = []
        for s in scenarios:
            val = df.loc[
                (df["scenario"] == s) & (df["techs"] == tech),
                "values",
            ].sum()
            tech_vals.append(val)

        tech_vals = np.array(tech_vals)

        # Skip tiny contributors
        if np.all(np.abs(tech_vals) < min_share * np.abs(df["values"]).sum()):
            continue

        # Positive emissions
        pos = np.where(tech_vals > 0, tech_vals, 0)
        ax.barh(
            y_pos,
            pos,
            left=pos_left,
            color=colors[tech],
            label=tech,
        )
        pos_left += pos

        # Negative emissions
        neg = np.where(tech_vals < 0, tech_vals, 0)
        ax.barh(
            y_pos,
            neg,
            left=neg_left,
            color=colors[tech],
        )
        neg_left += neg

    # --- NET CO₂ BALANCE (diamond marker) ---
    net_balance = pos_left + neg_left
    ax.scatter(
        net_balance,
        y_pos,
        marker="D",
        color="white",
        edgecolor="black",
        linewidths=1.6,
        zorder=5,
        s=60,
    )

    # --- AXES & GRID ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenarios)
    ax.set_xlabel("CO₂ balance [Mton]")
    ax.set_xlim(-scale, scale)
    ax.axvline(0, linewidth=0.8, color="black")

    ax.grid(True, axis="x", linestyle="--", color="gray", alpha=0.4)
    ax.grid(True, axis="y", linestyle="--", color="gray", alpha=0.3)
    ax.set_axisbelow(True)

    # --- LEGENDS ---
    handles, labels = ax.get_legend_handles_labels()

    pos_handles = []
    neg_handles = []

    for h, l in zip(handles, labels):
        vals = df.loc[df["techs"] == l, "values"]
        if vals.mean() >= 0:
            pos_handles.append((h, l))
        else:
            neg_handles.append((h, l))

    # Positive emitters legend
    leg1 = fig.legend(
        [h for h, _ in pos_handles],
        [l for _, l in pos_handles],
        bbox_to_anchor=(0.93, 0.98),
        loc="upper left",
        frameon=False,
        title="Positive emitters (+):",
    )

    # Negative emitters legend
    leg2 = fig.legend(
        [h for h, _ in neg_handles],
        [l for _, l in neg_handles],
        bbox_to_anchor=(0.93, 0.65),
        loc="upper left",
        frameon=False,
        title="Negative emitters (-):",
    )

    # Net balance legend (diamond)
    net_handle = Line2D(
        [0],
        [0],
        marker="D",
        color="white",
        markeredgecolor="black",
        markeredgewidth=1.6,
        linestyle="-",
        markersize=8,
        label="Net CO₂ balance",
    )

    leg3 = fig.legend(
        [net_handle],
        ["Net CO₂ balance"],
        bbox_to_anchor=(0.93, 0.22),
        loc="upper left",
        frameon=False,
    )

    leg1.get_title().set_ha("left")
    leg2.get_title().set_ha("left")
    leg3.get_title().set_ha("left")
    leg1._legend_box.align = "left"
    leg2._legend_box.align = "left"
    leg3._legend_box.align = "left"

    # --- SAVE ---
    plt.tight_layout(rect=[0, 0, 0.92, 1], h_pad=1.4, w_pad=0.0)
    fig_path = cnf.FIGURES_DIR / f"{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def compute_flexibility_metrics():
    """Compute the total flexibility."""
    supply = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "generation_h.csv").fillna(0)

    out = (
        supply.drop(columns=["timestep"])
        .groupby("scenario", as_index=False)
        .sum()
        .melt(
            id_vars="scenario",
            var_name="tech",
            value_name="value",
        )
        .assign(
            value=lambda df: df["value"] / 1000,
            unit="TWh",
        )
    )

    flex_option = [f for f in out.tech.unique() if "(Down)" in f]
    flex = out.query("tech in @flex_option")
    merge = flex.merge(
        flex.groupby("scenario", as_index=False).sum(), on="scenario", suffixes=("", "_total")
    )
    flex_tot = (
        flex.groupby("scenario", as_index=False).sum().rename(columns={"value": "flex_total"})
    )

    return flex_tot, merge, flex_option


def generation_capacity_curt_per_unit_flex(filename, ref_scen, scenario_list=None):
    """Plot metrics specific to the flexibility activated."""
    balance_df = postprocess.electricity_yearly_balance()
    balance_df["scenario"] = order_scenarios(balance_df["scenario"])
    balance_df = balance_df.sort_values("scenario")

    flex_tot, merge, flex_option = compute_flexibility_metrics()
    # ------------------------------------------------------------------
    # Remove other unwanted techs
    # ------------------------------------------------------------------
    tech_to_remove = [
        "Load (Net)",
        "Load (Total)",
        "Load Shed",
        "DSM (Down)",
        "HP Shift (Down)",
        "EMob Shift (Down)",
        "DSM (Up)",
        "HP Shift (Up)",
        "EMob Shift (Up)",
        "Pump (Load)",
        "Battery (Load)",
        "Import",
        "Import (Net)",
        "Export",
        "e-Mobility",
        "Conventional",
        "Heat Pump",
        "Electrolysis",
    ]
    balance_df = balance_df[~balance_df["tech"].isin(tech_to_remove)]
    balance_ref = balance_df[balance_df["scenario"] == ref_scen]
    balance_df = balance_df.merge(
        balance_ref[["tech", "value"]].rename(columns={"value": "ref_value"}), on="tech", how="left"
    )

    balance_df = balance_df.merge(
        flex_tot[["scenario", "flex_total"]],
        on="scenario",
        how="left",
    )

    balance_df["delta_value"] = (
        (balance_df["value"] - balance_df["ref_value"]) / balance_df["flex_total"] * 1000
    )
    balance_df["unit"] = "GWh/TWh"

    if scenario_list is not None:
        balance_df = balance_df[balance_df["scenario"].isin(scenario_list)]

    cap_path = cnf.RESULTS_NEXUS_DIR
    cap_file = cap_path / "cap.csv"

    cap_df = pd.read_csv(cap_file).fillna(0)
    cap_df = cap_df[~cap_df["tech"].isin(tech_to_remove)]
    cap_df["capacity"] = cap_df["capacity"].abs()

    cap_ref = cap_df[cap_df["scenario"] == ref_scen]
    cap_df = cap_df.merge(
        cap_ref[["tech", "capacity"]].rename(columns={"capacity": "ref_value"}),
        on="tech",
        how="left",
    )

    cap_df = cap_df.merge(
        flex_tot[["scenario", "flex_total"]],
        on="scenario",
        how="left",
    )

    cap_df["delta_value"] = (cap_df["capacity"] - cap_df["ref_value"]) / cap_df["flex_total"] * 1000
    cap_df["unit"] = "MW/TWh"

    if scenario_list is not None:
        cap_df = cap_df[cap_df["scenario"].isin(scenario_list)]

    curt_h = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "curt_h.csv").fillna(0)
    curt_df = postprocess.timeseries_to_yearly_total(curt_h)
    curt_df = curt_df[~curt_df["tech"].isin(tech_to_remove)]

    curt_ref = curt_df[curt_df["scenario"] == ref_scen]
    curt_df = curt_df.merge(
        curt_ref[["tech", "value"]].rename(columns={"value": "ref_value"}), on="tech", how="left"
    )

    curt_df = curt_df.merge(
        flex_tot[["scenario", "flex_total"]],
        on=["scenario"],
        how="left",
    )

    curt_df["delta_value"] = (
        (curt_df["value"] - curt_df["ref_value"]) / curt_df["flex_total"] * 1000
    )
    curt_df["unit"] = "GWh/TWh"

    if scenario_list is not None:
        curt_df = curt_df[curt_df["scenario"].isin(scenario_list)]

    # remove ref
    balance_df = balance_df[balance_df["scenario"] != ref_scen]
    cap_df = cap_df[cap_df["scenario"] != ref_scen]
    curt_df = curt_df[curt_df["scenario"] != ref_scen]

    plot_generation_and_capacity_delta(balance_df, cap_df, curt_df, filename)


def system_cost_horizontal_unit_flex(scenario_list, ref_scen, filename):
    """Horizontal stacked system cost plots with net totals and legends including diamond markers."""
    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    flex_tot, merge, flex_option = compute_flexibility_metrics()

    cost_cost_path = cnf.RESULTS_NEXUS_DIR / "system_cost_cost.csv"
    if not cost_cost_path.exists():
        raise FileNotFoundError("system_cost_cost.csv not found")

    cost_cost_df = pd.read_csv(cost_cost_path).fillna(0)
    cost_cost_df["cost_type"] = cost_cost_df["cost_type"].str.replace(" 2050", "", regex=False)

    cost_gen_path = cnf.RESULTS_NEXUS_DIR / "system_cost_gen.csv"
    if not cost_gen_path.exists():
        raise FileNotFoundError("system_cost_gen.csv not found")

    cost_gen_df = pd.read_csv(cost_gen_path).fillna(0)

    # -------------------------------------------------
    # Filter scenarios
    # -------------------------------------------------
    if scenario_list is not None:
        cost_cost_df = cost_cost_df[cost_cost_df["scenario"].isin(scenario_list)]
        cost_gen_df = cost_gen_df[cost_gen_df["scenario"].isin(scenario_list)]

    # -------------------------------------------------
    # Order scenarios
    # -------------------------------------------------
    if scenario_list is not None:
        cost_cost_df["scenario"] = pd.Categorical(
            cost_cost_df["scenario"],
            categories=scenario_list,
            ordered=True,
        )
        cost_gen_df["scenario"] = pd.Categorical(
            cost_gen_df["scenario"],
            categories=scenario_list,
            ordered=True,
        )

    # -------------------------------------------------
    # Prepare pivots
    # -------------------------------------------------
    cost_type_pivot = cost_cost_df.pivot(
        index="scenario", columns="cost_type", values="cost"
    ).fillna(0)

    tech_pivot = cost_gen_df.pivot(index="scenario", columns="tech", values="cost").fillna(0)

    ref_cost_type = cost_type_pivot.loc[ref_scen]
    ref_tech = tech_pivot.loc[ref_scen]

    cost_type_diff = cost_type_pivot.subtract(ref_cost_type, axis=1)
    tech_diff = tech_pivot.subtract(ref_tech, axis=1)
    scenario_order = cost_type_pivot.index

    # set scenario as index for flex_tot
    flex_tot_idx = flex_tot.set_index("scenario").loc[scenario_order, "flex_total"]

    # divide each column by matching scenario
    cost_type_diff_flex = cost_type_diff.div(flex_tot_idx, axis=0)
    tech_diff_flex = tech_diff.div(flex_tot_idx, axis=0)

    # -------------------------------------------------
    # Colors
    # -------------------------------------------------
    colors_cost_type = {ct: f"C{i}" for i, ct in enumerate(cost_type_pivot.columns)}

    custom_colors = {
        "Fixed Operation": "#51b6ff",
        "Investments": "#feb27b",
        "Trading": "#8fc778",
        "Variable Operation": "#e56868",
        "Grid Expansion": "#c4a7fd",
    }

    colors_cost_type = {
        ct: custom_colors.get(ct, f"C{i}") for i, ct in enumerate(cost_type_pivot.columns)
    }

    ordered_techs = [t for t in cnf.TECH_COLOR_MAPPING.keys() if t in tech_pivot.columns]
    tech_colors = {t: cnf.TECH_COLOR_MAPPING.get(t, "#cccccc") for t in ordered_techs}

    # -------------------------------------------------
    # Helper function
    # -------------------------------------------------
    def stacked_diverging_barh(ax, df, color_map, ref_totals=None):
        y = np.arange(len(df))
        labels = df.index.to_numpy()

        left_pos = np.zeros(len(df))
        left_neg = np.zeros(len(df))

        for col in df.columns:
            values = df[col].values
            pos = np.clip(values, 0, None)
            neg = np.clip(values, None, 0)

            ax.barh(y, pos, left=left_pos, color=color_map[col])
            ax.barh(y, neg, left=left_neg, color=color_map[col])

            left_pos += pos
            left_neg += neg

        totals = df.sum(axis=1).values
        totals[np.abs(totals) < 0.005] = np.nan

        # Diamond markers for totals
        ax.plot(
            totals,
            y,
            marker="D",
            linestyle="None",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,  # optional, but looks good
            zorder=5,
        )

        # Annotate % reduction only if ref_totals is provided (second column)
        if ref_totals is not None:
            pct_change = -(totals)
            for xi, yi, p in zip(totals, y, pct_change):
                if abs(p) >= 0.002:
                    ax.text(
                        xi,
                        yi + 0.3,
                        f"{-p:.1f}",
                        va="center",
                        ha="center",
                        fontsize=11,
                        color="black",
                        fontweight="bold",
                    )

        ax.axvline(0, color="black", linewidth=0.8)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        ax.set_yticklabels([format_scenario_name(s) for s in labels])

        return y, totals

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 7),
        sharey="row",
        gridspec_kw={"width_ratios": [1, 0.7], "height_ratios": [1, 1]},
    )

    font = 14
    title_letters = iter(string.ascii_lowercase)

    # ---------- Row 0: cost by type ----------
    y, totals_cost_abs = stacked_diverging_barh(
        axes[0, 0], cost_type_pivot, colors_cost_type, ref_totals=None
    )
    letter = next(title_letters)
    axes[0, 0].set_title(f"{letter}. Total system cost by cost type", fontsize=font, x=0, ha="left")
    axes[0, 0].set_xlabel("Total system cost\n[Million €]")

    ref_totals_cost = cost_type_pivot.loc[ref_scen].sum()
    y, totals_cost_diff = stacked_diverging_barh(
        axes[0, 1], cost_type_diff_flex, colors_cost_type, ref_totals=ref_totals_cost
    )
    letter = next(title_letters)
    axes[0, 1].set_title(f"{letter}. Cost difference by cost type", fontsize=font, x=0, ha="left")
    axes[0, 1].set_xlabel("Cost difference\n[Million € TWh$^{-1}$]")
    axes[0, 1].tick_params(axis="y", labelleft=False)

    # ---------- Row 1: generation cost ----------
    y, totals_gen_abs = stacked_diverging_barh(
        axes[1, 0], tech_pivot[ordered_techs], tech_colors, ref_totals=None
    )
    letter = next(title_letters)
    axes[1, 0].set_title(
        f"{letter}. Total system cost by technology", fontsize=font, x=0, ha="left"
    )
    axes[1, 0].set_xlabel("Total system cost\n[Million €]")

    ref_totals_gen = tech_pivot.loc[ref_scen, ordered_techs].sum()
    y, totals_gen_diff = stacked_diverging_barh(
        axes[1, 1], tech_diff_flex[ordered_techs], tech_colors, ref_totals=ref_totals_gen
    )
    letter = next(title_letters)
    axes[1, 1].set_title(f"{letter}. Cost difference by technology", fontsize=font, x=0, ha="left")
    axes[1, 1].set_xlabel("Cost difference\n[Million € TWh$^{-1}$]")
    axes[1, 1].tick_params(axis="y", labelleft=False)

    # -------------------------------------------------
    # Fontsize
    # -------------------------------------------------
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            ax.tick_params(axis="y", labelsize=font)
            ax.tick_params(axis="x", labelsize=font)
            if ax.get_ylabel():
                ax.yaxis.label.set_size(font)
            if ax.get_xlabel():
                ax.xaxis.label.set_size(font)

    # -------------------------------------------------
    # Harmonise x limits per column
    # -------------------------------------------------
    factor = 1.1
    xmin_col0 = min(axes[0, 0].get_xlim()[0], axes[1, 0].get_xlim()[0]) * factor
    xmax_col0 = max(axes[0, 0].get_xlim()[1], axes[1, 0].get_xlim()[1]) * factor
    axes[0, 0].set_xlim(xmin_col0, xmax_col0)
    axes[1, 0].set_xlim(xmin_col0, xmax_col0)

    xmin_col1 = min(axes[0, 1].get_xlim()[0], axes[1, 1].get_xlim()[0]) * factor
    xmax_col1 = max(axes[0, 1].get_xlim()[1], axes[1, 1].get_xlim()[1]) * factor
    axes[0, 1].set_xlim(xmin_col1, xmax_col1)
    axes[1, 1].set_xlim(xmin_col1, xmax_col1)

    # -------------------------------------------------
    # Legends (including diamond marker)
    # -------------------------------------------------

    diamond_handle = Line2D(
        [0],
        [0],
        marker="D",
        linestyle="None",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
        label="Total cost /\nTotal cost difference",
    )

    cost_type_handles = [
        Patch(facecolor=colors_cost_type[ct], label=ct) for ct in cost_type_pivot.columns
    ]
    tech_handles = [Patch(facecolor=tech_colors[t], label=t) for t in ordered_techs]

    cost_type_legend = fig.legend(
        handles=[*cost_type_handles, diamond_handle],
        title="Cost type:",
        loc="upper left",
        bbox_to_anchor=(0.82, 0.95),
        frameon=False,
        fontsize=font,
        title_fontsize=font,
        alignment="left",
    )

    cost_gen_legend = fig.legend(
        handles=[*tech_handles, diamond_handle],
        title="Technology:",
        loc="upper left",
        bbox_to_anchor=(0.82, 0.60),
        frameon=False,
        fontsize=font,
        title_fontsize=font,
        alignment="left",
    )

    cost_type_legend.get_title().set_ha("left")
    cost_gen_legend.get_title().set_ha("left")
    cost_type_legend._legend_box.align = "left"
    cost_gen_legend._legend_box.align = "left"

    # -------------------------------------------------
    # Layout
    # -------------------------------------------------
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    fig_path = cnf.FIGURES_DIR / f"{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def price_duration_curve(scenario_list):
    """
    Plots sorted values (ascending) for each scenario.
    X-axis: hours from 0 to n
    Y-axis: value
    """
    duals = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "elec_price.csv").fillna(0)
    duals = duals[duals["scenario"].isin(scenario_list)]

    fig, ax = plt.subplots(figsize=(8, 5))

    scenarios = duals["scenario"].unique()

    for scenario in scenarios:
        df_s = duals[duals["scenario"] == scenario].copy()

        # Sort by value
        df_s = df_s.sort_values(by="value", ascending=False).reset_index(drop=True)

        # Create ascending hour index
        hours = np.arange(len(df_s))

        ax.plot(hours, df_s["value"], label=format_scenario_name(scenario))

    ax.set_xlabel("Hour (sorted)")
    ax.set_ylabel("Value")
    ax.set_xlim(0, 8760)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def vres_vs_shift_subplots(scenario_list, ref_scenario, filename):
    # ---- Load data ----
    supply = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "generation_h.csv").fillna(0)

    supply["VRES"] = supply["PV Roof"] + supply["Wind Onshore"] + supply["PV Alpine"]

    # ---- Prepare reference ----
    df_ref = supply[supply["scenario"] == ref_scenario].copy()
    df_ref["Residual"] = df_ref["Load (Total)"] - df_ref["VRES"]

    flex_down = [f for f in df_ref.columns if "(Down)" in f]
    flex_up = [f for f in df_ref.columns if "(Up)" in f]

    df_ref["Up"] = df_ref[flex_down].sum(axis=1)
    df_ref["Down"] = df_ref[flex_up].sum(axis=1)

    df_ref["Net shift"] = (
        df_ref["Down"] + df_ref["Up"]
    )  # df_ref["Down"] + df_ref["Up"] # + df_ref["Residual"]

    df_ref_sorted = df_ref.sort_values("VRES", ascending=False).reset_index(drop=True)

    x_ref = df_ref_sorted["VRES"]
    y_ref = -df_ref_sorted["Net shift"]

    coef_ref = np.polyfit(x_ref, y_ref, 1)
    fit_ref = np.poly1d(coef_ref)

    # ---- Determine scenarios to plot ----
    scenarios_to_plot = [s for s in scenario_list if s != ref_scenario]

    if len(scenarios_to_plot) == 0:
        print("No scenarios to compare.")
        return

    n = len(scenarios_to_plot)
    title_letters = iter(string.ascii_lowercase)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True, sharex=True)

    if n == 1:
        axes = [axes]

    # ---- Loop over scenarios ----
    for ax, scenario in zip(axes, scenarios_to_plot):
        df = supply[supply["scenario"] == scenario].copy()

        df["Residual"] = df["Load (Total)"] - df["VRES"]

        flex_down = [f for f in df.columns if "(Down)" in f]
        flex_up = [f for f in df.columns if "(Up)" in f]

        df["Up"] = df[flex_down].sum(axis=1)
        df["Down"] = df[flex_up].sum(axis=1)

        df["Net shift"] = df["Down"] + df["Up"]  #  df["Down"] + df["Up"] # + df["Residual"]

        df_sorted = df.sort_values("VRES", ascending=False).reset_index(drop=True)

        x = df_sorted["VRES"]
        y = -df_sorted["Net shift"]

        key = scenario.split("_")[1]  # this will give "s2", but we probably want "hpflex"
        # Better: pick the part that matches your dictionary keys
        key = next((k for k in scenario.split("_") if k in cnf.SCENARIO_COLOR_DICT), "noflex")
        color = cnf.SCENARIO_COLOR_DICT[key]

        # Plot scenario
        ax.plot(
            x,
            y,
            linestyle="None",
            marker="o",
            markersize=3,
            label="Scenario",
            color=color,
            alpha=0.5,
        )

        # Fit scenario
        coef = np.polyfit(x, y, 1)
        fit = np.poly1d(coef)

        x_fit = np.linspace(min(x.min(), x_ref.min()), max(x.max(), x_ref.max()), 200)

        ax.plot(
            x_fit,
            fit(x_fit),
            linestyle="--",
            linewidth=2,
            label="Fit Scenario",
            color=darken_color(color, 0.7),
            zorder=3,
        )

        # Plot reference
        ax.plot(
            x_ref,
            y_ref,
            linestyle="None",
            marker="o",
            markersize=3,
            color="#BEBDBD",
            alpha=0.7,
            label="Reference",
        )

        ax.plot(
            x_fit,
            fit_ref(x_fit),
            color="#272727",
            linestyle="--",
            linewidth=2,
            label="Fit Reference",
            zorder=3,
        )

        # ---- Delta slope ----
        delta_m = coef[0] - coef_ref[0]

        ax.text(
            0.06,
            0.95,
            r"$\Delta M = %.3f\ \mathrm{GWh_{load}/GWh_{VRES}}$" % delta_m,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.5, facecolor="white", edgecolor="white"),
        )

        ax.text(
            0.01,
            1.09,  # slightly left of axes, slightly above
            f"{next(title_letters)}.",  # idx from enumerate
            transform=ax.transAxes,
            fontsize=14,
            va="center",
            ha="left",
        )

        ax.set_title(
            format_scenario_name_inline(scenario)
            + "\nvs "
            + format_scenario_name_inline(ref_scenario),
            loc="left",
            x=0.065,  # shifts the title slightly right
            fontsize=14,
        )

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Load shift [GWh]", fontsize=14)
    for ax in axes:
        ax.set_xlabel("VRES production [GWh]", fontsize=14)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=12)

    legend_elements = [
        Line2D([0], [0], color=cnf.SCENARIO_COLOR_DICT["noflex"], lw=3, label="No flex"),
        Line2D([0], [0], color=cnf.SCENARIO_COLOR_DICT["evflex"], lw=3, label="EV flex"),
        Line2D([0], [0], color=cnf.SCENARIO_COLOR_DICT["hpflex"], lw=3, label="HP flex"),
        Line2D([0], [0], color=cnf.SCENARIO_COLOR_DICT["bothflex"], lw=3, label="EV + HP flex"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Fit line"),
    ]

    # Add legend to figure (not per axis)
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.09),
        frameon=False,
        fontsize=14,
    )

    plt.tight_layout()
    fig_path = cnf.FIGURES_DIR / f"correlation_VRES_shift_{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def load_duration_curves_subplot(scenario_list, filename):
    """Plot load duration curves for multiple scenarios (max 2 per row)."""

    supply = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "generation_h.csv").fillna(0)

    supply["VRES"] = supply["PV Roof"] + supply["Wind Onshore"] + supply["PV Alpine"]

    n = len(scenario_list)

    # ---- Grid layout: max 2 per row ----
    ncols = 2
    nrows = math.ceil(n / 2)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10, 2.8 * nrows),
        sharey=True,
        gridspec_kw={"hspace": 0.3, "wspace": 0.1},
    )

    axes = np.atleast_1d(axes).flatten()
    title_letters = iter(string.ascii_lowercase)

    for ax, scenario in zip(axes, scenario_list):
        df = supply[supply["scenario"] == scenario].copy()

        df["Residual"] = df["Load (Total)"] - df["VRES"]

        flex_down = [f for f in df.columns if "(Down)" in f]
        flex_up = [f for f in df.columns if "(Up)" in f]

        df["Up"] = df[flex_down].sum(axis=1)
        df["Down"] = df[flex_up].sum(axis=1)

        df["Residual_net"] = df["Residual"] + df["Down"] + df["Up"]

        # ---- Sort for duration curve ----
        df_sorted = df.sort_values("Residual", ascending=False).reset_index(drop=True)
        hours = np.arange(len(df_sorted))

        # Base residual
        ax.plot(
            hours,
            df_sorted["Residual"],
            linewidth=1.5,
            color="black",
            label="Shifted residual load",
        )

        # Unshifted (sorted independently)
        residual_net_sorted = df.sort_values("Residual_net", ascending=False).reset_index(
            drop=True
        )["Residual_net"]

        ax.plot(
            hours,
            residual_net_sorted,
            linewidth=1.5,
            linestyle="--",
            color="black",
            label="Unshifted residual load",
        )

        # Delta shading
        delta = df_sorted["Residual_net"] - df_sorted["Residual"]

        ax.fill_between(
            hours,
            df_sorted["Residual"],
            df_sorted["Residual"] + delta,
            where=delta > 0,
            interpolate=True,
            alpha=0.4,
            color="#ce8fca",
            label="Shift Down",
        )

        ax.fill_between(
            hours,
            df_sorted["Residual"],
            df_sorted["Residual"] + delta,
            where=delta < 0,
            interpolate=True,
            alpha=0.4,
            color="#6a4b83",
            label="Shift Up",
        )

        # ---- Title formatting ----
        clean_name = format_scenario_name_inline(scenario)
        letter = next(title_letters)
        ax.set_title(f"{letter}. {clean_name}", fontsize=12, x=0, ha="left")

        ax.set_xlim(0, 8760)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=10)

    # Remove unused axes if odd number
    for i in range(len(scenario_list), len(axes)):
        fig.delaxes(axes[i])

    # Labels only on outer plots
    for ax in axes[-ncols:]:
        ax.set_xlabel("Hours", fontsize=12)

    for ax in axes[::2]:
        ax.set_ylabel("Residual load\n[GW]", fontsize=12)

    # ---- Global legend ----
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
    )

    plt.tight_layout()
    fig_path = cnf.FIGURES_DIR / f"load_duration_curves_{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def hourly_dispatch(
    scenarios,
    periods,
    filename,
):
    """Plot hourly dispatch for multiple scenarios and periods in a grid layout."""
    tech_to_remove = [
        "Import (Net)",
        "Load (Net)",
        "Load (Total)",
    ]

    supply = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "generation_h.csv").fillna(0)

    df = supply.copy()
    df["timestep"] = pd.to_datetime(df["timestep"])

    # Identify tech columns
    tech_cols = [
        c for c in df.columns if c not in ["scenario", "timestep"] and c not in tech_to_remove
    ]

    n_rows = len(scenarios)
    n_cols = 2  # fixed

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11, 4 * n_rows),
        sharey=True,
        squeeze=False,
        gridspec_kw={"hspace": 0.7, "wspace": 0.15},
    )

    letters = list(string.ascii_lowercase)

    global_min = 0
    global_max = 0

    # Compute global y-limits
    for scenario in scenarios:
        df_s = df[df["scenario"] == scenario]

        for start, end in periods.values():
            mask = (df_s["timestep"] >= start) & (df_s["timestep"] <= end)
            df_p = df_s.loc[mask, tech_cols]

            if df_p.empty:
                continue

            pos_sum = df_p.clip(lower=0).sum(axis=1)
            neg_sum = df_p.clip(upper=0).sum(axis=1)

            global_max = max(global_max, pos_sum.max())
            global_min = min(global_min, neg_sum.min())

    legend_handles = {}

    # Plotting
    for i, scenario in enumerate(scenarios):
        df_s = df[df["scenario"] == scenario]

        for j, (period_name, (start, end)) in enumerate(periods.items()):
            ax = axes[i, j]

            mask = (df_s["timestep"] >= start) & (df_s["timestep"] <= end)
            df_p = df_s.loc[mask].copy()

            if df_p.empty:
                continue

            x = df_p["timestep"]

            pos_base = np.zeros(len(df_p))
            neg_base = np.zeros(len(df_p))

            for tech in tech_cols:
                y = df_p[tech].values
                color = cnf.TECH_COLOR_MAPPING.get(tech, "grey")

                y_pos = np.where(y > 0, y, 0)
                y_neg = np.where(y < 0, y, 0)

                # Positive stack
                h_pos = ax.fill_between(
                    x,
                    pos_base,
                    pos_base + y_pos,
                    color=color,
                    linewidth=0,
                )
                pos_base += y_pos

                # Negative stack
                ax.fill_between(
                    x,
                    neg_base,
                    neg_base + y_neg,
                    color=color,
                    linewidth=0,
                )
                neg_base += y_neg

                # Store one handle per tech for legend
                if tech not in legend_handles:
                    legend_handles[tech] = h_pos

                # Overlay Load (Net) line
                if "Load (Total)" in df_s.columns:
                    load_vals = df_s.loc[mask, "Load (Total)"].values
                    ax.plot(x, load_vals, color="black", linewidth=1.0, label="Load (Total)")

            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(global_min * 1.05, global_max * 1.05)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)

            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.tick_params(axis="x", colors="black", which="both")  # label color black
            ax.tick_params(axis="y", colors="black", which="both")  # label color black

            # Make tick lines white
            ax.tick_params(axis="x", color="white", which="both")
            ax.tick_params(axis="y", color="white", which="both")

            # Period title in EVERY row
            ax.set_title(period_name.capitalize(), pad=7, fontsize=12)

            # Row label
            if j == 0:
                ax.text(
                    -0.02,
                    1.16,
                    f"{letters[i]}. {format_scenario_name_inline(scenario)}",
                    transform=ax.transAxes,
                    fontsize=12,
                    ha="left",
                    va="bottom",
                )

            # Custom datetime formatting
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d-%b"))
            ax.tick_params(axis="x", rotation=0)

            # Y label only on left column
            if j == 0:
                ax.set_ylabel("Electricity generation\n[GW]", fontsize=11)

    # Add legend on the right (include Load line)
    handles = list(legend_handles.values())
    labels = list(legend_handles.keys())
    if "Load (Total)" in df.columns:
        handles.append(plt.Line2D([0], [0], color="black", linewidth=1.5))
        labels.append("Load (Total)")

    leg = fig.legend(
        handles,
        labels,
        title="Technologies:",
        loc="center left",
        bbox_to_anchor=(0.94, 0.5),
        frameon=False,
    )

    leg.get_title().set_ha("left")
    leg._legend_box.align = "left"

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig_path = cnf.FIGURES_DIR / f"dispatch_{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_flexibility_quartiles(df, scenario_col="scenario", timestep_col="timestep"):

    df = df.copy()
    df[timestep_col] = pd.to_datetime(df[timestep_col])

    scenarios = df[scenario_col].unique()
    n = len(scenarios)

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), sharey=True)

    if n == 1:
        axes = axes.reshape(1, 2)

    hours = np.arange(24)
    xticks = np.arange(0, 24, 3)
    xtick_labels = [f"{int(h):02d}:00" for h in xticks]

    def compute_daily_stats(df_scen, col_unshifted, col_shifted):

        df_scen = df_scen.copy()
        df_scen["date"] = df_scen[timestep_col].dt.date

        daily_u = []
        daily_s = []

        for _, day in df_scen.groupby("date"):
            max_val = max(day[col_unshifted].max(), day[col_shifted].max())

            if max_val == 0:
                continue

            u = (day[col_unshifted] / max_val).values
            s = (day[col_shifted] / max_val).values

            if len(u) == 24:
                daily_u.append(u)
                daily_s.append(s)

        U = np.array(daily_u)
        S = np.array(daily_s)

        stats = {}

        for name, arr in zip(["unshifted", "shifted"], [U, S]):
            stats[name] = {
                "min": np.percentile(arr, 0, axis=0),
                "q25": np.percentile(arr, 25, axis=0),
                "median": np.percentile(arr, 50, axis=0),
                "q75": np.percentile(arr, 75, axis=0),
                "max": np.percentile(arr, 100, axis=0),
                "mean": arr.mean(axis=0),
            }

        return stats

    for i, scen in enumerate(scenarios):
        df_scen = df[df[scenario_col] == scen]

        # =========================
        # Heat Pump
        # =========================
        stats_hp = compute_daily_stats(df_scen, "Heat Pump", "Shifted Heat Pump")

        ax_hp = axes[i, 0]

        # --- Unshifted shading ---
        ax_hp.fill_between(
            hours,
            stats_hp["unshifted"]["min"],
            stats_hp["unshifted"]["max"],
            color="black",
            alpha=0.08,
        )

        ax_hp.fill_between(
            hours,
            stats_hp["unshifted"]["q25"],
            stats_hp["unshifted"]["q75"],
            color="black",
            alpha=0.25,
        )

        # --- Shifted shading ---
        ax_hp.fill_between(
            hours,
            stats_hp["shifted"]["min"],
            stats_hp["shifted"]["max"],
            color="lightgreen",
            alpha=0.08,
        )

        ax_hp.fill_between(
            hours,
            stats_hp["shifted"]["q25"],
            stats_hp["shifted"]["q75"],
            color="lightgreen",
            alpha=0.25,
        )

        # Median
        ax_hp.plot(hours, stats_hp["unshifted"]["median"], color="black", linewidth=2)

        ax_hp.plot(hours, stats_hp["shifted"]["median"], color="lightgreen", linewidth=2)

        # Mean
        ax_hp.plot(hours, stats_hp["unshifted"]["mean"], color="black", linestyle="--", linewidth=1)

        ax_hp.plot(hours, stats_hp["shifted"]["mean"], color="black", linestyle="--", linewidth=1)

        ax_hp.set_xlim(0, 23)
        ax_hp.set_xticks(xticks)
        ax_hp.set_xticklabels(xtick_labels)

        if i == 0:
            ax_hp.set_title("Daily heat pump flexibility")

        ax_hp.set_ylabel(scen)

        # =========================
        # E-Mobility
        # =========================
        stats_em = compute_daily_stats(df_scen, "e-Mobility", "Shifted e-Mobility")

        ax_em = axes[i, 1]

        # Unshifted shading
        ax_em.fill_between(
            hours,
            stats_em["unshifted"]["min"],
            stats_em["unshifted"]["max"],
            color="black",
            alpha=0.08,
        )

        ax_em.fill_between(
            hours,
            stats_em["unshifted"]["q25"],
            stats_em["unshifted"]["q75"],
            color="black",
            alpha=0.25,
        )

        # Shifted shading
        ax_em.fill_between(
            hours,
            stats_em["shifted"]["min"],
            stats_em["shifted"]["max"],
            color="lightgreen",
            alpha=0.08,
        )

        ax_em.fill_between(
            hours,
            stats_em["shifted"]["q25"],
            stats_em["shifted"]["q75"],
            color="lightgreen",
            alpha=0.25,
        )

        ax_em.plot(hours, stats_em["unshifted"]["median"], color="black", linewidth=2)

        ax_em.plot(hours, stats_em["shifted"]["median"], color="lightgreen", linewidth=2)

        ax_em.plot(hours, stats_em["unshifted"]["mean"], color="black", linestyle="--", linewidth=1)

        ax_em.plot(hours, stats_em["shifted"]["mean"], color="black", linestyle="--", linewidth=1)

        ax_em.set_xlim(0, 23)
        ax_em.set_xticks(xticks)
        ax_em.set_xticklabels(xtick_labels)

        if i == 0:
            ax_em.set_title("Daily E-mobility flexibility")

    # =========================
    # Legend
    # =========================
    handles = [
        Line2D([0], [0], color="black", linewidth=2, label="Unshifted"),
        Line2D([0], [0], color="lightgreen", linewidth=2, label="Shifted"),
        Line2D([0], [0], color="black", linestyle="--", label="Mean"),
        Line2D([0], [0], color="black", linestyle="-", label="Median"),
        Patch(facecolor="black", alpha=0.08, label="5–95% (Unshifted)"),
        Patch(facecolor="black", alpha=0.25, label="25–75% (Unshifted)"),
        Patch(facecolor="lightgreen", alpha=0.08, label="5–95% (Shifted)"),
        Patch(facecolor="lightgreen", alpha=0.25, label="25–75% (Shifted)"),
    ]

    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def plot_daily_positive_negative(df, scenario_col="scenario", timestep_col="timestep"):
    """
    Plots daily normalized curves for Heat Pump and E-Mobility per scenario,
    shading positive differences (shifted > unshifted) in green and negative
    (shifted < unshifted) in red, alpha=0.05.
    """
    # Ensure datetime
    df[timestep_col] = pd.to_datetime(df[timestep_col])

    # Extract date and time-of-day
    df["date"] = df[timestep_col].dt.date
    df["time_of_day"] = (
        df[timestep_col].dt.hour
        + df[timestep_col].dt.minute / 60
        + df[timestep_col].dt.second / 3600
    )

    scenarios = df[scenario_col].unique()
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(n_scenarios, 2, figsize=(12, 4 * n_scenarios), sharex=True)

    if n_scenarios == 1:
        axes = axes.reshape(1, 2)

    def fill_pos_neg(ax, x, y1, y2):
        """Fill area between y1 and y2 with green if y2>y1, red if y2<y1"""
        diff = y2 - y1
        # Positive part
        ax.fill_between(x, y1, y2, where=(diff > 0), color="green", alpha=0.01, interpolate=True)
        # Negative part
        ax.fill_between(x, y1, y2, where=(diff < 0), color="red", alpha=0.01, interpolate=True)

    for i, scen in enumerate(scenarios):
        df_scen = df[df[scenario_col] == scen]
        grouped = df_scen.groupby("date")

        # Heat Pump
        ax1 = axes[i, 0]
        for _, day in grouped:
            max_hp = max(day["Heat Pump"].max(), day["Shifted Heat Pump"].max())
            hp = day["Heat Pump"] / max_hp
            shp = day["Shifted Heat Pump"] / max_hp
            fill_pos_neg(ax1, day["time_of_day"], hp, shp)
        ax1.set_title(f"{scen} — Heat Pump: Shifted vs Unshifted (daily)")
        ax1.set_ylabel("Normalized")
        ax1.set_xlim(0, 24)
        ax1.set_xticks(range(0, 25, 3))
        ax1.set_xlabel("Hour of Day")

        # E-Mobility
        ax2 = axes[i, 1]
        for _, day in grouped:
            max_em = max(day["e-Mobility"].max(), day["Shifted e-Mobility"].max())
            em = day["e-Mobility"] / max_em
            sem = day["Shifted e-Mobility"] / max_em
            fill_pos_neg(ax2, day["time_of_day"], em, sem)
        ax2.set_title(f"{scen} — E-Mobility: Shifted vs Unshifted (daily)")
        ax2.set_xlim(0, 24)
        ax2.set_xticks(range(0, 25, 3))
        ax2.set_xlabel("Hour of Day")

    plt.tight_layout()
    plt.show()


def flexibility_3rows_updown(
    scenario_list,
    filename,
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[1, 3, 2],
    hspace=0.5,
):
    """Plot cumulative flexibility and horly dispersion.

    Args:
        df (_type_): _description_
        filename (_type_): _description_
        scenario_col (str, optional): _description_. Defaults to "scenario".
        timestep_col (str, optional): _description_. Defaults to "timestep".
        height_ratios_per_scenario (list, optional): _description_. Defaults to [1, 3, 2].
        hspace (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    df = helper.cumulative_shifted_demand(scenario_list)
    df = df.copy()
    df[timestep_col] = pd.to_datetime(df[timestep_col])
    scenarios = df[scenario_col].unique()
    n = len(scenarios)
    letters = list(string.ascii_lowercase)
    hours = np.arange(24)
    xticks = np.arange(0, 24, 3)
    xtick_labels = [f"{int(h):02d}:00" for h in xticks]

    n_rows = n * 3
    height_ratios = height_ratios_per_scenario * n
    fig = plt.figure(figsize=(11, 4 * n))
    gs = GridSpec(n_rows, 2, figure=fig, height_ratios=height_ratios, hspace=hspace)

    # --------------------------
    # NEW: store shift axes + global limits
    # --------------------------
    shift_axes = []
    global_shift_min = np.inf
    global_shift_max = -np.inf

    for i, scen in enumerate(scenarios):
        df_scen = df[df[scenario_col] == scen]
        title_row = i * 3
        abs_row = i * 3 + 1
        shift_row = i * 3 + 2

        ax_title_left = fig.add_subplot(gs[title_row, 0])
        ax_title_left.axis("off")
        ax_title_left.text(
            0.01,
            -0.28,
            f"{letters[i]}. {format_scenario_name_inline(scen)}",
            ha="left",
            va="center",
            fontsize=12,
            transform=ax_title_left.transAxes,
        )
        ax_title_right = fig.add_subplot(gs[title_row, 1])
        ax_title_right.axis("off")

        for j, (col_abs_u, col_abs_s, col_label, shift_up_col, shift_down_col) in enumerate(
            [
                (
                    "Heat Pump",
                    "Shifted Heat Pump",
                    "Heat pump flexibility",
                    "HP Shift (Up)",
                    "HP Shift (Down)",
                ),
                (
                    "e-Mobility",
                    "Shifted e-Mobility",
                    "E-mobility flexibility",
                    "EMob Shift (Up)",
                    "EMob Shift (Down)",
                ),
            ]
        ):
            # ---------------- Row 1 (UNCHANGED)
            daily_u, daily_s = [], []
            for _, day in df_scen.groupby(df_scen[timestep_col].dt.date):
                max_val = max(day[col_abs_u].max(), day[col_abs_s].max())
                if max_val == 0:
                    continue
                u = (day[col_abs_u] / max_val).values
                s = (day[col_abs_s] / max_val).values
                if len(u) == 24:
                    daily_u.append(u)
                    daily_s.append(s)

            U = np.array(daily_u)
            S = np.array(daily_s)

            stats_u = get_stats(U)
            stats_s = get_stats(S)

            if j == 0:
                ax_abs = fig.add_subplot(gs[abs_row, j])
            else:
                ax_abs = fig.add_subplot(gs[abs_row, j], sharey=ax_abs)

            for stats, color in [(stats_s, "orange"), (stats_u, "black")]:
                ax_abs.fill_between(hours, stats["p0"], stats["p100"], color=color, alpha=0.15)
                ax_abs.fill_between(hours, stats["p10"], stats["p90"], color=color, alpha=0.22)
                ax_abs.fill_between(hours, stats["p25"], stats["p75"], color=color, alpha=0.35)
                ax_abs.plot(hours, stats["p50"], color=color, linewidth=2)

            ax_abs.set_xlim(0, 23)
            ax_abs.set_xticks(xticks)
            ax_abs.set_xticklabels(xtick_labels)
            if j == 0:
                ax_abs.set_ylabel("Normalised\ncumulative load\n[-]")
                ax_abs.yaxis.set_label_coords(-0.13, 0.5)
            if i == 0:
                ax_abs.set_title(col_label, pad=35)
            ax_abs.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.1f}"))
            ax_abs.grid(True, linestyle="--", alpha=0.4)
            ax_abs.set_axisbelow(True)
            ax_abs.tick_params(axis="x", color="white", which="both")
            ax_abs.tick_params(axis="y", color="white", which="both")
            ax_abs.set_facecolor("whitesmoke")
            for spine in ax_abs.spines.values():
                spine.set_visible(False)

            # ---------------- Row 2: SHIFT (UPDATED LIMIT LOGIC)

            daily_up, daily_down = [], []
            for _, day in df_scen.groupby(df_scen[timestep_col].dt.date):
                u = day[shift_up_col].values
                d = day[shift_down_col].values
                if len(u) != 24 or len(d) != 24:
                    continue
                daily_up.append(u)
                daily_down.append(d)

            UP = np.array(daily_up) * -1
            DOWN = np.array(daily_down) * -1

            stats_up = get_percentiles(UP)
            stats_down = get_percentiles(DOWN)

            if j == 0:
                ax_shift = fig.add_subplot(gs[shift_row, j])
            else:
                ax_shift = fig.add_subplot(gs[shift_row, j], sharey=ax_shift)

            shift_axes.append(ax_shift)

            # Update global limits using full percentile envelope
            global_shift_min = min(
                global_shift_min,
                stats_up["p0"].min(),
                stats_down["p0"].min(),
            )
            global_shift_max = max(
                global_shift_max,
                stats_up["p100"].max(),
                stats_down["p100"].max(),
            )

            # Plot
            ax_shift.fill_between(
                hours, stats_up["p0"], stats_up["p100"], color="purple", alpha=0.18
            )
            ax_shift.fill_between(
                hours, stats_up["p10"], stats_up["p90"], color="purple", alpha=0.28
            )
            ax_shift.fill_between(
                hours, stats_up["p25"], stats_up["p75"], color="purple", alpha=0.45
            )
            ax_shift.plot(hours, stats_up["p50"], color="purple", linewidth=1.5)

            ax_shift.fill_between(
                hours, stats_down["p0"], stats_down["p100"], color="deeppink", alpha=0.18
            )
            ax_shift.fill_between(
                hours, stats_down["p10"], stats_down["p90"], color="deeppink", alpha=0.28
            )
            ax_shift.fill_between(
                hours, stats_down["p25"], stats_down["p75"], color="deeppink", alpha=0.45
            )
            ax_shift.plot(hours, stats_down["p50"], color="deeppink", linewidth=1.5)

            ax_shift.axhline(0, color="black", linewidth=0.8)
            ax_shift.set_xlim(0, 23)
            ax_shift.set_xticks(xticks)
            ax_shift.set_xticklabels(xtick_labels)
            ax_shift.grid(True, linestyle="--", alpha=0.4)
            ax_shift.set_axisbelow(True)
            ax_shift.tick_params(axis="x", color="white", which="both")
            ax_shift.tick_params(axis="y", color="white", which="both")
            ax_shift.set_facecolor("whitesmoke")

            if j == 0:
                ax_shift.set_ylabel("Load\nshift\n[GWh]")
                ax_shift.yaxis.set_label_coords(-0.13, 0.5)

            ax_shift.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.1f}"))
            for spine in ax_shift.spines.values():
                spine.set_visible(False)

    # --------------------------
    # APPLY GLOBAL SHIFT LIMITS
    # --------------------------
    for ax in shift_axes:
        ax.set_ylim(global_shift_min * 1.5, global_shift_max * 1.01)

    handles_1 = [
        Line2D([0], [0], color="black", linewidth=2, label="Cumulative Unshifted"),
        Line2D([0], [0], color="orange", linewidth=2, label="Cumulative Shifted"),
        Line2D([0], [0], color="purple", linewidth=2, label="Shift Up"),
        Line2D([0], [0], color="deeppink", linewidth=2, label="Shift Down"),
    ]

    handles_2 = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="Median"),
        Patch(facecolor="dimgray", alpha=0.18, label="5–95%"),
        Patch(facecolor="dimgray", alpha=0.28, label="10–90%"),
        Patch(facecolor="dimgray", alpha=0.45, label="25–75%"),
    ]

    legend1 = fig.legend(
        handles=handles_1,
        loc="upper left",
        bbox_to_anchor=(0.17, 0.08),  # figure coordinates
        bbox_transform=fig.transFigure,
        ncol=4,
        frameon=False,
        title="Flexibility categories:",
    )

    legend2 = fig.legend(
        handles=handles_2,
        loc="upper left",
        bbox_to_anchor=(0.17, 0.04),  # slightly below first legend
        bbox_transform=fig.transFigure,
        ncol=4,
        frameon=False,
        title="Percentiles:",
    )

    legend1.get_title().set_ha("left")
    legend2.get_title().set_ha("left")
    legend1._legend_box.align = "left"
    legend2._legend_box.align = "left"

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = cnf.FIGURES_DIR / f"{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def flexibility_4cols_seasonal(
    scenario_list,
    filename_prefix,
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[1, 3, 2],
    hspace=0.5,
):
    """Plot cumulative flexibility and hourly dispersion per season.

    Args:
        scenario_list: list of scenarios or DataFrame
        filename_prefix: prefix for saved figures
        scenario_col (str, optional): name of scenario column
        timestep_col (str, optional): name of timestep column
        height_ratios_per_scenario (list, optional): relative row heights
        hspace (float, optional): vertical spacing
    """

    df = helper.cumulative_shifted_demand(scenario_list)
    df = df.copy()
    df[timestep_col] = pd.to_datetime(df[timestep_col])

    # Map months to seasons
    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }
    df["season"] = df[timestep_col].dt.month.map(season_map)
    seasons = ["Winter", "Spring", "Summer", "Autumn"]

    scenarios = df[scenario_col].unique()
    letters = list(string.ascii_lowercase)
    hours = np.arange(24)
    xticks = np.arange(0, 24, 6)
    xtick_labels = [f"{int(h):02d}:00" for h in xticks]

    # --------------------------
    # Plot one figure per technology
    # --------------------------
    for tech_col, tech_label, abs_col, shift_up_col, shift_down_col in [
        (
            "Heat Pump",
            "Heat pump flexibility",
            "Shifted Heat Pump",
            "HP Shift (Up)",
            "HP Shift (Down)",
        ),
        (
            "e-Mobility",
            "E-mobility flexibility",
            "Shifted e-Mobility",
            "EMob Shift (Up)",
            "EMob Shift (Down)",
        ),
    ]:
        n = len(scenarios)
        n_rows = n * 3
        height_ratios = height_ratios_per_scenario * n

        fig = plt.figure(figsize=(13, 4 * n))
        gs = GridSpec(
            n_rows, 4, figure=fig, height_ratios=height_ratios, hspace=hspace, wspace=0.07
        )

        shift_axes = []
        global_shift_min = np.inf
        global_shift_max = -np.inf

        for i, scen in enumerate(scenarios):
            df_scen = df[df[scenario_col] == scen]
            title_row = i * 3
            abs_row = i * 3 + 1
            shift_row = i * 3 + 2

            # Title
            ax_title_left = fig.add_subplot(gs[title_row, 0])
            ax_title_left.axis("off")
            ax_title_left.text(
                0.01,
                -0.05,
                f"{letters[i]}. {format_scenario_name_inline(scen)}",
                ha="left",
                va="center",
                fontsize=12,
                transform=ax_title_left.transAxes,
            )
            ax_title_right = fig.add_subplot(gs[title_row, 1:])
            ax_title_right.axis("off")

            for j, season in enumerate(seasons):
                df_season = df_scen[df_scen["season"] == season]

                # ---------------- Row 1: cumulative
                daily_u, daily_s = [], []
                for _, day in df_season.groupby(df_season[timestep_col].dt.date):
                    max_val = max(day[tech_col].max(), day[abs_col].max())
                    if max_val == 0:
                        continue
                    u = (day[tech_col] / max_val).values
                    s = (day[abs_col] / max_val).values
                    if len(u) == 24:
                        daily_u.append(u)
                        daily_s.append(s)

                U = np.array(daily_u)
                S = np.array(daily_s)

                stats_u = get_stats(U)
                stats_s = get_stats(S)

                if i == 0:
                    ax_abs = fig.add_subplot(gs[abs_row, j])
                else:
                    ax_abs = fig.add_subplot(gs[abs_row, j])

                for data, color in [(stats_s, "orange"), (stats_u, "black")]:
                    ax_abs.fill_between(hours, data["p0"], data["p100"], color=color, alpha=0.15)
                    ax_abs.fill_between(hours, data["p10"], data["p90"], color=color, alpha=0.22)
                    ax_abs.fill_between(hours, data["p25"], data["p75"], color=color, alpha=0.35)
                    ax_abs.plot(hours, data["p50"], color=color, linewidth=2)

                ax_abs.set_xlim(0, 23)
                ax_abs.set_xticks(xticks)
                ax_abs.set_xticklabels(xtick_labels)
                if j == 0:
                    ax_abs.set_ylabel("Normalised\ncumulative load\n[-]")
                    ax_abs.yaxis.set_label_coords(-0.2, 0.5)
                ax_abs.set_title(season, pad=5)
                ax_abs.grid(True, linestyle="--", alpha=0.4)
                ax_abs.set_axisbelow(True)
                # ax_abs.tick_params(axis="x", color="white", which="both")
                # ax_abs.tick_params(axis="y", color="white", which="both")
                ax_abs.set_facecolor("whitesmoke")
                ax_abs.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.1f}"))
                for spine in ax_abs.spines.values():
                    spine.set_visible(True)

                # spine adjustments
                if j == 0:  # first column
                    ax_abs.spines["left"].set_visible(True)
                    ax_abs.spines["right"].set_visible(False)
                elif j == 4 - 1:  # last column
                    ax_abs.spines["left"].set_visible(False)
                    ax_abs.spines["right"].set_visible(True)
                else:  # middle columns
                    ax_abs.spines["left"].set_visible(False)
                    ax_abs.spines["right"].set_visible(False)

                if j != 0:
                    # ax_abs.tick_params(axis="y", colors="white")  # tick lines
                    ax_abs.set_yticklabels([])
                    ax_abs.tick_params(axis="y", color="white", which="both")

                # ---------------- Row 2: shift
                daily_up, daily_down = [], []
                for _, day in df_season.groupby(df_season[timestep_col].dt.date):
                    u = day[shift_up_col].values
                    d = day[shift_down_col].values
                    if len(u) != 24 or len(d) != 24:
                        continue
                    daily_up.append(u)
                    daily_down.append(d)

                UP = np.array(daily_up) * -1
                DOWN = np.array(daily_down) * -1

                stats_up = get_percentiles(UP)
                stats_down = get_percentiles(DOWN)

                if i == 0:
                    ax_shift = fig.add_subplot(gs[shift_row, j])
                else:
                    ax_shift = fig.add_subplot(gs[shift_row, j])

                shift_axes.append(ax_shift)

                global_shift_min = min(
                    global_shift_min, stats_up["p0"].min(), stats_down["p0"].min()
                )
                global_shift_max = max(
                    global_shift_max, stats_up["p100"].max(), stats_down["p100"].max()
                )

                ax_shift.fill_between(
                    hours, stats_up["p0"], stats_up["p100"], color="purple", alpha=0.18
                )
                ax_shift.fill_between(
                    hours, stats_up["p10"], stats_up["p90"], color="purple", alpha=0.28
                )
                ax_shift.fill_between(
                    hours, stats_up["p25"], stats_up["p75"], color="purple", alpha=0.45
                )
                ax_shift.plot(hours, stats_up["p50"], color="purple", linewidth=1.5)

                ax_shift.fill_between(
                    hours, stats_down["p0"], stats_down["p100"], color="deeppink", alpha=0.18
                )
                ax_shift.fill_between(
                    hours, stats_down["p10"], stats_down["p90"], color="deeppink", alpha=0.28
                )
                ax_shift.fill_between(
                    hours, stats_down["p25"], stats_down["p75"], color="deeppink", alpha=0.45
                )
                ax_shift.plot(hours, stats_down["p50"], color="deeppink", linewidth=1.5)

                ax_shift.axhline(0, color="black", linewidth=0.8)
                ax_shift.set_xlim(0, 23)
                ax_shift.set_xticks(xticks)
                ax_shift.set_xticklabels(xtick_labels)
                ax_shift.grid(True, linestyle="--", alpha=0.4)
                ax_shift.set_axisbelow(True)
                # ax_shift.tick_params(axis="x", color="white", which="both")
                # ax_shift.tick_params(axis="y", color="white", which="both")
                ax_shift.set_facecolor("whitesmoke")
                ax_shift.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.1f}"))
                if j == 0:
                    ax_shift.set_ylabel("Load\nshift\n[GWh]")
                    ax_shift.yaxis.set_label_coords(-0.2, 0.5)

                for spine in ax_shift.spines.values():
                    spine.set_visible(True)

                # spine adjustments
                if j == 0:  # first column
                    ax_shift.spines["left"].set_visible(True)
                    ax_shift.spines["right"].set_visible(False)
                elif j == 4 - 1:  # last column
                    ax_shift.spines["left"].set_visible(False)
                    ax_shift.spines["right"].set_visible(True)
                else:  # middle columns
                    ax_shift.spines["left"].set_visible(False)
                    ax_shift.spines["right"].set_visible(False)

                if j != 0:
                    # ax_shift.tick_params(axis="y", colors="white")  # tick lines
                    ax_shift.set_yticklabels([])
                    ax_shift.tick_params(axis="y", color="white", which="both")

        # Apply global limits
        for ax in shift_axes:
            ax.set_ylim(global_shift_min * 1.5, global_shift_max * 1.01)

        # Legends
        handles_1 = [
            Line2D([0], [0], color="black", linewidth=2, label="Cumulative Unshifted"),
            Line2D([0], [0], color="orange", linewidth=2, label="Cumulative Shifted"),
            Line2D([0], [0], color="purple", linewidth=2, label="Shift Up"),
            Line2D([0], [0], color="deeppink", linewidth=2, label="Shift Down"),
        ]
        handles_2 = [
            Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="Median"),
            Patch(facecolor="dimgray", alpha=0.18, label="5–95%"),
            Patch(facecolor="dimgray", alpha=0.28, label="10–90%"),
            Patch(facecolor="dimgray", alpha=0.45, label="25–75%"),
        ]

        leg_2_pos = len(scenario_list) * 0.01
        leg_1_pos = 0.08

        legend1 = fig.legend(
            handles=handles_1,
            loc="upper left",
            bbox_to_anchor=(0.17, leg_1_pos),
            bbox_transform=fig.transFigure,
            ncol=4,
            frameon=False,
            title=f"{tech_label} categories:",
        )
        legend2 = fig.legend(
            handles=handles_2,
            loc="upper left",
            bbox_to_anchor=(0.17, leg_2_pos),
            bbox_transform=fig.transFigure,
            ncol=4,
            frameon=False,
            title="Percentiles:",
        )
        legend1.get_title().set_ha("left")
        legend2.get_title().set_ha("left")
        legend1._legend_box.align = "left"
        legend2._legend_box.align = "left"

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = (
            cnf.FIGURES_DIR / f"{filename_prefix}_{tech_col.replace(' ', '_')}.png"
        )
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()


def flexibility_seasonal_percentile(
    scenario_list,
    filename_prefix,
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[1, 3, 2],
    hspace=0.5,
    min_abs_y=None,
):
    """Plot flexibility and hourly dispersion per season.

    Args:
        scenario_list: list of scenarios or DataFrame
        filename_prefix: prefix for saved figures
        scenario_col (str, optional): name of scenario column
        timestep_col (str, optional): name of timestep column
        height_ratios_per_scenario (list, optional): relative row heights
        hspace (float, optional): vertical spacing
    """
    df = helper.hourly_supply_demand(scenario_list)
    df[timestep_col] = pd.to_datetime(df[timestep_col])

    # Map months to seasons
    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }
    df["season"] = df[timestep_col].dt.month.map(season_map)
    seasons = ["Winter", "Spring", "Summer", "Autumn"]

    scenarios = df[scenario_col].unique()
    letters = list(string.ascii_lowercase)
    hours = np.arange(24)
    xticks = np.arange(0, 24, 6)
    xtick_labels = [f"{int(h):02d}:00" for h in xticks]

    # --------------------------
    # Plot one figure per technology
    # --------------------------
    for tech_col, tech_label, abs_col, shift_up_col, shift_down_col in [
        (
            "Heat Pump",
            "Heat pump flexibility",
            "Shifted Heat Pump",
            "HP Shift (Up)",
            "HP Shift (Down)",
        ),
        (
            "e-Mobility",
            "E-mobility flexibility",
            "Shifted e-Mobility",
            "EMob Shift (Up)",
            "EMob Shift (Down)",
        ),
    ]:
        n = len(scenarios)
        n_rows = n * 3
        height_ratios = height_ratios_per_scenario * n

        fig = plt.figure(figsize=(13, 4 * n))
        gs = GridSpec(
            n_rows, 4, figure=fig, height_ratios=height_ratios, hspace=hspace, wspace=0.06
        )

        shift_axes = []
        global_shift_min = np.inf
        global_shift_max = -np.inf

        abs_axes = []
        global_abs_min = np.inf
        global_abs_max = -np.inf

        for i, scen in enumerate(scenarios):
            df_scen = df[df[scenario_col] == scen]
            title_row = i * 3
            abs_row = i * 3 + 1
            shift_row = i * 3 + 2

            # Title
            ax_title_left = fig.add_subplot(gs[title_row, 0])
            ax_title_left.axis("off")
            ax_title_left.text(
                0.01,
                -0.02,
                f"{letters[i]}. {format_scenario_name_inline(scen)}",
                ha="left",
                va="center",
                fontsize=12,
                transform=ax_title_left.transAxes,
            )
            ax_title_right = fig.add_subplot(gs[title_row, 1:])
            ax_title_right.axis("off")

            for j, season in enumerate(seasons):
                df_season = df_scen[df_scen["season"] == season]

                # ---------------- Row 1
                daily_u, daily_s = [], []
                for _, day in df_season.groupby(df_season[timestep_col].dt.date):
                    max_val = max(day[tech_col].max(), day[abs_col].max())
                    if max_val == 0:
                        continue
                    u = (day[tech_col]).values
                    s = (day[abs_col]).values
                    if len(u) == 24:
                        daily_u.append(u)
                        daily_s.append(s)

                U = np.array(daily_u)
                S = np.array(daily_s)

                stats_u = get_stats(U)
                stats_s = get_stats(S)

                if i == 0:
                    ax_abs = fig.add_subplot(gs[abs_row, j])
                else:
                    ax_abs = fig.add_subplot(gs[abs_row, j])

                abs_axes.append(ax_abs)

                for data, color in [(stats_s, "orange"), (stats_u, "black")]:
                    ax_abs.fill_between(hours, data["p0"], data["p100"], color=color, alpha=0.15)
                    ax_abs.fill_between(hours, data["p10"], data["p90"], color=color, alpha=0.22)
                    ax_abs.fill_between(hours, data["p25"], data["p75"], color=color, alpha=0.35)
                    ax_abs.plot(hours, data["p50"], color=color, linewidth=2)

                global_abs_min = min(global_abs_min, stats_s["p0"].min(), stats_u["p0"].min())
                global_abs_max = max(global_abs_max, stats_s["p100"].max(), stats_u["p100"].max())

                ax_abs.set_xlim(0, 23)
                ax_abs.set_xticks(xticks)
                ax_abs.set_xticklabels(xtick_labels)
                if j == 0:
                    ax_abs.set_ylabel("Demand\n[GWh]")
                    ax_abs.yaxis.set_label_coords(-0.2, 0.5)
                ax_abs.set_title(season, pad=5)
                ax_abs.grid(True, linestyle="--", alpha=0.4)
                ax_abs.set_axisbelow(True)
                # ax_abs.tick_params(axis="x", color="white", which="both")
                # ax_abs.tick_params(axis="y", color="white", which="both")
                ax_abs.set_facecolor("whitesmoke")
                ax_abs.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.1f}"))
                ax_abs.axhline(0, color="black", linewidth=0.8)

                for spine in ax_abs.spines.values():
                    spine.set_visible(True)

                # spine adjustments
                if j == 0:  # first column
                    ax_abs.spines["left"].set_visible(True)
                    ax_abs.spines["right"].set_visible(False)
                elif j == 4 - 1:  # last column
                    ax_abs.spines["left"].set_visible(False)
                    ax_abs.spines["right"].set_visible(True)
                else:  # middle columns
                    ax_abs.spines["left"].set_visible(False)
                    ax_abs.spines["right"].set_visible(False)

                if j != 0:
                    # ax_abs.tick_params(axis="y", colors="white")  # tick lines
                    ax_abs.set_yticklabels([])
                    ax_abs.tick_params(axis="y", color="white", which="both")

                # ---------------- Row 2: shift
                daily_up, daily_down = [], []
                for _, day in df_season.groupby(df_season[timestep_col].dt.date):
                    u = day[shift_up_col].values
                    d = day[shift_down_col].values
                    if len(u) != 24 or len(d) != 24:
                        continue
                    daily_up.append(u)
                    daily_down.append(d)

                UP = np.array(daily_up) * -1
                DOWN = np.array(daily_down) * -1

                stats_up = get_percentiles(UP)
                stats_down = get_percentiles(DOWN)

                if i == 0:
                    ax_shift = fig.add_subplot(gs[shift_row, j])
                else:
                    ax_shift = fig.add_subplot(gs[shift_row, j])

                shift_axes.append(ax_shift)

                global_shift_min = min(
                    global_shift_min, stats_up["p0"].min(), stats_down["p0"].min()
                )
                global_shift_max = max(
                    global_shift_max, stats_up["p100"].max(), stats_down["p100"].max()
                )

                ax_shift.fill_between(
                    hours, stats_up["p0"], stats_up["p100"], color="purple", alpha=0.18
                )
                ax_shift.fill_between(
                    hours, stats_up["p10"], stats_up["p90"], color="purple", alpha=0.28
                )
                ax_shift.fill_between(
                    hours, stats_up["p25"], stats_up["p75"], color="purple", alpha=0.45
                )
                ax_shift.plot(hours, stats_up["p50"], color="purple", linewidth=1.5)

                ax_shift.fill_between(
                    hours, stats_down["p0"], stats_down["p100"], color="deeppink", alpha=0.18
                )
                ax_shift.fill_between(
                    hours, stats_down["p10"], stats_down["p90"], color="deeppink", alpha=0.28
                )
                ax_shift.fill_between(
                    hours, stats_down["p25"], stats_down["p75"], color="deeppink", alpha=0.45
                )
                ax_shift.plot(hours, stats_down["p50"], color="deeppink", linewidth=1.5)

                ax_shift.axhline(0, color="black", linewidth=0.8)
                ax_shift.set_xlim(0, 23)
                ax_shift.set_xticks(xticks)
                ax_shift.set_xticklabels(xtick_labels)
                ax_shift.grid(True, linestyle="--", alpha=0.4)
                ax_shift.set_axisbelow(True)
                # ax_shift.tick_params(axis="x", color="white", which="both")
                # ax_shift.tick_params(axis="y", color="white", which="both")
                ax_shift.set_facecolor("whitesmoke")
                ax_shift.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.1f}"))
                if j == 0:
                    ax_shift.set_ylabel("Load shift\n[GWh]")
                    ax_shift.yaxis.set_label_coords(-0.2, 0.5)

                for spine in ax_shift.spines.values():
                    spine.set_visible(True)

                # spine adjustments
                if j == 0:  # first column
                    ax_shift.spines["left"].set_visible(True)
                    ax_shift.spines["right"].set_visible(False)
                elif j == 4 - 1:  # last column
                    ax_shift.spines["left"].set_visible(False)
                    ax_shift.spines["right"].set_visible(True)
                else:  # middle columns
                    ax_shift.spines["left"].set_visible(False)
                    ax_shift.spines["right"].set_visible(False)

                if j != 0:
                    # ax_shift.tick_params(axis="y", colors="white")  # tick lines
                    ax_shift.set_yticklabels([])
                    ax_shift.tick_params(axis="y", color="white", which="both")

        # Apply global limits
        for ax in shift_axes:
            ax.set_ylim(global_shift_min * 1.5, global_shift_max * 1.01)

        # Apply global limits
        for ax in abs_axes:
            if min_abs_y is not None:
                global_abs_min = min_abs_y

            ax.set_ylim(min(0, global_abs_min), global_abs_max * 1.01)

        # Legends
        handles_1 = [
            Line2D([0], [0], color="black", linewidth=2, label="Unshifted"),
            Line2D([0], [0], color="orange", linewidth=2, label="Shifted"),
            Line2D([0], [0], color="purple", linewidth=2, label="Shift Up"),
            Line2D([0], [0], color="deeppink", linewidth=2, label="Shift Down"),
        ]
        handles_2 = [
            Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="Median"),
            Patch(facecolor="dimgray", alpha=0.18, label="5–95%"),
            Patch(facecolor="dimgray", alpha=0.28, label="10–90%"),
            Patch(facecolor="dimgray", alpha=0.45, label="25–75%"),
        ]

        leg_2_pos = len(scenario_list) * 0.01
        leg_1_pos = 0.08

        legend1 = fig.legend(
            handles=handles_1,
            loc="upper left",
            bbox_to_anchor=(0.17, leg_1_pos),
            bbox_transform=fig.transFigure,
            ncol=4,
            frameon=False,
            title=f"{tech_label} categories:",
        )
        legend2 = fig.legend(
            handles=handles_2,
            loc="upper left",
            bbox_to_anchor=(0.17, leg_2_pos),
            bbox_transform=fig.transFigure,
            ncol=4,
            frameon=False,
            title="Percentiles:",
        )
        legend1.get_title().set_ha("left")
        legend2.get_title().set_ha("left")
        legend1._legend_box.align = "left"
        legend2._legend_box.align = "left"

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = (
            cnf.FIGURES_DIR / f"{filename_prefix}_{tech_col.replace(' ', '_')}.png"
        )
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()


def tech_dispatch_seasonal_percentile(
    scenario_list,
    filename_prefix,
    ref_scenario,
    tech_list,
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[1, 3, 2],
    hspace=0.5,
    min_abs_y=None,
):
    """Plot flexibility and hourly dispersion per season.

    Args:
        scenario_list: list of scenarios or DataFrame
        filename_prefix: prefix for saved figures
        scenario_col (str, optional): name of scenario column
        timestep_col (str, optional): name of timestep column
        height_ratios_per_scenario (list, optional): relative row heights
        hspace (float, optional): vertical spacing
    """
    if ref_scenario not in scenario_list:
        raise ValueError("Referece scenario not in scenario list")

    df = helper.hourly_supply_demand(scenario_list)
    df = df.copy()
    df[timestep_col] = pd.to_datetime(df[timestep_col])

    if not all(tech in df.columns for tech in tech_list):
        raise ValueError("All or some of the techs selected are not present in the columns")

    df["Shift (Up)"] = df["HP Shift (Up)"] + df["EMob Shift (Up)"] + df["DSM (Up)"]
    df["Shift (Down)"] = df["HP Shift (Down)"] + df["EMob Shift (Down)"] + df["DSM (Down)"]

    # Map months to seasons
    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }
    df["season"] = df[timestep_col].dt.month.map(season_map)
    seasons = ["Winter", "Spring", "Summer", "Autumn"]

    scenarios = df[scenario_col].unique()
    scenario_list_no_ref = [s for s in scenario_list if s != ref_scenario]

    df["Tech"] = df[tech_list].sum(axis=1).mul(1000)  # MWh

    df_ref = df[df["scenario"] == ref_scenario]

    cols_to_merge = ["timestep", "Tech"]
    df_merge = df.merge(df_ref[cols_to_merge], on=["timestep"], how="left", suffixes=["", "_ref"])

    df_merge["Tech_diff"] = df_merge["Tech"] - df_merge["Tech_ref"]

    letters = list(string.ascii_lowercase)
    hours = np.arange(24)
    xticks = np.arange(0, 24, 6)
    xtick_labels = [f"{int(h):02d}:00" for h in xticks]

    # --------------------------
    # Plot one figure per technology
    # --------------------------
    for tech_label, abs_col, shift_up_col, shift_down_col in [
        (
            "",
            "Tech_diff",
            "Shift (Up)",
            "Shift (Down)",
        )
    ]:
        n = len(scenario_list_no_ref)
        n_rows = n * 3
        height_ratios = height_ratios_per_scenario * n

        fig = plt.figure(figsize=(13, 4 * n))
        gs = GridSpec(
            n_rows, 4, figure=fig, height_ratios=height_ratios, hspace=hspace, wspace=0.06
        )

        shift_axes = []
        global_shift_min = np.inf
        global_shift_max = -np.inf

        abs_axes = []
        global_abs_min = np.inf
        global_abs_max = -np.inf

        for i, scen in enumerate(scenario_list_no_ref):
            df_scen = df_merge[df_merge[scenario_col] == scen]
            title_row = i * 3
            abs_row = i * 3 + 1
            shift_row = i * 3 + 2

            # Title
            ax_title_left = fig.add_subplot(gs[title_row, 0])
            ax_title_left.axis("off")
            ax_title_left.text(
                0.01,
                -0.02,
                f"{letters[i]}. {format_scenario_name_inline(scen)}",
                ha="left",
                va="center",
                fontsize=12,
                transform=ax_title_left.transAxes,
            )
            ax_title_right = fig.add_subplot(gs[title_row, 1:])
            ax_title_right.axis("off")

            for j, season in enumerate(seasons):
                df_season = df_scen[df_scen["season"] == season]

                # ---------------- Row 1
                daily_s = []
                for _, day in df_season.groupby(df_season[timestep_col].dt.date):
                    max_val = day[abs_col].max()
                    if max_val == 0:
                        continue

                    s = (day[abs_col]).values
                    if len(s) == 24:
                        daily_s.append(s)

                S = np.array(daily_s)

                stats_s = get_stats(S)

                if i == 0:
                    ax_abs = fig.add_subplot(gs[abs_row, j])
                else:
                    ax_abs = fig.add_subplot(gs[abs_row, j])

                abs_axes.append(ax_abs)

                for data, color in [(stats_s, "orange")]:
                    ax_abs.fill_between(hours, data["p0"], data["p100"], color=color, alpha=0.15)
                    ax_abs.fill_between(hours, data["p10"], data["p90"], color=color, alpha=0.22)
                    ax_abs.fill_between(hours, data["p25"], data["p75"], color=color, alpha=0.35)
                    ax_abs.plot(hours, data["p50"], color=color, linewidth=2)

                global_abs_min = min(global_abs_min, stats_s["p0"].min())
                global_abs_max = max(global_abs_max, stats_s["p100"].max())

                ax_abs.set_xlim(0, 23)
                ax_abs.set_xticks(xticks)
                ax_abs.set_xticklabels(xtick_labels)
                if j == 0:
                    ax_abs.set_ylabel("Supply / Demand\ndifference\n[MWh]")
                    ax_abs.yaxis.set_label_coords(-0.2, 0.5)
                ax_abs.set_title(season, pad=5)
                ax_abs.grid(True, linestyle="--", alpha=0.4)
                ax_abs.set_axisbelow(True)
                # ax_abs.tick_params(axis="x", color="white", which="both")
                # ax_abs.tick_params(axis="y", color="white", which="both")
                ax_abs.set_facecolor("whitesmoke")
                ax_abs.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}"))
                ax_abs.axhline(0, color="black", linewidth=0.8)

                for spine in ax_abs.spines.values():
                    spine.set_visible(True)

                # spine adjustments
                if j == 0:  # first column
                    ax_abs.spines["left"].set_visible(True)
                    ax_abs.spines["right"].set_visible(False)
                elif j == 4 - 1:  # last column
                    ax_abs.spines["left"].set_visible(False)
                    ax_abs.spines["right"].set_visible(True)
                else:  # middle columns
                    ax_abs.spines["left"].set_visible(False)
                    ax_abs.spines["right"].set_visible(False)

                if j != 0:
                    # ax_abs.tick_params(axis="y", colors="white")  # tick lines
                    ax_abs.set_yticklabels([])
                    ax_abs.tick_params(axis="y", color="white", which="both")

                # ---------------- Row 2: shift
                daily_up, daily_down = [], []
                for _, day in df_season.groupby(df_season[timestep_col].dt.date):
                    u = day[shift_up_col].values
                    d = day[shift_down_col].values
                    if len(u) != 24 or len(d) != 24:
                        continue
                    daily_up.append(u)
                    daily_down.append(d)

                UP = np.array(daily_up) * -1
                DOWN = np.array(daily_down) * -1

                stats_up = get_percentiles(UP)
                stats_down = get_percentiles(DOWN)

                if i == 0:
                    ax_shift = fig.add_subplot(gs[shift_row, j])
                else:
                    ax_shift = fig.add_subplot(gs[shift_row, j])

                shift_axes.append(ax_shift)

                global_shift_min = min(
                    global_shift_min, stats_up["p0"].min(), stats_down["p0"].min()
                )
                global_shift_max = max(
                    global_shift_max, stats_up["p100"].max(), stats_down["p100"].max()
                )

                ax_shift.fill_between(
                    hours, stats_up["p0"], stats_up["p100"], color="purple", alpha=0.18
                )
                ax_shift.fill_between(
                    hours, stats_up["p10"], stats_up["p90"], color="purple", alpha=0.28
                )
                ax_shift.fill_between(
                    hours, stats_up["p25"], stats_up["p75"], color="purple", alpha=0.45
                )
                ax_shift.plot(hours, stats_up["p50"], color="purple", linewidth=1.5)

                ax_shift.fill_between(
                    hours, stats_down["p0"], stats_down["p100"], color="deeppink", alpha=0.18
                )
                ax_shift.fill_between(
                    hours, stats_down["p10"], stats_down["p90"], color="deeppink", alpha=0.28
                )
                ax_shift.fill_between(
                    hours, stats_down["p25"], stats_down["p75"], color="deeppink", alpha=0.45
                )
                ax_shift.plot(hours, stats_down["p50"], color="deeppink", linewidth=1.5)

                ax_shift.axhline(0, color="black", linewidth=0.8)
                ax_shift.set_xlim(0, 23)
                ax_shift.set_xticks(xticks)
                ax_shift.set_xticklabels(xtick_labels)
                ax_shift.grid(True, linestyle="--", alpha=0.4)
                ax_shift.set_axisbelow(True)
                # ax_shift.tick_params(axis="x", color="white", which="both")
                # ax_shift.tick_params(axis="y", color="white", which="both")
                ax_shift.set_facecolor("whitesmoke")
                ax_shift.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}"))
                if j == 0:
                    ax_shift.set_ylabel("Load shift\n[GWh]")
                    ax_shift.yaxis.set_label_coords(-0.2, 0.5)

                for spine in ax_shift.spines.values():
                    spine.set_visible(True)

                # spine adjustments
                if j == 0:  # first column
                    ax_shift.spines["left"].set_visible(True)
                    ax_shift.spines["right"].set_visible(False)
                elif j == 4 - 1:  # last column
                    ax_shift.spines["left"].set_visible(False)
                    ax_shift.spines["right"].set_visible(True)
                else:  # middle columns
                    ax_shift.spines["left"].set_visible(False)
                    ax_shift.spines["right"].set_visible(False)

                if j != 0:
                    # ax_shift.tick_params(axis="y", colors="white")  # tick lines
                    ax_shift.set_yticklabels([])
                    ax_shift.tick_params(axis="y", color="white", which="both")

        # Apply global limits
        for ax in shift_axes:
            ax.set_ylim(global_shift_min * 1.15, global_shift_max * 1.15)

        # Apply global limits
        for ax in abs_axes:
            if min_abs_y is not None:
                global_abs_min = min_abs_y

            ax.set_ylim(min(0, global_abs_min) * 1.05, global_abs_max * 1.15)

        # Legends
        handles_1 = [
            Line2D(
                [0], [0], color="orange", linewidth=2, label="Generation/Consumption difference"
            ),
            Line2D([0], [0], color="purple", linewidth=2, label="Shift Up"),
            Line2D([0], [0], color="deeppink", linewidth=2, label="Shift Down"),
        ]
        handles_2 = [
            Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="Median"),
            Patch(facecolor="dimgray", alpha=0.18, label="5–95%"),
            Patch(facecolor="dimgray", alpha=0.28, label="10–90%"),
            Patch(facecolor="dimgray", alpha=0.45, label="25–75%"),
        ]

        leg_2_pos = len(scenario_list) * 0.01
        leg_1_pos = 0.08

        legend1 = fig.legend(
            handles=handles_1,
            loc="upper left",
            bbox_to_anchor=(0.17, leg_1_pos),
            bbox_transform=fig.transFigure,
            ncol=4,
            frameon=False,
            title="Categories:",
        )
        legend2 = fig.legend(
            handles=handles_2,
            loc="upper left",
            bbox_to_anchor=(0.17, leg_2_pos),
            bbox_transform=fig.transFigure,
            ncol=4,
            frameon=False,
            title="Percentiles:",
        )
        legend1.get_title().set_ha("left")
        legend2.get_title().set_ha("left")
        legend1._legend_box.align = "left"
        legend2._legend_box.align = "left"

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = cnf.FIGURES_DIR / f"{filename_prefix}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()


def duals_seasonal_percentile(
    scenario_list,
    ref_scenario,
    filename_prefix,
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[1, 3, 2],
    hspace=0.5,
    min_abs_y=None,
):
    """Plot duals and hourly dispersion per season.

    Args:
        scenario_list: list of scenarios or DataFrame
        ref_scenario: reference scenario for duals
        filename_prefix: prefix for saved figures
        scenario_col (str, optional): name of scenario column
        timestep_col (str, optional): name of timestep column
        height_ratios_per_scenario (list, optional): relative row heights
        hspace (float, optional): vertical spacing
    """
    if ref_scenario not in scenario_list:
        raise ValueError("Referece scenario not in scenario list")

    duals = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "elec_price.csv").fillna(0)
    duals = duals[duals["scenario"].isin(scenario_list)]
    duals["timestep"] = pd.to_datetime(duals["timestep"])

    balance = helper.hourly_supply_demand(scenario_list)

    dual_balance = balance.merge(duals, on=["scenario", "timestep"])

    # Electricity cost
    dual_balance["elec_cost_EMob"] = dual_balance["value"] * dual_balance["Shifted e-Mobility"]
    dual_balance["elec_cost_HP"] = dual_balance["value"] * dual_balance["Shifted Heat Pump"]
    dual_balance["elec_cost_total"] = dual_balance["value"] * (
        dual_balance["Shifted e-Mobility"]
        + dual_balance["Shifted Heat Pump"]
        + dual_balance["Conventional"]
    )  # Million EUR

    dual_balance_ref = dual_balance[dual_balance["scenario"] == ref_scenario]

    cols_to_merge = ["scenario", "timestep", "value", "elec_cost_total"]
    dual_balance_savings = dual_balance[cols_to_merge].merge(
        dual_balance_ref[["timestep", "value", "elec_cost_total"]],
        on=["timestep"],
        how="left",
        suffixes=["", "_ref"],
    )

    dual_balance_savings["price_spread"] = (
        dual_balance_savings["value"] - dual_balance_savings["value_ref"]
    )
    dual_balance_savings["savings"] = (
        dual_balance_savings["elec_cost_total_ref"] - dual_balance_savings["elec_cost_total"]
    )

    # dual_balance_savings.groupby(["scenario"])[["elec_cost_total","savings"]].sum().reset_index()

    # df = helper.hourly_supply_demand(scenario_list)
    df = dual_balance_savings.copy()
    df[timestep_col] = pd.to_datetime(df[timestep_col])

    # Map months to seasons
    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }
    df["season"] = df[timestep_col].dt.month.map(season_map)
    seasons = ["Winter", "Spring", "Summer", "Autumn"]

    scenarios = df[scenario_col].unique()
    letters = list(string.ascii_lowercase)
    hours = np.arange(24)
    xticks = np.arange(0, 24, 6)
    xtick_labels = [f"{int(h):02d}:00" for h in xticks]

    scenario_list_no_ref = [s for s in scenario_list if s != ref_scenario]

    n = len(scenario_list_no_ref)
    n_rows = n * 3
    height_ratios = height_ratios_per_scenario * n

    fig = plt.figure(figsize=(13, 4 * n))
    gs = GridSpec(n_rows, 4, figure=fig, height_ratios=height_ratios, hspace=hspace, wspace=0.06)

    shift_axes = []
    global_shift_min = np.inf
    global_shift_max = -np.inf

    abs_axes = []
    global_abs_min = np.inf
    global_abs_max = -np.inf

    for i, scen in enumerate(scenario_list_no_ref):
        df_scen = df[df[scenario_col] == scen]
        title_row = i * 3
        abs_row = i * 3 + 1
        shift_row = i * 3 + 2

        # Title
        ax_title_left = fig.add_subplot(gs[title_row, 0])
        ax_title_left.axis("off")
        ax_title_left.text(
            0.01,
            -0.02,
            f"{letters[i]}. {format_scenario_name_inline(scen)}",
            ha="left",
            va="center",
            fontsize=12,
            transform=ax_title_left.transAxes,
        )
        ax_title_right = fig.add_subplot(gs[title_row, 1:])
        ax_title_right.axis("off")

        for j, season in enumerate(seasons):
            df_season = df_scen[df_scen["season"] == season]

            # ---------------- Row 1
            daily_s = []
            for _, day in df_season.groupby(df_season[timestep_col].dt.date):
                max_val = day["price_spread"].max()

                if max_val == 0:
                    continue
                s = (day["price_spread"]).values  # shifted
                if len(s) == 24:
                    daily_s.append(s)

            S = np.array(daily_s)
            stats_s = get_stats(S)

            if i == 0:
                ax_abs = fig.add_subplot(gs[abs_row, j])
            else:
                ax_abs = fig.add_subplot(gs[abs_row, j])

            abs_axes.append(ax_abs)

            for data, color in [(stats_s, "orange")]:
                ax_abs.fill_between(hours, data["p0"], data["p100"], color=color, alpha=0.15)
                ax_abs.fill_between(hours, data["p10"], data["p90"], color=color, alpha=0.22)
                ax_abs.fill_between(hours, data["p25"], data["p75"], color=color, alpha=0.35)
                ax_abs.plot(hours, data["p50"], color=color, linewidth=2)

            global_abs_min = min(global_abs_min, stats_s["p0"].min())
            global_abs_max = max(global_abs_max, stats_s["p100"].max())

            ax_abs.set_xlim(0, 23)
            ax_abs.set_xticks(xticks)
            ax_abs.set_xticklabels(xtick_labels)
            if j == 0:
                ax_abs.set_ylabel("Electricity price\ndifference\n[€ MWh$^{-1}$]")
                ax_abs.yaxis.set_label_coords(-0.2, 0.5)
            ax_abs.set_title(season, pad=5)
            ax_abs.grid(True, linestyle="--", alpha=0.4)
            ax_abs.set_axisbelow(True)
            # ax_abs.tick_params(axis="x", color="white", which="both")
            # ax_abs.tick_params(axis="y", color="white", which="both")
            ax_abs.set_facecolor("whitesmoke")
            ax_abs.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}"))
            ax_abs.axhline(0, color="black", linewidth=0.8)

            for spine in ax_abs.spines.values():
                spine.set_visible(True)

            # spine adjustments
            if j == 0:  # first column
                ax_abs.spines["left"].set_visible(True)
                ax_abs.spines["right"].set_visible(False)
            elif j == 4 - 1:  # last column
                ax_abs.spines["left"].set_visible(False)
                ax_abs.spines["right"].set_visible(True)
            else:  # middle columns
                ax_abs.spines["left"].set_visible(False)
                ax_abs.spines["right"].set_visible(False)

            if j != 0:
                # ax_abs.tick_params(axis="y", colors="white")  # tick lines
                ax_abs.set_yticklabels([])
                ax_abs.tick_params(axis="y", color="white", which="both")

            # ---------------- Row 2: shift
            daily_up = []
            for _, day in df_season.groupby(df_season[timestep_col].dt.date):
                u = day["savings"].values
                if len(u) != 24:
                    continue
                daily_up.append(u)

            UP = np.array(daily_up)

            stats_up = get_percentiles(UP)

            if i == 0:
                ax_shift = fig.add_subplot(gs[shift_row, j])
            else:
                ax_shift = fig.add_subplot(gs[shift_row, j])

            shift_axes.append(ax_shift)

            global_shift_min = min(global_shift_min, stats_up["p0"].min())
            global_shift_max = max(global_shift_max, stats_up["p100"].max())

            ax_shift.fill_between(
                hours, stats_up["p0"], stats_up["p100"], color="purple", alpha=0.18
            )
            ax_shift.fill_between(
                hours, stats_up["p10"], stats_up["p90"], color="purple", alpha=0.28
            )
            ax_shift.fill_between(
                hours, stats_up["p25"], stats_up["p75"], color="purple", alpha=0.45
            )
            ax_shift.plot(hours, stats_up["p50"], color="purple", linewidth=1.5)

            ax_shift.axhline(0, color="black", linewidth=0.8)
            ax_shift.set_xlim(0, 23)
            ax_shift.set_xticks(xticks)
            ax_shift.set_xticklabels(xtick_labels)
            ax_shift.grid(True, linestyle="--", alpha=0.4)
            ax_shift.set_axisbelow(True)
            # ax_shift.tick_params(axis="x", color="white", which="both")
            # ax_shift.tick_params(axis="y", color="white", which="both")
            ax_shift.set_facecolor("whitesmoke")
            ax_shift.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}"))
            if j == 0:
                ax_shift.set_ylabel("Electricity cost\nsavings\n[1000 €]")
                ax_shift.yaxis.set_label_coords(-0.2, 0.5)

            for spine in ax_shift.spines.values():
                spine.set_visible(True)

            # spine adjustments
            if j == 0:  # first column
                ax_shift.spines["left"].set_visible(True)
                ax_shift.spines["right"].set_visible(False)
            elif j == 4 - 1:  # last column
                ax_shift.spines["left"].set_visible(False)
                ax_shift.spines["right"].set_visible(True)
            else:  # middle columns
                ax_shift.spines["left"].set_visible(False)
                ax_shift.spines["right"].set_visible(False)

            if j != 0:
                # ax_shift.tick_params(axis="y", colors="white")  # tick lines
                ax_shift.set_yticklabels([])
                ax_shift.tick_params(axis="y", color="white", which="both")

    # Apply global limits
    for ax in shift_axes:
        ax.set_ylim(global_shift_min * 1.1, global_shift_max * 1.1)

    # Apply global limits
    for ax in abs_axes:
        if min_abs_y is not None:
            global_abs_min = min_abs_y

        ax.set_ylim(min(0, global_abs_min), global_abs_max * 1.05)

    # Legends
    handles_1 = [
        Line2D([0], [0], color="orange", linewidth=2, label="Electricity price difference"),
        Line2D([0], [0], color="purple", linewidth=2, label="Elelctricity cost savings"),
    ]
    handles_2 = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="Median"),
        Patch(facecolor="dimgray", alpha=0.18, label="5–95%"),
        Patch(facecolor="dimgray", alpha=0.28, label="10–90%"),
        Patch(facecolor="dimgray", alpha=0.45, label="25–75%"),
    ]

    leg_2_pos = len(scenario_list) * 0.01
    leg_1_pos = 0.08

    # legend1 = fig.legend(
    #     handles=handles_1,
    #     loc="upper left",
    #     bbox_to_anchor=(0.17, leg_1_pos),
    #     bbox_transform=fig.transFigure,
    #     ncol=4,
    #     frameon=False,
    #     title="",
    # )
    legend2 = fig.legend(
        handles=handles_2,
        loc="upper left",
        bbox_to_anchor=(0.17, leg_1_pos),
        bbox_transform=fig.transFigure,
        ncol=4,
        frameon=False,
        title="Percentiles:",
    )
    # legend1.get_title().set_ha("left")
    legend2.get_title().set_ha("left")
    # legend1._legend_box.align = "left"
    legend2._legend_box.align = "left"

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = cnf.FIGURES_DIR / f"duals_seasonal_{filename_prefix}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def cumulative_seasonal_savings(
    scenario_list,
    ref_scenario,
    filename_prefix,
    scenario_col="scenario",
    timestep_col="timestep",
    height_ratios_per_scenario=[1, 3, 2],
    hspace=0.5,
    min_abs_y=None,
):
    """Plot duals and hourly dispersion per season.

    Args:
        scenario_list: list of scenarios or DataFrame
        ref_scenario: reference scenario for duals
        filename_prefix: prefix for saved figures
        scenario_col (str, optional): name of scenario column
        timestep_col (str, optional): name of timestep column
        height_ratios_per_scenario (list, optional): relative row heights
        hspace (float, optional): vertical spacing
    """
    if ref_scenario not in scenario_list:
        raise ValueError("Referece scenario not in scenario list")

    duals = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "elec_price.csv").fillna(0)
    duals = duals[duals["scenario"].isin(scenario_list)]
    duals["timestep"] = pd.to_datetime(duals["timestep"])

    balance = helper.hourly_supply_demand(scenario_list)

    dual_balance = balance.merge(duals, on=["scenario", "timestep"])

    # Electricity cost
    dual_balance["elec_cost_EMob"] = dual_balance["value"] * dual_balance["Shifted e-Mobility"]
    dual_balance["elec_cost_HP"] = dual_balance["value"] * dual_balance["Shifted Heat Pump"]
    dual_balance["elec_cost_total"] = dual_balance["value"] * (
        dual_balance["Shifted e-Mobility"]
        + dual_balance["Shifted Heat Pump"]
        + dual_balance["Conventional"]
    )  # Million EUR ?

    dual_balance_ref = dual_balance[dual_balance["scenario"] == ref_scenario]

    cols_to_merge = ["scenario", "timestep", "value", "elec_cost_total"]
    dual_balance_savings = dual_balance[cols_to_merge].merge(
        dual_balance_ref[["timestep", "value", "elec_cost_total"]],
        on=["timestep"],
        how="left",
        suffixes=["", "_ref"],
    )

    dual_balance_savings["price_spread"] = (
        dual_balance_savings["value_ref"] - dual_balance_savings["value"]
    )
    dual_balance_savings["savings"] = (
        dual_balance_savings["elec_cost_total_ref"] - dual_balance_savings["elec_cost_total"]
    ).div(1000)

    # dual_balance_savings.groupby(["scenario"])[["elec_cost_total","savings"]].sum().reset_index()

    # df = helper.hourly_supply_demand(scenario_list)
    df = dual_balance_savings.copy()
    df[timestep_col] = pd.to_datetime(df[timestep_col])

    # Map months to seasons
    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }
    df["season"] = df[timestep_col].dt.month.map(season_map)


    df["hour"] = df[timestep_col].dt.strftime("%H:%M")

    tot_savings = (
        df.groupby(["scenario", "season", "hour"])[
            ["elec_cost_total", "elec_cost_total_ref", "savings"]
        ]
        .sum()
        .reset_index()
    )

    # 🔹 IMPORTANT: sort by hour before cumsum
    tot_savings = tot_savings.sort_values(["scenario", "season", "hour"])

    # cumulative sum over the day (within scenario & season)
    tot_savings["cumulative_savings"] = (
        tot_savings.groupby(["scenario", "season"])["savings"]
        .cumsum()
    )

    seasons = ["Winter", "Spring", "Summer", "Autumn"]

    scenarios = df[scenario_col].unique()
    letters = list(string.ascii_lowercase)
    hours = np.arange(24)
    xticks = np.arange(0, 24, 6)
    xtick_labels = [f"{int(h):02d}:00" for h in xticks]

    scenario_list_no_ref = [s for s in scenario_list if s != ref_scenario]

    n = len(scenario_list_no_ref)
    n_rows = n * 3
    height_ratios = height_ratios_per_scenario * n

    fig = plt.figure(figsize=(13, 4 * n))
    gs = GridSpec(n_rows, 4, figure=fig, height_ratios=height_ratios, hspace=hspace, wspace=0.06)

    shift_axes = []
    global_shift_min = np.inf
    global_shift_max = -np.inf

    abs_axes = []
    global_abs_min = np.inf
    global_abs_max = -np.inf

    for i, scen in enumerate(scenario_list_no_ref):
        df_scen = tot_savings[tot_savings[scenario_col] == scen]
        title_row = i * 3
        abs_row = i * 3 + 1
        shift_row = i * 3 + 2

        # Title
        ax_title_left = fig.add_subplot(gs[title_row, 0])
        ax_title_left.axis("off")
        ax_title_left.text(
            0.01,
            -0.02,
            f"{letters[i]}. {format_scenario_name_inline(scen)}",
            ha="left",
            va="center",
            fontsize=12,
            transform=ax_title_left.transAxes,
        )
        ax_title_right = fig.add_subplot(gs[title_row, 1:])
        ax_title_right.axis("off")

        for j, season in enumerate(seasons):
            df_season = df_scen[df_scen["season"] == season]

            # ---------------- Row 1
            if i == 0:
                ax_abs = fig.add_subplot(gs[abs_row, j])
            else:
                ax_abs = fig.add_subplot(gs[abs_row, j])

            abs_axes.append(ax_abs)

            ax_abs.plot(df_season["hour"], df_season["savings"], color="orange", linewidth=2)

            global_abs_min = min(global_abs_min, df_season["savings"].min())
            global_abs_max = max(global_abs_max, df_season["savings"].max())

            ax_abs.set_xlim(0, 23)
            ax_abs.set_xticks(xticks)
            ax_abs.set_xticklabels(xtick_labels)
            if j == 0:
                ax_abs.set_ylabel("Savings\n[Million €]")
                ax_abs.yaxis.set_label_coords(-0.2, 0.5)
            ax_abs.set_title(season, pad=5)
            ax_abs.grid(True, linestyle="--", alpha=0.4)
            ax_abs.set_axisbelow(True)
            # ax_abs.tick_params(axis="x", color="white", which="both")
            # ax_abs.tick_params(axis="y", color="white", which="both")
            ax_abs.set_facecolor("whitesmoke")
            ax_abs.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}"))
            ax_abs.axhline(0, color="black", linewidth=0.8)

            for spine in ax_abs.spines.values():
                spine.set_visible(True)

            # spine adjustments
            if j == 0:  # first column
                ax_abs.spines["left"].set_visible(True)
                ax_abs.spines["right"].set_visible(False)
            elif j == 4 - 1:  # last column
                ax_abs.spines["left"].set_visible(False)
                ax_abs.spines["right"].set_visible(True)
            else:  # middle columns
                ax_abs.spines["left"].set_visible(False)
                ax_abs.spines["right"].set_visible(False)

            if j != 0:
                # ax_abs.tick_params(axis="y", colors="white")  # tick lines
                ax_abs.set_yticklabels([])
                ax_abs.tick_params(axis="y", color="white", which="both")

            # ---------------- Row 2: shift


            if i == 0:
                ax_shift = fig.add_subplot(gs[shift_row, j])
            else:
                ax_shift = fig.add_subplot(gs[shift_row, j])

            shift_axes.append(ax_shift)

            global_shift_min = min(global_shift_min, df_season["cumulative_savings"].min())
            global_shift_max = max(global_shift_max, df_season["cumulative_savings"].max())

            ax_shift.plot(df_season["hour"], df_season["cumulative_savings"], color="purple", linewidth=1.5)

            ax_shift.axhline(0, color="black", linewidth=0.8)
            ax_shift.set_xlim(0, 23)
            ax_shift.set_xticks(xticks)
            ax_shift.set_xticklabels(xtick_labels)
            ax_shift.grid(True, linestyle="--", alpha=0.4)
            ax_shift.set_axisbelow(True)
            # ax_shift.tick_params(axis="x", color="white", which="both")
            # ax_shift.tick_params(axis="y", color="white", which="both")
            ax_shift.set_facecolor("whitesmoke")
            ax_shift.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}"))
            if j == 0:
                ax_shift.set_ylabel("Cumulative\nsavings\n[Million €]")
                ax_shift.yaxis.set_label_coords(-0.2, 0.5)

            for spine in ax_shift.spines.values():
                spine.set_visible(True)

            # spine adjustments
            if j == 0:  # first column
                ax_shift.spines["left"].set_visible(True)
                ax_shift.spines["right"].set_visible(False)
            elif j == 4 - 1:  # last column
                ax_shift.spines["left"].set_visible(False)
                ax_shift.spines["right"].set_visible(True)
            else:  # middle columns
                ax_shift.spines["left"].set_visible(False)
                ax_shift.spines["right"].set_visible(False)

            if j != 0:
                # ax_shift.tick_params(axis="y", colors="white")  # tick lines
                ax_shift.set_yticklabels([])
                ax_shift.tick_params(axis="y", color="white", which="both")

    # Apply global limits
    for ax in shift_axes:
        ax.set_ylim(global_shift_min * 1.1, global_shift_max * 1.1)

    # Apply global limits
    for ax in abs_axes:
        if min_abs_y is not None:
            global_abs_min = min_abs_y

        ax.set_ylim(min(0, global_abs_min), global_abs_max * 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = cnf.FIGURES_DIR / f"cumulative_savings_{filename_prefix}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

def electricity_savings_seasonal(scenario_list, ref_scenario, filename):
    """Plot total savings from electricity price change."""
    # -----------------------------
    # Load data
    # -----------------------------
    if ref_scenario not in scenario_list:
        raise ValueError("Referece scenario not in scenario list")

    duals = pd.read_csv(cnf.RESULTS_NEXUS_DIR / "elec_price.csv").fillna(0)
    duals = duals[duals["scenario"].isin(scenario_list)]
    duals["timestep"] = pd.to_datetime(duals["timestep"])

    balance = helper.hourly_supply_demand(scenario_list)
    dual_balance = balance.merge(duals, on=["scenario", "timestep"])

    # -----------------------------
    # Add season
    # -----------------------------
    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }
    dual_balance["season"] = dual_balance["timestep"].dt.month.map(season_map)
    season_order = ["Winter", "Spring", "Summer", "Autumn"]

    season_colors = {
        "Winter": "#53436B",
        "Spring": "#8CB369",
        "Summer": "#BC4B51",
        "Autumn": "#F4E285",
    }

    tech_groups = [
        ["Shifted e-Mobility", "Shifted Heat Pump", "Conventional"],
        ["Shifted e-Mobility"],
        ["Shifted Heat Pump"],
    ]

    results = []
    total_savings_all = []
    total_share_all = []

    # -----------------------------
    # Precompute results
    # -----------------------------
    for tech_list in tech_groups:
        df = dual_balance.copy()
        df["elec_cost_total"] = (df["value"] * df[tech_list].sum(axis=1)) / 1000
        df_ref = df[df["scenario"] == ref_scenario]

        df_savings = df.merge(
            df_ref[["timestep", "value", "elec_cost_total"]],
            on="timestep",
            how="left",
            suffixes=("", "_ref"),
        )
        df_savings["savings"] = df_savings["elec_cost_total_ref"] - df_savings["elec_cost_total"]

        # Seasonal aggregation
        df_cost = (
            df_savings.groupby(["scenario", "season"])[["elec_cost_total", "savings"]]
            .sum()
            .reset_index()
        )
        df_cost = df_cost[df_cost["scenario"] != ref_scenario]

        # Annual total savings per scenario
        total_savings = df_cost.groupby("scenario")["savings"].sum()
        total_savings_all.extend(total_savings.values)

        # Annual savings share
        df_share = total_savings.to_frame().merge(
            df_cost.groupby("scenario")["elec_cost_total"].sum().to_frame(),
            left_index=True,
            right_index=True,
        )
        df_share["savings_share"] = df_share["savings"] / df_share["elec_cost_total"]
        total_share_all.extend(df_share["savings_share"].values)

        results.append((tech_list, df_cost, df_share))

    # -----------------------------
    # Compute global x limits (total savings) and zero alignment
    # -----------------------------
    x_min = min(min(total_savings_all) * 2.5, -1 * max(total_savings_all)/10) if min(total_savings_all) < 0 else 0
    x_max = max(total_savings_all) * 1.1 if max(total_savings_all) > 0 else 0

    # Global secondary axis limits
    share_min = min(min(total_share_all) * 2.5, -1 * max(total_share_all)/10) if min(total_share_all) < 0 else 0
    share_max = max(total_share_all) * 1.1 if max(total_share_all) > 0 else 0

    # Compute zero alignment factor for secondary axis
    zero_pos = abs(x_min) / (x_max - x_min) if x_max - x_min != 0 else 0.5
    share_range = share_max - share_min
    share_shift = zero_pos * share_range - abs(share_min)

    share_min_aligned = share_min - share_shift
    share_max_aligned = share_max - share_shift

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)

    for ax, (tech_list, df_cost, df_share) in zip(axes, results):
        pivot = (
            df_cost.pivot(index="scenario", columns="season", values="savings")
            .reindex(columns=season_order)
            .fillna(0)
        )
        scenarios = pivot.index
        y_pos = np.arange(len(scenarios))

        pos_left = np.zeros(len(pivot))
        neg_left = np.zeros(len(pivot))

        for season in season_order:
            values = pivot[season].values
            pos_values = np.where(values > 0, values, 0)
            neg_values = np.where(values < 0, values, 0)

            ax.barh(y_pos, pos_values, left=pos_left, color=season_colors[season], label=season)
            pos_left += pos_values

            ax.barh(y_pos, neg_values, left=neg_left, color=season_colors[season])
            neg_left += neg_values

        ax.set_yticks(y_pos)
        ax.set_yticklabels([format_scenario_name(s) for s in scenarios])
        ax.invert_yaxis()
        ax.set_xlim(x_min, x_max)
        ax.axvline(0, linestyle="-", linewidth=1, color= "black")
        ax.set_xlabel("Savings [Million €]")
        tech_list_clean = [t.replace("Shifted ", "") for t in tech_list]
        ax.set_title(" + ".join(tech_list_clean), pad=10)

        # Secondary axis (aligned)
        ax2 = ax.twiny()
        share_vals = df_share.loc[scenarios]["savings_share"]
        ax2.scatter(
            share_vals.values, y_pos, marker="D", facecolors="white", edgecolors="black", zorder=5
        )
        ax2.set_xlim(share_min_aligned, share_max_aligned)
        ax2.axvline(0, linestyle="-", linewidth=1, color= "black")
        ax2.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax2.set_xlabel("Savings share [-]", labelpad=10)

    leg = axes[2].legend(title="Season", loc="lower right")
    leg.get_title().set_ha("left")
    leg._legend_box.align = "left"
    plt.tight_layout()
    fig_path = cnf.FIGURES_DIR / f"savings_{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def emobility_duration_shift(scenario_list, filename):

    crossing_days_df_sum = helper.compute_emobility_flexibility_shift(scenario_list)

    crossing_days_df_sum['days_between_zero_crossing'] = (
        crossing_days_df_sum['days_between_zero_crossing'].astype(int)
    )

    scenarios = crossing_days_df_sum['scenario'].unique()
    n_scen = len(scenarios)

    max_day = crossing_days_df_sum['days_between_zero_crossing'].max()
    title_letters = iter(string.ascii_lowercase)

    fig, axes = plt.subplots(
        n_scen,
        2,
        figsize=(12, 3*n_scen),
        sharey='col'
    )

    if n_scen == 1:
        axes = np.array(axes).reshape(2, 1)

    for i, scenario in enumerate(scenarios):

        df_s = crossing_days_df_sum[
            crossing_days_df_sum['scenario'] == scenario
        ].copy()

        min_day = 1
        # max_day = df_s['days_between_zero_crossing'].max()
        x_ticks = np.arange(min_day, max_day + 1)

        # ==========================
        # Row 1: Histogram
        # ==========================
        ax_hist = axes[i, 0]

        bins = np.arange(min_day - 0.5, max_day + 1.5, 1)

        ax_hist.hist(
            df_s['days_between_zero_crossing'],
            bins=bins,
            color='tab:blue',
            edgecolor='black',
            alpha=0.7
        )

        ax_hist.set_xlim(min_day - 0.5, max_day + 0.5)
        ax_hist.set_xticks(x_ticks)
        letter = next(title_letters)
        ax_hist.set_title(f"{letter}. {format_scenario_name_inline(scenario)}", fontsize=12, x=0, ha="left", pad=15)
        ax_hist.set_xlabel("e-Mobility shift duration [Days]", fontsize=12)
        ax_hist.set_ylabel("Number of occurrences\n[-]", fontsize=12)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.set_axisbelow(True)

        # ==========================
        # Row 2: Boxplot
        # ==========================
        ax_box = axes[i, 1]

        sns.boxplot(
            x='days_between_zero_crossing',
            y='sum_shift_up_before_crossing',
            data=df_s,
            order=x_ticks,  # <-- THIS is the key line
            ax=ax_box,
            color='orange'
        )

        ax_box.set_xlim(-0.5, len(x_ticks) - 0.5)
        ax_box.set_xticks(range(len(x_ticks)))
        ax_box.set_xticklabels(x_ticks)

        # letter = next(title_letters)
        # ax_box.set_title(f"{letter}. {format_scenario_name_inline(scenario)}", fontsize=12, x=0, ha="left")
        ax_box.set_xlabel("e-Mobility shift duration [Days]", fontsize=12)
        ax_box.set_ylabel("Cumulative e-Mobility\nShift [GWh]", fontsize=12)
        ax_box.grid(True, alpha=0.3)
        ax_box.set_axisbelow(True)

    fig.subplots_adjust(wspace=0.4, hspace=0.8)
    # plt.tight_layout()
    fig_path = cnf.FIGURES_DIR / f"eMobility_duration_{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


def shift_capacity_boxplots(scenario_list, filename):
    """Plot shift up and down capacities."""
    df_hourly = helper.hourly_supply_demand(scenario_list)
    df = df_hourly[
        ["scenario", "EMob Shift (Up)", "EMob Shift (Down)", "HP Shift (Up)", "HP Shift (Down)"]
    ]

    # Apply formatting
    df[df.select_dtypes(include="number").columns] *= -1
    df["scenario_formatted"] = df["scenario"].apply(format_scenario_name)

    scenarios = sorted(df["scenario_formatted"].unique())

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8, 6),
        sharex=True,
        sharey=True
    )

    # Symmetric limits
    all_shift_cols = [
        "EMob Shift (Up)", "EMob Shift (Down)",
        "HP Shift (Up)",   "HP Shift (Down)"
    ]

    xmin = df[all_shift_cols].min().min()
    xmax = df[all_shift_cols].max().max()
    limit = max(abs(xmin), abs(xmax))

    # ===============================
    # Row 1: EMob
    # ===============================
    ax1 = axes[0]

    sns.boxplot(
        y="scenario_formatted",
        x="EMob Shift (Up)",
        data=df,
        order=scenarios,
        ax=ax1,
        color="#f2c200",
        width=0.35
    )

    sns.boxplot(
        y="scenario_formatted",
        x="EMob Shift (Down)",
        data=df,
        order=scenarios,
        ax=ax1,
        color="#7a3db8",
        width=0.35
    )

    ax1.set_xlim(-limit, limit)
    ax1.set_title("a. E-Mobility Flexibility", x=0, ha="left")
    ax1.set_xlabel("Shift Up (+) / Down (-) [GWh/h]", fontsize=12)
    ax1.set_ylabel("")
    ax1.tick_params(axis='x', labelbottom=True, labelsize=12)
    ax1.grid(True, alpha=0.3)

    # ===============================
    # Row 2: HP
    # ===============================
    ax2 = axes[1]

    sns.boxplot(
        y="scenario_formatted",
        x="HP Shift (Up)",
        data=df,
        order=scenarios,
        ax=ax2,
        color="#f2c200",
        width=0.35
    )

    sns.boxplot(
        y="scenario_formatted",
        x="HP Shift (Down)",
        data=df,
        order=scenarios,
        ax=ax2,
        color="#7a3db8",
        width=0.35
    )

    ax2.set_xlim(-limit, limit)
    ax2.set_title("b. Heat Pump Flexibility", x=0, ha="left")
    ax2.set_xlabel("Shift Up (+) / Down (-) [GWh/h]", fontsize = 12)
    ax2.set_ylabel("")
    ax2.grid(True, alpha=0.3)

    fig.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    fig_path = cnf.FIGURES_DIR / f"shift_capacity_{filename}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
