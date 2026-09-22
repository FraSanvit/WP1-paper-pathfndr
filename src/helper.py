"""Helper fucntions."""

# %%
import json
from functools import partial
from pathlib import Path

import pandas as pd
import numpy as np

import config as cnf


def process_json_file(file_path: Path):
    """
    Reads a JSON file and returns its content.

    Extend this function later with your processing logic.
    """
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def process_file(file_path: Path):
    """
    Placeholder for file-level processing logic.

    Add your processing code here later.
    """
    pass


def extract_responses(data):
    """
    Extract all Dash callback response payloads from a long HAR-style JSON.

    Returns a dict where top-level Dash response keys
    (e.g. 'page_content', 'progress', 'image', 'version')
    are directly accessible.
    """
    responses = []

    # HAR-style structure
    entries = data.get("log", {}).get("entries", [])

    for entry in entries:
        content = entry.get("response", {}).get("content", {})
        text = content.get("text")

        if not text:
            continue

        try:
            payload = json.loads(text)
        except Exception:
            continue

        # Dash callback responses contain this key
        if "response" in payload:
            responses.append(payload)

    return merge_dash_responses(responses)


def merge_dash_responses(responses):
    """Merge a list of Dash callback responses into a single accessible dict."""
    merged = {}

    for item in responses:
        response = item.get("response", {})
        for key, value in response.items():
            merged[key] = value

    return merged


def extract_plotly_elec_price(dash_data):
    """Extract hourly electricity prices."""
    series = {}

    for response_value in dash_data.values():
        if not isinstance(response_value, dict):
            continue

        children = response_value.get("children", [])
        if not isinstance(children, list):
            continue

        for child in children:
            props = child.get("props", {})
            figure = props.get("figure")

            if not figure:
                continue

            for trace in figure.get("data", []):
                x = trace.get("x")
                y = trace.get("y")

                if x is None or y is None:
                    continue

                series["value"] = pd.Series(y, index=x)

    if not series:
        return pd.DataFrame()

    df = pd.concat(series, axis=1)
    df.index.name = "timestep"
    df = df.reset_index()

    df["timestep"] = pd.to_datetime(df["timestep"], format="%Y-%m-%dT%H:%M:%S")

    return df


def extract_plotly_timeseries(dash_data):
    """Extract hourly timeseries."""
    series = {}

    for response_value in dash_data.values():
        if not isinstance(response_value, dict):
            continue

        children = response_value.get("children", [])
        if not isinstance(children, list):
            continue

        for child in children:
            props = child.get("props", {})
            figure = props.get("figure")

            if not figure:
                continue

            for trace in figure.get("data", []):
                meta = trace.get("meta")
                x = trace.get("x")
                y = trace.get("y")

                if meta is None or x is None or y is None:
                    continue

                series[meta] = pd.Series(y, index=x)

    if not series:
        return pd.DataFrame()

    df = pd.concat(series, axis=1)
    df.index.name = "timestep"
    df = df.reset_index()

    df["timestep"] = pd.to_datetime(df["timestep"], format="%Y-%m-%dT%H:%M:%S")

    return df


def find_figures(node):
    """Recursively search a nested dict/list for all Plotly figures."""
    figures = []

    if isinstance(node, dict):
        if "figure" in node:
            figures.append(node["figure"])
        for value in node.values():
            figures.extend(find_figures(value))

    elif isinstance(node, list):
        for item in node:
            figures.extend(find_figures(item))

    return figures


def extract_tech_values(layout_dict, var_column="tech", value_column="value"):
    """Extract all 'tech' and 'value' pairs from any figure in the layout."""
    figures = find_figures(layout_dict)

    all_rows = []
    for fig in figures:
        for trace in fig.get("data", []):
            y_val = trace.get("y", [None])

            if not y_val:
                continue

            all_rows.append(
                {
                    var_column: trace.get("meta", trace.get("name")),
                    value_column: y_val[0],
                }
            )

    return pd.DataFrame(all_rows)


def get_processor(filename: str):
    """Assign the function and the arguments to each variable."""
    PROCESSORS = {
        "_h": extract_plotly_timeseries,
        "cap": partial(
            extract_tech_values,
            var_column="tech",
            value_column="capacity",
        ),
        "elec_price": extract_plotly_elec_price,
        "system_cost_cost": partial(
            extract_tech_values,
            var_column="cost_type",
            value_column="cost",
        ),
        "system_cost_gen": partial(
            extract_tech_values,
            var_column="tech",
            value_column="cost",
        ),
    }

    for pattern, func in PROCESSORS.items():
        if pattern in filename:
            return func

    return None


def split_tech_and_country(df):
    """
    Split import/export.

    - Remove rows where tech contains '(' or ')'
    - Split tech into tech and country on ' to ' or ' from '
    - Keep 'to/from' only for non Import/Export techs (e.g. Transit)
    - Remove leading/trailing spaces from tech and country
    """
    df = df.copy()
    tech_remove_to_from = ["Imports", "Exports"]

    # 1. Remove rows with brackets
    df = df[~df["tech"].str.contains(r"[()]", regex=True)]

    # 2. Initialize country column
    df["country"] = pd.NA

    # --- handle 'to' ---
    mask_to = df["tech"].str.contains(" to ")
    if mask_to.any():
        left_to, right_to = df.loc[mask_to, "tech"].str.split(" to ", expand=True).T.values

        # country always right-hand side
        df.loc[mask_to, "country"] = right_to

        # tech handling
        df.loc[mask_to, "tech"] = [
            f"{l} to" if l not in tech_remove_to_from else l for l in left_to
        ]

    # --- handle 'from' ---
    mask_from = df["tech"].str.contains(" from ")
    if mask_from.any():
        left_from, right_from = df.loc[mask_from, "tech"].str.split(" from ", expand=True).T.values

        df.loc[mask_from, "country"] = right_from

        df.loc[mask_from, "tech"] = [
            f"{l} from" if l not in tech_remove_to_from else l for l in left_from
        ]

    # --- remove leading/trailing spaces from tech and country ---
    df["tech"] = df["tech"].str.strip()
    df["country"] = df["country"].astype(str).str.strip().replace("nan", pd.NA)

    return df

def cumulative_shifted_demand(scenario_list):
    """Compute the cumulative shift per day."""
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

    df = hourly_df.copy()
    # Example: your DataFrame is called df
    # First, make sure 'timestep' is a datetime
    df['timestep'] = pd.to_datetime(df['timestep'])

    # Extract the date (without time) for grouping
    df['date'] = df['timestep'].dt.date

    # Columns you want to cumulate
    cols_to_cumsum = ['e-Mobility', 'Heat Pump', 'Shifted e-Mobility', 'Shifted Heat Pump']

    # Compute daily cumulative sum
    df[cols_to_cumsum] = (
        df
        .groupby(['scenario', 'date'])[cols_to_cumsum]
        .cumsum()
    )
    # Optional: drop the helper 'date' column if you don't need it
    df.drop(columns='date', inplace=True)
    return df

def hourly_supply_demand(scenario_list):
    """Merge hourly supply and demand data for given scenarios."""
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

    df = hourly_df.copy()
    # Example: your DataFrame is called df
    # First, make sure 'timestep' is a datetime
    df['timestep'] = pd.to_datetime(df['timestep'])

    # Extract the date (without time) for grouping
    df['date'] = df['timestep'].dt.date

    # Optional: drop the helper 'date' column if you don't need it
    df.drop(columns='date', inplace=True)
    return df

def compute_emobility_flexibility_shift(scenario_list, timestep_col = "timestep"):
    """Compute the cumulative daily net shift of flexibility."""
    df = hourly_supply_demand(scenario_list)
    df[timestep_col] = pd.to_datetime(df[timestep_col])

    # Add a 'date' column for daily aggregation
    df['date'] = df[timestep_col].dt.date

    day_flex = df.groupby(["scenario", "date"])[["EMob Shift (Down)", 'EMob Shift (Up)']].sum().mul(-1).reset_index()
    day_flex["Net EMob Shift"] = day_flex["EMob Shift (Down)"] + day_flex['EMob Shift (Up)']

    daily_ev = df.groupby(['scenario','date'])[['e-Mobility','Shifted e-Mobility']].sum().reset_index()

    # Merge with day_flex
    day_flex = day_flex.merge(daily_ev, on=['scenario','date'], how='left')

    # Sort by scenario and date to ensure cumulative sums are correct
    day_flex = day_flex.sort_values(['scenario','date'])

    # Compute cumulative sums per scenario
    day_flex['cumulative_ev'] = day_flex.groupby('scenario')['e-Mobility'].cumsum()
    day_flex['cumulative_ev_shifted'] = day_flex.groupby('scenario')['Shifted e-Mobility'].cumsum()
    day_flex['cumulative_net_shift'] = day_flex.groupby('scenario')['Net EMob Shift'].cumsum()

    # Rename 'date' to 'timestep' if you want
    day_flex = day_flex.rename(columns={'date':'timestep'})

    crossing_days_list = []

    # Loop over scenarios
    for scenario, df_s in day_flex.groupby('scenario'):
        # Extract cumulative net shift
        cum_shift = df_s['cumulative_net_shift'].values

        # Compute the sign (-1, 0, 1) of each value
        signs = np.sign(cum_shift)

        # Find indices where sign changes (ignoring zeros)
        sign_change = np.diff(signs)
        crossing_indices = np.where(sign_change != 0)[0] + 1  # +1 because diff shifts by 1

        # Compute days between crossings
        if len(crossing_indices) > 0:
            # Include first segment from day 0 to first crossing
            days_between = [crossing_indices[0]]  # first segment
            # Then differences between consecutive crossings
            days_between += list(np.diff(crossing_indices))
            # Round up
            days_between = [int(np.ceil(d)) for d in days_between]
        else:
            # No crossing
            days_between = []

        # Store in DataFrame
        df_cross = pd.DataFrame({
            'scenario': scenario,
            'days_between_zero_crossing': days_between
        })

        crossing_days_list.append(df_cross)

    # Combine all scenarios
    crossing_days_df = pd.concat(crossing_days_list, ignore_index=True)

    sum_shift_up_list = []

    for scenario, df_s in day_flex.groupby('scenario'):
        daily_shift_up = df_s['EMob Shift (Up)'].values

        # Select only rows of this scenario
        df_cross_s = crossing_days_df[crossing_days_df['scenario'] == scenario].copy()
        df_cross_s = df_cross_s.reset_index(drop=True)

        # Use days_between_zero_crossing to split the daily_shift_up array
        sums = []
        start_idx = 0
        for segment_days in df_cross_s['days_between_zero_crossing']:
            end_idx = start_idx + int(segment_days)  # ensure integer
            segment_sum = daily_shift_up[start_idx:end_idx].sum()
            sums.append(segment_sum)
            start_idx = end_idx

        # Now the lengths match
        df_cross_s['sum_shift_up_before_crossing'] = sums
        sum_shift_up_list.append(df_cross_s)

    # Combine all scenarios
    crossing_days_df_sum = pd.concat(sum_shift_up_list, ignore_index=True)

    return crossing_days_df_sum

def fill_dataframe(cap_df, ext_col, add_col, value_col):
    """Fill in with missing values."""
    all_techs = cap_df[add_col].unique()
    all_scenarios = cap_df[ext_col].unique()

    # 2️⃣ Create a "full index" of all tech × scenario combinations
    full_index = pd.MultiIndex.from_product([all_scenarios,all_techs], names=[ext_col, add_col])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # 3️⃣ Merge with your existing cap_df, filling missing capacity with 0
    cap_df = full_df.merge(cap_df, on=[ext_col,add_col], how="left")
    cap_df[value_col] = cap_df[value_col].fillna(0)
    return cap_df
