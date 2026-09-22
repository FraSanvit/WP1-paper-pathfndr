"""Post-processing results."""

import re

import numpy as np
import pandas as pd

import config as cnf
import helper as help


def create_results(base_path):
    """Create the csv files from json files."""
    nexus_data: dict[str, dict] = {}

    if not base_path.exists():
        raise FileNotFoundError(f"Path does not exist: {base_path}")

    for folder in base_path.iterdir():
        if folder.is_dir():
            print(f"Processing folder: {folder.name}")
            nexus_data[folder.name] = {}

            for json_file in folder.iterdir():
                if json_file.is_file() and json_file.suffix == ".json":
                    print(f"  Reading JSON file: {json_file.name}")
                    data = help.process_json_file(json_file)
                    responses = help.extract_responses(data)

                    processor = help.get_processor(json_file.stem)

                    if processor is None:
                        print(f"    No processor defined for {json_file.name}")
                        continue

                    df = processor(responses)
                    df.insert(0, "scenario", folder.name)
                    try:
                        df.insert(1, "unit", responses["unit"])
                    except KeyError:
                        print(f"    'unit' not found in responses for {json_file.name}")
                    nexus_data[folder.name][json_file.stem] = df

    combined: dict[str, list] = {}

    for scenario, files in nexus_data.items():
        for file_type, df in files.items():
            combined.setdefault(file_type, []).append(df)

    combined = {file_type: pd.concat(dfs, ignore_index=True) for file_type, dfs in combined.items()}

    output_dir = cnf.RESULTS_NEXUS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_type, df in combined.items():
        output_file = output_dir / f"{file_type}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

    level_df, cap_df = compute_storage_level_and_capacity()

    level_df.to_csv(cnf.RESULTS_NEXUS_DIR / "storage_level_h.csv", index=False)
    cap_df.to_csv(cnf.RESULTS_NEXUS_DIR / "storage_capacity.csv", index=False)


def compute_storage_level_and_capacity():
    base_path = cnf.RESULTS_NEXUS_DIR
    gen_file = base_path / "generation_h.csv"

    if not gen_file.exists():
        raise FileNotFoundError("generation_h.csv not found")

    df = pd.read_csv(gen_file)
    df["timestep"] = pd.to_datetime(df["timestep"])

    # Identify Gen / Load columns
    gen_cols = [c for c in df.columns if "(Gen)" in c]
    load_cols = [c for c in df.columns if "(Load)" in c]

    def clean_tech(col):
        return re.sub(r"\s*\((Gen|Load)\)\s*", "", col).strip()

    gen_map = {clean_tech(c): c for c in gen_cols}
    load_map = {clean_tech(c): c for c in load_cols}

    storage_techs = sorted(set(gen_map) & set(load_map))

    level_rows = []
    cap_rows = []

    for scenario, sdf in df.groupby("scenario"):
        sdf = sdf.set_index("timestep")

        for tech in storage_techs:
            gen = sdf[gen_map[tech]].fillna(0)
            load = sdf[load_map[tech]].fillna(0)

            # net flow per timestep
            net = gen + load

            # cumulative energy level
            level = net.cumsum()

            # 🔹 level in long format
            level_rows.append(
                level.rename("value").reset_index().assign(scenario=scenario, tech=tech, unit="GWh")
            )

            # 🔹 capacity in long format
            cap_rows.append(
                {"scenario": scenario, "tech": tech, "unit": "GWh", "value": level.abs().max()}
            )

    level_df = pd.concat(level_rows, ignore_index=True)
    cap_df = pd.DataFrame(cap_rows)

    # reorder columns: timestep/scenario/tech/unit/value
    level_df = level_df[["timestep", "scenario", "tech", "unit", "value"]]
    cap_df = cap_df[["scenario", "tech", "unit", "value"]]

    return level_df, cap_df


def timeseries_to_yearly_total(df):
    """Convert wide time series dataframe to yearly total per scenario and tech.

    Automatically detects numeric tech columns.
    """
    df = df.copy()

    # Ensure datetime
    df["timestep"] = pd.to_datetime(df["timestep"])

    # Extract year
    df["year"] = df["timestep"].dt.year

    # Detect numeric columns (techs)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Melt to long format
    df_long = df.melt(
        id_vars=["scenario", "year"], value_vars=numeric_cols, var_name="tech", value_name="value"
    )

    # Yearly sum
    df_yearly = df_long.groupby(["scenario", "year", "tech"], as_index=False).agg({"value": "sum"})

    df_yearly = df_yearly.drop(columns=["year"])
    df_yearly["value"] = df_yearly["value"] / 1000  # GWh to TWh

    return df_yearly


def electricity_yearly_balance():
    """
    Compute yearly electricity balance per tech in long format, summing over timesteps, in TWh.

    - Generation columns containing 'Load' are negative.
    - All demand columns are negative, but kept with their tech names.
    - If a tech appears in both generation and demand, values are summed into one row.
    """
    base_path = cnf.RESULTS_NEXUS_DIR
    gen_file = base_path / "generation_h.csv"
    demand_file = base_path / "demand_h.csv"

    if not gen_file.exists():
        raise FileNotFoundError("generation_h.csv not found")
    if not demand_file.exists():
        raise FileNotFoundError("demand_h.csv not found")

    gen_df = pd.read_csv(gen_file).fillna(0)
    dem_df = pd.read_csv(demand_file).fillna(0)

    gen_df["timestep"] = pd.to_datetime(gen_df["timestep"])
    dem_df["timestep"] = pd.to_datetime(dem_df["timestep"])

    rows = []

    # --- Process generation ---
    gen_tech_cols = [c for c in gen_df.columns if c not in ["scenario", "unit", "timestep"]]
    for scenario, gdf in gen_df.groupby("scenario"):
        yearly_sums = gdf[gen_tech_cols].sum() / 1000  # GWh → TWh
        for tech, value in yearly_sums.items():
            if "Load" in tech:
                value = -abs(value)  # flip sign for generation loads
            rows.append({"scenario": scenario, "tech": tech, "unit": "TWh", "value": value})

    # --- Process demand ---
    dem_tech_cols = [c for c in dem_df.columns if c not in ["scenario", "unit", "timestep"]]
    for scenario, ddf in dem_df.groupby("scenario"):
        yearly_sums = ddf[dem_tech_cols].sum() / 1000  # GWh → TWh
        for tech, value in yearly_sums.items():
            value = -abs(value)  # demand treated as negative
            rows.append({"scenario": scenario, "tech": tech, "unit": "TWh", "value": value})

    # --- Build DataFrame ---
    balance_long_df = pd.DataFrame(rows)

    # --- Combine duplicates (net value per tech) ---
    balance_long_df = balance_long_df.groupby(["scenario", "tech", "unit"], as_index=False).sum()

    return balance_long_df


def electricity_monthly_balance():
    """
    Compute monthly electricity balance per tech in long format, summing over timesteps, in TWh.

    - Generation columns containing 'Load' are negative.
    - All demand columns are negative, but kept with their tech names.
    - If a tech appears in both generation and demand, values are summed into one row.
    - Adds a 'month' column as datetime (first day of month) and formatted as three-letter string (Jan, Feb, ...).
    """
    base_path = cnf.RESULTS_NEXUS_DIR
    gen_file = base_path / "generation_h.csv"
    demand_file = base_path / "demand_h.csv"

    if not gen_file.exists():
        raise FileNotFoundError("generation_h.csv not found")
    if not demand_file.exists():
        raise FileNotFoundError("demand_h.csv not found")

    gen_df = pd.read_csv(gen_file).fillna(0)
    dem_df = pd.read_csv(demand_file).fillna(0)

    # Ensure timestep is datetime
    gen_df["timestep"] = pd.to_datetime(gen_df["timestep"])
    dem_df["timestep"] = pd.to_datetime(dem_df["timestep"])

    # Add month column (first day of month)
    gen_df["month"] = gen_df["timestep"].values.astype("datetime64[M]")
    dem_df["month"] = dem_df["timestep"].values.astype("datetime64[M]")

    rows = []

    # --- Process generation ---
    gen_tech_cols = [
        c for c in gen_df.columns if c not in ["scenario", "unit", "timestep", "month"]
    ]
    for (scenario, month), gdf in gen_df.groupby(["scenario", "month"]):
        monthly_sums = gdf[gen_tech_cols].sum() / 1000  # GWh → TWh
        for tech, value in monthly_sums.items():
            if "Load" in tech:
                value = -abs(value)  # flip sign for generation loads
            rows.append(
                {"scenario": scenario, "month": month, "tech": tech, "unit": "TWh", "value": value}
            )

    # --- Process demand ---
    dem_tech_cols = [
        c for c in dem_df.columns if c not in ["scenario", "unit", "timestep", "month"]
    ]
    for (scenario, month), ddf in dem_df.groupby(["scenario", "month"]):
        monthly_sums = ddf[dem_tech_cols].sum() / 1000  # GWh → TWh
        for tech, value in monthly_sums.items():
            value = -abs(value)  # demand treated as negative
            rows.append(
                {"scenario": scenario, "month": month, "tech": tech, "unit": "TWh", "value": value}
            )

    # --- Build DataFrame ---
    balance_monthly_df = pd.DataFrame(rows)

    # --- Combine duplicates (net value per tech per month) ---
    balance_monthly_df = balance_monthly_df.groupby(
        ["scenario", "month", "tech", "unit"], as_index=False
    ).sum()

    # --- Add three-letter month string ---
    balance_monthly_df["month_str"] = balance_monthly_df["month"].dt.strftime("%b")

    return balance_monthly_df


def friendly_secmod_results(tol=1e-8):
    input_path = cnf.SECMOD_RESULTS_PATH
    output_path = cnf.RESULTS_SECMOD_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    for excel_file in input_path.glob("*.xlsx"):
        fname = excel_file.name.lower()

        if "yearly" in fname:
            sheet_col = "carrier"
            col_order = ["scenario", "techs", "carrier", "values", "units"]
        elif "output" in fname:
            sheet_col = "variable"
            col_order = ["scenario", "variable", "techs", "values", "units"]
        else:
            continue

        sheets = pd.read_excel(
            excel_file,
            sheet_name=None,
            index_col=None,   # <-- IMPORTANT
        )

        dfs = []
        for sheet_name, df in sheets.items():
            df = df.copy()

            # first column is ALWAYS techs
            df.rename(columns={df.columns[0]: "techs"}, inplace=True)

            # wide → long
            df_long = df.melt(
                id_vars="techs",
                var_name="scenario",
                value_name="values",
            )

            # coerce ONLY values (never techs)
            df_long["values"] = pd.to_numeric(
                df_long["values"], errors="coerce"
            )

            # add variable / carrier
            df_long[sheet_col] = sheet_name.replace("_CH", "")
            dfs.append(df_long)

        combined_df = pd.concat(dfs, ignore_index=True)

        # drop near-zero and NaN values
        combined_df = combined_df[
            combined_df["values"].abs() >= tol
        ]

        # units
        combined_df["units"] = (
            combined_df["variable"]
            .map(cnf.UNIT_MAPPING_SECMOD)
            .fillna("TWh")
        )

        combined_df = combined_df[col_order]

        out_file = output_path / f"{excel_file.stem}.csv"
        combined_df.to_csv(out_file, index=False)

        print(f"Saved: {out_file}")

# %%
