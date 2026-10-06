"""Grid topology (bus/branch) extraction and mapping from Nexus-e DB-Input Excel files."""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import config as cnf

SWITZERLAND_MAP_PATH = cnf.REPO_ROOT / "resources" / "maps" / "SwissMap_WithBorders_Transparent.png"

# Approximate Swiss LV03 bounding box, derived from this dataset's own CH node
# coordinates (padded). The background PNG carries no georeferencing metadata,
# so this is an approximate registration (xmin, xmax, ymin, ymax) - to polish later.
SWITZERLAND_LV03_EXTENT = (475_000, 835_000, 65_000, 295_000)

VOLTAGE_COLOR_MAPPING = {
    150: "#2ca02c",
    220: "#1f77b4",
    380: "#d62728",
}

# Perpendicular offset (meters) applied when >1 voltage level connects the same
# pair of nodes, so parallel lines are drawn side by side instead of overlapping.
PARALLEL_VOLTAGE_OFFSET_M = 2000


def _load_bus_branch(filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the "bus" and "branch" sheets of a Nexus-e DB-Input Excel file."""
    db_path = cnf.DATA_NEXUS_FLEX_DIR / f"{filename}.xlsx"
    if not db_path.exists():
        raise FileNotFoundError(f"DB-Input file not found: {db_path}")

    bus = pd.read_excel(db_path, sheet_name="bus", header=2)
    branch = pd.read_excel(db_path, sheet_name="branch", header=2)
    return bus, branch


def build_branch_topology(filename: str) -> pd.DataFrame:
    """Build a branch table with S_max_spr and from/to node coordinates.

    Reads the "bus" and "branch" sheets of
    ``{filename}.xlsx`` under `cnf.DATA_NEXUS_FLEX_DIR` and returns one row
    per branch with from_node_id, to_node_id, s_max_spr_mva, and the
    (LV03) coordinates of both endpoint nodes.
    """
    bus, branch = _load_bus_branch(filename)

    coords = bus[["node_id", "coord_x", "coord_y"]]

    branch_topology = branch[["from_node_id", "to_node_id", "S_max_spr (MVA)"]].rename(
        columns={"S_max_spr (MVA)": "s_max_spr_mva"}
    )

    branch_topology = branch_topology.merge(
        coords.rename(
            columns={"node_id": "from_node_id", "coord_x": "from_coord_x", "coord_y": "from_coord_y"}
        ),
        on="from_node_id",
        how="left",
    )
    branch_topology = branch_topology.merge(
        coords.rename(
            columns={"node_id": "to_node_id", "coord_x": "to_coord_x", "coord_y": "to_coord_y"}
        ),
        on="to_node_id",
        how="left",
    )

    return branch_topology


def save_branch_topology(filename: str) -> Path:
    """Build the branch topology table for `filename` and save it as a CSV."""
    branch_topology = build_branch_topology(filename)

    output_dir = cnf.RESULTS_NEXUS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"{filename}_grid_topology.csv"
    branch_topology.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

    return out_file


def _linewidth_scaler(values: pd.Series, min_lw: float = 0.5, max_lw: float = 6.0):
    """Return a function mapping an MVA value to a linewidth in [min_lw, max_lw]."""
    min_val, max_val = values.min(), values.max()

    def scale(value: float) -> float:
        if max_val == min_val:
            return (min_lw + max_lw) / 2
        return min_lw + (value - min_val) / (max_val - min_val) * (max_lw - min_lw)

    return scale


def plot_grid_map(
    filename: str,
    background_map: Path | None = SWITZERLAND_MAP_PATH,
    map_extent: tuple[float, float, float, float] = SWITZERLAND_LV03_EXTENT,
) -> Path:
    """Plot grid nodes and branches, grouped by voltage level, over a Switzerland map.

    - Each node is drawn as a black dot at its (coord_x, coord_y).
    - Branches are grouped by (from_node_id, to_node_id, voltage level); parallel
      circuits in the same group are summed into a single line, whose thickness
      is proportional to the summed S_max_spr (MVA). Color encodes voltage level.
    - When two voltage levels connect the same pair of nodes, their lines are
      offset side by side instead of overlapping.

    Registration of the background map is approximate (the PNG carries no
    georeferencing metadata) - to polish later.
    """
    bus, branch = _load_bus_branch(filename)
    bus = bus[bus["country"] == "CH"]
    coords = bus.set_index("node_id")[["coord_x", "coord_y"]]

    ch_node_ids = set(coords.index)
    branch = branch[
        branch["from_node_id"].isin(ch_node_ids) & branch["to_node_id"].isin(ch_node_ids)
    ]

    edges = (
        branch.groupby(["from_node_id", "to_node_id", "Line Voltage (kV)"])["S_max_spr (MVA)"]
        .sum()
        .reset_index()
        .rename(columns={"Line Voltage (kV)": "voltage_kv", "S_max_spr (MVA)": "s_max_spr_mva"})
    )
    edges = edges.merge(
        coords.rename(columns={"coord_x": "from_x", "coord_y": "from_y"}),
        left_on="from_node_id",
        right_index=True,
    )
    edges = edges.merge(
        coords.rename(columns={"coord_x": "to_x", "coord_y": "to_y"}),
        left_on="to_node_id",
        right_index=True,
    )
    edges["pair"] = edges.apply(lambda r: tuple(sorted((r["from_node_id"], r["to_node_id"]))), axis=1)

    pair_voltages = edges.groupby("pair")["voltage_kv"].unique().apply(sorted).to_dict()
    linewidth = _linewidth_scaler(edges["s_max_spr_mva"])

    fig, ax = plt.subplots(figsize=(9, 7))

    if background_map is not None and Path(background_map).exists():
        img = mpimg.imread(background_map)
        ax.imshow(img, extent=map_extent, zorder=0)
    elif background_map is not None:
        print(f"Background map not found at {background_map}, plotting grid without it.")

    for row in edges.itertuples():
        volt_list = pair_voltages[row.pair]
        offset_idx = volt_list.index(row.voltage_kv) - (len(volt_list) - 1) / 2

        dx, dy = row.to_x - row.from_x, row.to_y - row.from_y
        length = (dx**2 + dy**2) ** 0.5
        perp_x, perp_y = (-dy / length, dx / length) if length else (0, 0)
        offset_x = perp_x * PARALLEL_VOLTAGE_OFFSET_M * offset_idx
        offset_y = perp_y * PARALLEL_VOLTAGE_OFFSET_M * offset_idx

        ax.plot(
            [row.from_x + offset_x, row.to_x + offset_x],
            [row.from_y + offset_y, row.to_y + offset_y],
            color=VOLTAGE_COLOR_MAPPING.get(row.voltage_kv, "gray"),
            linewidth=linewidth(row.s_max_spr_mva),
            solid_capstyle="round",
            alpha=0.85,
            zorder=1,
        )

    ax.scatter(bus["coord_x"], bus["coord_y"], color="black", s=10, zorder=2)

    ax.set_xlim(map_extent[0], map_extent[1])
    ax.set_ylim(map_extent[2], map_extent[3])
    ax.set_aspect("equal")
    ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color=color, lw=3, label=f"{kv} kV")
        for kv, color in VOLTAGE_COLOR_MAPPING.items()
        if kv in edges["voltage_kv"].unique()
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False, title="Voltage level")

    output_dir = cnf.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"{filename}_grid_map.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    return fig_path


if __name__ == "__main__":
    save_branch_topology("Nexuse_DB-Input_v47_TYNDP22-GA08_CentIvPV_pathfndr_WithSecMOD_HighInteg")
    plot_grid_map("Nexuse_DB-Input_v47_TYNDP22-GA08_CentIvPV_pathfndr_WithSecMOD_HighInteg")
