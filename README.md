# WP1-paper-pathfndr

Postprocessing and visualisation of Calliope / SecMOD-Nexus-e results.

## Setup

1. Create the conda environment:

   ```sh
   conda env create -f environment.yml
   conda activate pathfndr
   ```

2. Edit [`config/paths.yaml`](config/paths.yaml) to point at your local raw-data
   locations (e.g. the Nexus-e Dash export folder, SecMOD Excel exports). Paths
   can be absolute or relative to the repo root.

3. Populate `data/` with your raw inputs, following the existing subfolders
   (`data/calliope`, `data/nexus-e`, `data/secmod`). This folder is git-ignored
   (only the folder structure is tracked), so nothing you drop in there gets
   committed.

## Usage

Scripts under `src/` assume the working directory is `src/` itself (they
resolve output locations as `../results` and `../figures` relative to `cnf`
constants, based on repo root). Run `src/run.py` as an interactive script
(it's organised in `# %%` cells) from within `src/`, e.g. in VS Code's Python
Interactive window or Jupyter.

`run.py` does two things:

1. **Postprocessing** (`postprocess.py`): reads raw Nexus-e/SecMOD exports and
   writes tidy CSVs to `results/nexus-e/` and `results/secmod/`.
2. **Plotting** (`plot.py`): reads those CSVs and writes figures to
   `figures/`.

`results/` and `figures/` are generated locally and are git-ignored.

## Repo layout

```
config/       # paths.yaml — machine-specific input/output locations
data/         # raw inputs (git-ignored, structure only)
src/          # postprocessing (postprocess.py, helper.py) and plotting (plot.py) code
```
