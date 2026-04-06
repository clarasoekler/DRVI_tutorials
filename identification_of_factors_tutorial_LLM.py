# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: DRVI tutorial (LLM)
#     language: python
#     name: drvi_tutorial_llm
# ---

# %% [markdown]
# # Identification of factors

# %% [markdown]
# QUESTIONS: Heatmaps: Both plot_latent_dims_in_heatmap calls (Celltypist and available annotations) still use embed, so the heatmaps show one row per factor (no separate +/− rows). 

# %% [markdown]
# In this notebook, we use the already trained DRVI model on the immune dataset to identify biological processes captured by each latent factor. We combine multiple complementary annotation strategies:
#
# 1. **Cell type annotation** — match factors to known cell types using existing labels, pre-trained classifiers (CellTypist), enrichment based tools (BlitzGSEA) and LLM-based tools (AnnDictionary and CASSIA)
# 2. **Biological process annotation** — identify enriched pathways and regulatory programs using gene set enrichment analysis (BlitzGSEA, GSEApy), over-representation analysis (g:Profiler), regulator activity inference (decoupler), and LLM-based summarization (gs2txt, AnnDictionary)
#
# Each tool operates on the gene-level interpretability scores produced by DRVI's built-in scoring API (`model.get_interpretability_scores`). DRVI provides two complementary scoring approaches that can be selected via the `INTERPRETABILITY_MODE` config variable:
#
# - **OOD (Out-of-Distribution)**: Uses decoder reconstructions to calculate per-gene effect scores. Recommended for identifying cell types and the most specific genes of a program.
# - **IND (Within-Distribution)**:  Iterates over all cells to compute weighted mean effects. Captures broader mechanistic effects including shared genes.
#
# All tools in this notebook are **guiding tools**: they summarize large gene-level patterns into interpretable scores, but they do **not** provide definitive labels. Their outputs should always be interpreted in context, compared across methods, and validated against known biology and the original data.
#
# **We always advise examination by a biologist and validation against published literature for any identified processes.**
#
# > **What is a latent factor?**
# > Think of a latent factor as a "hidden program" that DRVI discovered in the data. Each factor captures a pattern of coordinated gene activity across cells — for example, a set of genes that turn on together in T cells, or genes that respond collectively to interferon signaling. Each factor has two directions (+ and −), representing opposite ends of that program (e.g., genes that go *up* vs. *down*). The goal of this notebook is to give each factor-direction a biological name by gathering evidence from multiple annotation tools.

# %% [markdown]
# ## Roadmap
#
# This tutorial follows a four-stage workflow. Each stage builds on the previous one and feeds into the **Curation Table** — the final deliverable where all evidence comes together.
#
# **Stage 0 · Setup and Prepare shared inputs** — Install dependencies, import libraries, configure paths and parameters. Load the DRVI model and its interpretability scores. Split them into per-factor-direction ranked gene lists that all downstream tools consume.
#
# **Stage 1 · Cell type annotation** — Determine which factors correspond to known cell types.
# - *1.1 Known annotations (SMI)* — Align factors with existing cell-type labels via Scaled Mutual Information.
# - *1.2 CellTypist* — Classify cells with a pre-trained atlas model and correlate predictions with factors.
# - *1.3 Cell type enrichment (BlitzGSEA)* — Test whether a factor's top genes overlap curated marker gene sets (CellMarker, PanglaoDB).
# - *1.4 AnnDictionary* — Ask an LLM to identify cell types from gene lists.
# - *1.5 CASSIA* — Multi-agent LLM system with validation and quality scoring.
#
# **Stage 2 · Biological process annotation** — Annotate factors that capture pathways, signaling, or regulatory programs rather than a single cell type.
# - *2.1 BlitzGSEA* — Fast pre-ranked GSEA with analytical null.
# - *2.2 GSEApy* — Permutation-based GSEA (prerank) and Enrichr ORA.
# - *2.3 g:Profiler* — Ordered-query ORA with hierarchical GO correction.
# - *2.4 decoupler* — Infer transcription factor and pathway activity from prior knowledge networks.
# - *2.5 gs2txt* — LLM-based summarization of gene sets into process descriptions.
#
# **Stage 3 · Curation & export** — Merge all tool outputs into a single curation table, add manual annotations, and store final labels in the embedding object.
#
# **Stage 4 · Visual validation** — Spot-check annotated factors on UMAP plots.
#
# > **Sections are independent within each stage.** You can skip any tool you don't need (e.g., skip CellTypist if no model matches your tissue, skip LLM sections if no LLM backend is available). 

# %% [markdown]
# ## Prerequisites
#
# This notebook assumes that you have already trained a DRVI model and computed the interpretability scores (via `model.calculate_interpretability_scores` in the general pipeline).
#
# Please refer to the [General training and interpretability pipeline](./general_pipeline.html) tutorial.
#
# While we use the immune dataset as a running example, all code is dataset-agnostic. Global configuration (paths, species, thresholds, LLM backend) is defined in the **Global Configurations** section. Each tool section has its own config cell for tool-specific settings (e.g., database choice, model name).

# %% [markdown]
# ## Contact
#
# For questions and help requests, you can reach out in the [scverse discourse](https://discourse.scverse.org/).
#
# If you found a bug, please use the [issue tracker](https://github.com/theislab/drvi/issues).

# %% [markdown]
# ## Adapting this notebook to your own dataset
#
# To reuse this notebook on a different dataset or DRVI model, check and update the following items:
#
# - **1. File paths and IO**
#   - Update `io_dir` to point to your project directory.
#   - Make sure the following files exist under `io_dir` with your data:
#     - `adata_preprocesses.h5ad` (preprocessed AnnData with HVGs and UMAP)
#     - `drvi_model/` (trained DRVI model directory)
#     - `embed.h5ad`
#     - Optionally: `immune_all.h5ad` (or your equivalent full-gene data) for defining `all_genes`.
#
# - **2. Interpretability mode**
#   - Set `INTERPRETABILITY_MODE` in the Config cell to `"OOD"` (out-of-distribution, default) or `"IND"` (within-distribution).
#   - Both score sets must have been computed in the general pipeline via `model.calculate_interpretability_scores(embed, "OOD")` and `model.calculate_interpretability_scores(embed, "IND")`.
#
#
# - **3. Cell-level annotations (optional but recommended)**
#   - If you have cell-type labels, set `annot_col` in the Config cell to the corresponding column in `adata.obs` (e.g. `"final_annotation"`).
#   - If you do **not** have annotations, set `annot_col = None` and skip:
#     - Section 1.1 (SMI with known annotations)
#     - Section 4 (visual validation on UMAP).
#
# - **4. Species**
#   - Set `organism`, `dc_organism`, and `llm_species` in the Config cell to match your species. All three are different formats for the same thing:
#     - `organism = "hsapiens"` (g:Profiler format)
#     - `dc_organism = "human"` (decoupler format)
#     - `llm_species = "Homo sapiens"` (LLM tools format)
#   - For non-human species, check that your gene-set resources and `all_genes` use the same gene-name casing; you may need to remove `.str.upper()` when working with mouse.
#
# - **5. Gene-set databases (section-level configs)**
#   - Each tool section has its own database config cell:
#     - BlitzGSEA: `gsea_db` (e.g. `"GO_Biological_Process_2023"`)
#     - g:Profiler: `gp_source` (e.g. `["GO:BP"]`, `["REAC"]`)
#     - decoupler: `dc_geneset` (e.g. `"collectri"`, `"dorothea"`, `"progeny"`)
#     - Cell-type ORA: `celltype_dbs` (e.g. `["CellMarker_Augmented_2021"]`)
#     - GSEApy: `gseapy_db`
#
#
# - **6. CellTypist (optional)**
#   - Choose a model via `ct_model` that matches your tissue / species (e.g. `"Immune_All_Low.pkl"` for PBMC, `"Developing_Mouse_Brain.pkl"` for mouse brain).
#   - If no suitable model exists, skip the CellTypist section and rely on your own annotations plus the enrichment / decoupler tools using cell-type databases.
#
#
# - **7. Significance thresholds**
#   - `significance_threshold` controls:
#     - FDR cutoffs for BlitzGSEA and decoupler.
#     - The g:SCS-corrected p-value cutoff in g:Profiler (treated analogously to an FDR threshold).
#
# - **8. LLM backend**
#   - `OLLAMA_URL` and `OLLAMA_MODEL` in the Config cell control all three LLM tools (AnnDictionary, CASSIA, gs2txt).
#   - For local inference: set up Ollama following the setup guide in §1.4–1.5 and point `OLLAMA_URL` to your server.
#   - For cloud providers: update the provider and API key in each tool's config cell (see each tool's GitHub for supported backends).
#
# - **9. Manual curation**
#   - Use the exported `factor_annotation_curation.csv` as your central place to:
#     - Inspect top genes and tool suggestions per factor-direction.
#     - Define `MANUAL_LABELS` and `MANUAL_NOTES` in the helper cell.
#   - Re-import the curated CSV and re-run the final cells to store `embed.var["annotation_final"]` and `embed.var["annotation_source"]` with your labels.

# %% [markdown]
# ## 0. Setup and prepare shared inputs

# %% [markdown]
# ### Install
#
# If you try DRVI on colab, the next cell will install dependencies.
#
# Please remove this part if your environment is already set up.

# %%
import sys
import subprocess

branch = "latest"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB and branch == "stable":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "drvi[tutorials]"])
elif IN_COLAB and branch != "stable":
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "git+https://github.com/theislab/drvi.git#egg=drvi[tutorials]"])

if IN_COLAB:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "celltypist", "blitzgsea", "gprofiler-official", "decoupler",
                           "anndict","CASSIA", "gs2txt"])

# %% [markdown]
# ### Imports

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

import scvi
import drvi
from pathlib import Path
from drvi.model import DRVI

# %%
print("Last run with scvi-tools version:", scvi.__version__)
print("Last run with DRVI version:", drvi.__version__)

# %% [markdown]
# ### Global Configurations
# All parameters that control this tutorial are set in the next two cells. Tool-specific settings (database choices, model names) are set at the top of each tool section.

# %%
# Plot defaults
sc.settings.set_figure_params(dpi=100, frameon=False, figsize=(3, 3))
plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14})

# %%
# ---- Set input output directory ----
# We use tmp_io/ directory in the same place as this notebook. Update accordingly.
io_dir = Path("/home/icb/clara.sanchez/data/drvi_immune_128")

# ---- Interpretability method ----
# "OOD" — Out-of-distribution: uses decoder reconstructions (faster, default)
# "IND" — Within-distribution: iterates over all cells (captures broader effects)
INTERPRETABILITY_MODE = "OOD"

_SCORE_KEY_MAP = {
    "OOD": "OOD_combined",
    "IND": "IND_linear_weighted_mean",
}
assert INTERPRETABILITY_MODE in _SCORE_KEY_MAP, (
    f"Invalid INTERPRETABILITY_MODE={INTERPRETABILITY_MODE!r}. Choose 'OOD' or 'IND'."
)
score_key = _SCORE_KEY_MAP[INTERPRETABILITY_MODE]
print(f"Interpretability mode: {INTERPRETABILITY_MODE} (score_key={score_key!r})")

# ---- Global significance threshold used across all tools ----
# Note: for BlitzGSEA and decoupler this is an FDR cutoff, while for g:Profiler
# it is applied to g:SCS-corrected p-values (not classical FDR).
significance_threshold = 0.05

# ---- Cell-type annotation ----
# Column in adata.obs containing cell type labels. Set to None if not available.
annot_col = "final_annotation"
# Minimum SMI score to consider a factor associated with a cell type.
smi_threshold = 0.5

# ---- Species (used by g:Profiler, decoupler, LLM tools) ----
organism = "hsapiens"       # g:Profiler format: "hsapiens", "mmusculus"
dc_organism = "human"       # decoupler format: "human", "mouse"
llm_species = "Homo sapiens"

UPPERCASE_GENES = True  # Set False for mouse or other species with mixed-case gene symbols

# %% [markdown]
# ### Ollama setup guide 
#
# Ollama is an open-source free tool that lets you run and manage large language models (like Llama 3 or Qwen) locally on your own hardware or cluster
#
# **0. Installation (one-time)**
#
# Since most clusters do not allow Docker directly, we use Apptainer (formerly Singularity) to convert the official Ollama image into a portable `.sif` file. Run this on a login node:
#
# ```bash
# mkdir -p ~/containers
# apptainer pull ~/containers/ollama.sif docker://ollama/ollama:latest
# ```
#
# **1. Start a GPU job**
#
# Ollama requires a GPU for acceptable performance. Request a GPU node via your cluster's scheduler (e.g. Slurm).
#
# **2. Launch the container**
#
# Once on a compute node, launch the Apptainer shell. Replace the `--bind` paths with your cluster storage paths:
#
# ```bash
# apptainer shell --nv \
#     --bind /localscratch \
#     --bind /lustre/groups/ml01/ \
#     ~/containers/ollama.sif
# ```
#
# **3. Start the Ollama server**
#
# On shared clusters, use a custom port to avoid conflicts with other users. Replace with your port.
#
# ```bash
# OLLAMA_HOST=0.0.0.0:8979 ollama serve & 
# ```
#
# If the server stays in the foreground, press `Ctrl+Z` then type `bg` to background it.
#
# **4. Connect and pull a model**
#
# Point the client to the same port and download your model:
#
# ```bash
# export OLLAMA_HOST=127.0.0.1:8979
# ollama pull qwen2.5
# ```
#
# Update `OLLAMA_URL` and `OLLAMA_MODEL` in the Config cell to match the hostname, port, and model you chose here.

# %%
import requests

# ---- OLLAMA CONFIGURATION (shared by AnnDictionary, CASSIA, gs2txt)----

# Ensure you have run: ollama pull qwen2.5:latest
NODE_NAME = "supergpu23.scidom.de" # Update with your node's hostname 
OLLAMA_PORT = "8979" # Update if you used a different port when starting the server

# Use the exact name from your 'ollama list' output
OLLAMA_MODEL = "qwen2.5:latest" 

OLLAMA_URL = f"http://{NODE_NAME}:{OLLAMA_PORT}" 

# Biological Context
llm_top_n_genes = 30
llm_tissue_context = "human bone marrow / immune"

# Standardized output columns for all tools
LLM_COLS = [
    "Factor_ID", "Direction", "Tool_Name", "Model_Used",
    "Predicted_Celltype", "Predicted_Process", "Confidence_Score", "Evidence_Reasoning",
]

print(f"Configured for Ollama ({OLLAMA_MODEL}) at {OLLAMA_URL}")

# %%
# testing connection 
try:
    response = requests.get(f"{OLLAMA_URL}/api/tags")
    print("✅ Connection to LLM-Server is working!")
    print("Available models:", response.json())
except Exception as e:
    print(f"❌ Error: {e}")

# %% [markdown]
# ### Load Data

# %%
# Update this path to point to your project directory
adata = sc.read_h5ad(io_dir / "adata_preprocesses.h5ad")
adata

# %% [markdown]
# ### Load DRVI outputs

# %%
model_path = io_dir / "drvi_model"
embed_path = io_dir / "embed.h5ad"


model = DRVI.load(model_path, adata)
embed = sc.read_h5ad(embed_path)

# %% [markdown]
# ### Prepare shared inputs
#
# All annotation tools operate on the gene-level interpretability scores computed by DRVI. For each latent factor, the positive (+) and negative (−) direction each produce a vector of per-gene scores that quantify how much each gene's predicted expression changes. These scores serve as:
#
# - **Ranked gene lists** for GSEA-style tools (BlitzGSEA)
# - **Ordered gene queries** for ORA-style tools (g:Profiler)
# - **Gene × factor score matrices** for decoupler which uses statistical models (like ULM or MLR) to infer regulator activity by integrating these scores with a prior knowledge network
#
# The scoring approach is controlled by `INTERPRETABILITY_MODE` (set in Config above):
# - **OOD** scores emphasize factor-specific genes via decoder reconstructions.
# - **IND** scores capture broader mechanistic effects by averaging over all cells.
#
# We prepare these shared inputs once and reuse them across all tools.

# %%
# Align adata and embed to shared cells
common_cells = adata.obs_names.intersection(embed.obs_names)
print(f"Cells: adata={adata.n_obs}, embed={embed.n_obs}, shared={len(common_cells)}")
adata = adata[common_cells].copy()
embed = embed[common_cells].copy()

# Remove vanished dimensions
embed_nv = embed[:, ~embed.var["vanished"].astype(bool)].copy()
factor_ids = embed_nv.var["title"].astype(str).tolist()
print(f"Active (non-vanished) factors: {embed_nv.n_vars}")

# Get gene-level interpretability scores via the DRVI model API.
# Returns a DataFrame of shape (n_genes, 2*n_factors): one column per factor-direction.
scores_df = model.get_interpretability_scores(embed, adata, key=score_key)
print(f"Scores shape: {scores_df.shape}  (genes x factor-directions)")

# Split into positive (+) and negative (-) direction DataFrames, filtered to non-vanished factors.
# Strip the +/- suffix so column names match factor_ids for downstream tools.
pos_cols = [f"{fid}+" for fid in factor_ids]
neg_cols = [f"{fid}-" for fid in factor_ids]
pos_df = scores_df[pos_cols].rename(columns=lambda c: c[:-1])
neg_df = scores_df[neg_cols].rename(columns=lambda c: c[:-1])

# Background gene universe: all genes measured in the experiment (before HVG filtering).
raw_data_path = io_dir / "immune_all.h5ad"
adata_full = sc.read_h5ad(raw_data_path, backed="r")


all_genes = adata_full.var_names.astype(str).str.strip()
if UPPERCASE_GENES:
    all_genes = all_genes.str.upper()
adata_full.file.close()
all_genes = pd.Index(all_genes).drop_duplicates().tolist()
print(f"Background genes: {len(all_genes)}")


# %% [markdown]
# ### Helper functions
# These utilities standardize gene names, build ranked gene lists, and provide a shared iterator over all factor-directions. They are used by every tool below.

# %%
# Function to build standardized inputs for enrichment tools from the raw scores DataFrames.
def build_inputs(df, all_genes=all_genes):
    """Build enrichment inputs from a genes x factors score matrix.

    Returns
    -------
    std : DataFrame
        Standardized genes x factors scores (uppercased, duplicates merged by max).
    ranked : dict
        {factor: Series sorted by descending score} for GSEA tools.
    """
    std = df.copy()
    std.index = std.index.astype(str).str.strip()
    if UPPERCASE_GENES:
        std.index = std.index.str.upper()
    std = std.groupby(std.index).max()

    if all_genes is not None:
        idx = pd.Index(pd.Series(all_genes).astype(str)).drop_duplicates()
        std = std.reindex(idx)

    ranked = {c: std[c].dropna().sort_values(ascending=False) for c in std.columns}
    return std, ranked


pos_std, pos_ranked = build_inputs(pos_df)
neg_std, neg_ranked = build_inputs(neg_df)

# For tools that require separate factor-directions, we can iterate over them like this:
def iter_factor_directions():
    """Yield (factor_id, factor_label, sign, ranked_series) for each factor-direction."""
    for fac in factor_ids:
        for sign, ranked in [("+", pos_ranked), ("-", neg_ranked)]:
            yield fac, f"{fac}{sign}", sign, ranked[fac]

# Factor-direction labels used throughout (e.g., "DR 36+")
factor_dir_labels = [f"{f}+" for f in factor_ids] + [f"{f}-" for f in factor_ids]

print(f"Factors: {len(factor_ids)}")
print(f"Factor-directions: {len(factor_dir_labels)}")
print(f"Genes per ranked list: {len(next(iter(pos_ranked.values())))}")

# %% [markdown]
# ## 1. Cell type annotation
#
# Some latent factors capture cell-type identity. We identify these using five complementary approaches:
#
# - **1.1 Known annotations (SMI)** — if your dataset has cell-type labels, measure alignment via Scaled Mutual Information.
# - **1.2 CellTypist** — classify cells with a pre-trained atlas model and correlate predictions with factors.
# - **1.3 Cell type enrichment (BlitzGSEA)** — test factor gene lists against curated marker databases (CellMarker, PanglaoDB).
# - **1.4 AnnDictionary** — LLM-based cell type annotation from gene lists.
# - **1.5 CASSIA** — multi-agent LLM annotation with quality scoring.

# %% [markdown]
# ### 1.1 Known annotations (SMI)
#
# If your dataset has existing cell type annotations, Scaled Mutual Information (SMI) measures how well each latent factor aligns with each annotated category. SMI is normalized to [0, 1], where 1 indicates perfect correspondence between a factor and a cell type.
#
# In this dataset we have annotations stored in `adata.obs["final_annotation"]`.
#
# **Skip this section if your dataset does not have cell type annotations.**

# %% [markdown]
# ####  Imports

# %%
import math
import networkx as nx
from drvi.utils.metrics import DiscreteDisentanglementBenchmark

# %%
# annot_col is set in Global Configurations above.
# Change it there if you need a different column.

# %% [markdown]
# #### Scaled Mutual Information

# %%
# Compute SMI between each factor-direction (+/-) and each annotated cell type.
titles = embed_nv.var["title"]
combined_X = np.hstack([embed_nv.X, -embed_nv.X])
combined_titles = [f"{t}+" for t in titles] + [f"{t}-" for t in titles]

benchmark = DiscreteDisentanglementBenchmark(
    combined_X,
    dim_titles=combined_titles,
    discrete_target=embed.obs[annot_col],
    metrics=["SMI-disc"],
    aggregation_methods=["LMS"],
)
benchmark.evaluate()
smi_similarity = benchmark.get_results_details()["SMI-disc"]
smi_similarity.index.name = "title"

# %% [markdown]
# #### Heatmap

# %%
drvi.utils.pl.plot_latent_dims_in_heatmap(
    embed, 
    annot_col, 
    title_col="title", 
    sort_by_categorical=True,
    figsize=(40, 16),
    show=False,
)

plt.tight_layout()
plt.savefig(io_dir / "heatmap_known_annotations.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# #### Summary Table

# %%

smi_top_matches = (
    smi_similarity.reset_index()
    .melt(id_vars="title", value_vars=smi_similarity.columns)
    .query("value >= @smi_threshold")
    .reset_index(drop=True)
)
print(
    f"Factor–cell type pair matches with SMI >= {smi_threshold}: "
    f"{len(smi_top_matches)} / {len(smi_similarity)} factors"
)
display(smi_top_matches.sort_values("value", ascending=False))


# %% [markdown]
# #### Helper function for Plot Packed Network Visualization

# %%
def plot_packed_network(df, title_col="title", var_col="variable", val_col="value"):
    """Visualizes factor–cell type associations as a network with edge weights."""
    G = nx.from_pandas_edgelist(df, title_col, var_col, edge_attr=val_col)

    pos = {}
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    cols = math.ceil(len(components) ** 0.5)
    for i, nodes in enumerate(components):
        sub_pos = nx.spring_layout(G.subgraph(nodes), weight=val_col, k=0.5, seed=42)
        r, c = divmod(i, cols)
        for n, (x, y) in sub_pos.items():
            pos[n] = (x + c * 3, y - r * 3)

    plt.figure(figsize=(14, 10))
    titles = set(df[title_col])
    nx.draw(
        G, pos,
        with_labels=True, font_size=8, font_weight="bold", node_size=600,
        node_color=["#A0CBE2" if n in titles else "#FF9E9E" for n in G.nodes()],
        width=[d[val_col] * 4 for u, v, d in G.edges(data=True)],
        edge_color="grey", alpha=0.6,
    )
    edge_labels = {(u, v): f"{d[val_col]:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.axis("off")
    plt.show()


# %% [markdown]
# #### Plot Packed Visualization

# %%
plot_packed_network(smi_top_matches)

# %% [markdown]
# ### 1.2 CellTypist
#
# [CellTypist](https://www.celltypist.org/) uses pre-trained logistic regression models trained on large-scale annotated atlases to classify individual cells. We calculate the Scaled Mutual Information (SMI) between the CellTypist probability matrix (cells × cell types) and the DRVI factor activity matrix (cells × factors) to identify which factors correspond to which cell types.
#
# **Skip this section if no CellTypist model matches your tissue.**
#

# %% [markdown]
# #### Imports

# %%
import celltypist
from celltypist import models

# %% [markdown]
# #### CellTypist Model

# %%
# Run celltypist.models.models_description() to see all available models. Choose one matching your tissue. 
ct_model = "Immune_All_Low.pkl"  # e.g., "Developing_Mouse_Brain.pkl" for mouse brain
models.download_models(force_update=False, model=ct_model)

ct_model = models.Model.load(model=ct_model)
# Run print(ct_model.cell_types) to see available cell types


# %% [markdown]
# #### CellTypist Annotation

# %% [markdown]
# Each cell receives a predicted label via logistic regression based on its transcriptomic profile. Setting majority_voting=True refines these labels by assigning the most frequent label within a cell's local neighborhood (kNN), reducing technical noise. The resulting per-cell labels are stored in adata.obs.

# %%
predictions = celltypist.annotate(adata, model=ct_model, majority_voting=True)
adata.obs["celltypist_labels"] = predictions.predicted_labels["predicted_labels"]
adata.obs["celltypist_majority"] = predictions.predicted_labels["majority_voting"]

# %% [markdown]
# #### Extract Probability Matrix

# %%
# Probability matrix: sigmoid-transformed decision scores (cells x cell types)
prob_matrix = predictions.probability_matrix
prob_matrix.index = adata.obs_names
print(f"Probability matrix: {prob_matrix.shape[0]} cells x {prob_matrix.shape[1]} cell types")

# %% [markdown]
# #### CellTypist Mutual Information

# %%
# Compute SMI between each factor-direction and each CellTypist-predicted cell type.
ct_titles = embed_nv.var["title"]
ct_combined_X = np.hstack([embed_nv.X, -embed_nv.X])
ct_combined_titles = [f"{t}+" for t in ct_titles] + [f"{t}-" for t in ct_titles]

ct_benchmark = DiscreteDisentanglementBenchmark(
    ct_combined_X,
    dim_titles=ct_combined_titles,
    discrete_target=adata.obs.loc[embed_nv.obs_names, "celltypist_majority"],
    metrics=["SMI-disc"],
    aggregation_methods=["LMS"],
)
ct_benchmark.evaluate()
ct_smi_matrix = ct_benchmark.get_results_details()["SMI-disc"]
ct_smi_matrix.index.name = "factor"
print(f"CellTypist SMI matrix: {ct_smi_matrix.shape} (factor-directions x CellTypist labels)")

# %% [markdown]
# #### Heatmap

# %%
embed.obs["celltypist_majority"] = (
    adata.obs["celltypist_majority"]
    .reindex(embed.obs_names)
)

drvi.utils.pl.plot_latent_dims_in_heatmap(
    embed, 
    "celltypist_majority", 
    title_col="title", 
    sort_by_categorical=True,
    figsize=(40, 16),
    show=False,  
)

plt.tight_layout()
plt.savefig(io_dir / "heatmap_celltypist.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# #### Summary Table

# %%
# Top CellTypist match per factor-direction
idx_name = ct_smi_matrix.index.name or "index"
ct_long = (
    ct_smi_matrix.reset_index()
    .melt(id_vars=idx_name, var_name="cell_type", value_name="smi")
    .rename(columns={idx_name: "factor"})
    .sort_values("smi", ascending=False)
    .drop_duplicates(subset="factor", keep="first")
)

ct_significant = ct_long.query("smi >= @smi_threshold").copy()

print(
    f"CellTypist matches with SMI >= {smi_threshold}: "
    f"{len(ct_significant)} / {len(ct_long)} factors"
)

display(ct_significant.sort_values("smi", ascending=False))


# %% [markdown]
# ### 1.3 Cell Type Enrichment (BlitzGSEA)

# %% [markdown]
# A purely statistical cell type annotation approach: run BlitzGSEA with cell-type marker gene libraries
# (CellMarker 2.0, PanglaoDB) instead of pathway databases. Gene sets in these libraries correspond to
# known marker gene profiles for specific cell types.
#
# This is complementary to CellTypist (which classifies cells using expression matrices) and LLM tools
# (which reason over gene lists). This method tests whether a factor's top genes significantly overlap with curated marker sets — no API key or pretrained model required.

# %%
import blitzgsea as blitz

# %%
# ─── Config ───────────────────────────────────────────────────────────────────
celltype_dbs = ["CellMarker_2024", "PanglaoDB_Augmented_2021"]
# Or try: "ARCHS4_Cell-lines", "Tabula_Sapiens", "Tabula_Muris", "Human_Gene_Atlas"
# ─────────────────────────────────────────────────────────────────────────────

celltype_ora_rows = []

for db in celltype_dbs:
    lib = blitz.enrichr.get_library(db)
    for fac, factor_label, sign, series in iter_factor_directions():
            signature = series.rename("v").reset_index().rename(columns={"index": "i"})
            signature["v"] = pd.to_numeric(signature["v"], errors="coerce")
            signature = signature.dropna(subset=["v"])
            try:
                res = blitz.gsea(signature, lib, processes=4)
                sig = res[res["fdr"] < significance_threshold].sort_values("fdr").head(5)
                for term, row in sig.iterrows():
                    celltype_ora_rows.append({
                        "factor": factor_label,
                        "database": db,
                        "cell_type": term,
                        "NES": round(float(row["nes"]), 3),
                        "FDR": float(row["fdr"]),
                    })
            except Exception as e:
                print(f"Cell type ORA failed for {factor_label} / {db}: {e}")

celltype_ora_results = pd.DataFrame(celltype_ora_rows)

# %%
print(f"Cell type ORA hits: {len(celltype_ora_results)} across {celltype_ora_results['factor'].nunique()} factors")

display(celltype_ora_results.sort_values(["factor", "FDR"]))

# %% [markdown]
# ### 1.4–1.5 LLM-based cell type annotation
#
# The following tools use Large Language Models (LLMs) to annotate factor-directions by interpreting their top gene loadings. Each tool sends a gene list to one or more LLMs and returns a predicted cell type or process label.
#
# **Default provider:** Local Ollama (`provider="ollama"`) for fully local inference with no external API key.
#
# **Alternative providers:** Change any tool's provider or model in the respective section according to the tools respective github repository.
#
#

# %% [markdown]
# ### 1.4 AnnDictionary
#
# [AnnDictionary](https://github.com/ggit12/anndictionary) ([Nature Comms 2025](https://nature.com/articles/s41467-025-64511-x)) provides LLM-provider-agnostic cell type and biological process annotation built on LangChain and AnnData. It sends a gene list to a configured LLM backend with a structured prompt and returns a cell type label. In benchmarks on the Tabula Sapiens atlas, LLM annotation of major cell types achieved 80–90% accuracy, with Claude 3.5 Sonnet performing best.
#
# - **Input**: Unordered gene list (top-N genes by DRVI effect score per factor-direction) and optional tissue context
# - **Algorithm**: Sends the gene list and tissue information to the configured LLM via a structured prompt asking "what cell type do these marker genes represent?" The LLM performs zero-shot semantic matching against its training knowledge of gene–cell type associations
# - **Output**: A single cell type label or a biological process label as a plain text string
# - **LLM backend**: Provider-agnostic — supports Google Gemini, OpenAI, Anthropic, AWS Bedrock, Azure OpenAI, Azure ML endpoints, Cohere, HuggingFace, Vertex AI, Ollama and others via a single `configure_llm_backend()` call. For further information on how to implement your desired LLM provider an model please check out [AnnDictionary](https://github.com/ggit12/anndictionary).
#
# AnnDictionary does not provide a confidence score natively. 
# We use AnnDictionary here with the open-source and free resource Ollama. 

# %%
import anndict as adic

# %%
adic.configure_llm_backend(
    provider="ollama",   # Options: "openai", "anthropic", "google", "bedrock", "azureml_endpoint" etc. 
    model="qwen2.5:latest", # Options: "gpt-3.5-turbo", "claude-sonnet-4-5-20250929", "gemini-1.5-pro", etc. 
    api_key="ollama", 
    base_url=OLLAMA_URL
)

def run_anndictionary(pos_df, neg_df):
    rows = []
    for fac in pos_df.columns:
        for sign, df in [("+", pos_df), ("-", neg_df)]:
            genes = df[fac].sort_values(ascending=False).head(llm_top_n_genes).index.tolist()
            try:
                ct = adic.ai_cell_type(gene_list=genes, tissue=llm_tissue_context)
                bp = adic.ai_biological_process(gene_list=genes)
            except Exception as e:
                print(f"❌ AnnDictionary failed for {fac}{sign}: {e}")
                ct, bp = None, None
            rows.append({
                "Factor_ID": fac, "Direction": sign,
                "Tool_Name": "AnnDictionary", "Model_Used": OLLAMA_MODEL,
                "Predicted_Celltype": ct, "Predicted_Process": bp,
                "Confidence_Score": None, "Evidence_Reasoning": None,
            })
            print(f"✅ {fac}{sign} done")
    return pd.DataFrame(rows).reindex(columns=LLM_COLS)


# %%
# To run:
df_anndictionary = run_anndictionary(pos_df, neg_df)
df_anndictionary

# %% [markdown]
# ### 1.5 CASSIA
#
# [CASSIA](https://github.com/ElliotXie/CASSIA) ([Nature Comms 2025](https://www.nature.com/articles/s41467-025-67084-x)) is a multi-agent LLM system for automated, interpretable cell type annotation. It uses dedicated agents for annotation, validation, formatting, quality scoring, and reporting.
#
# - **Input**: Marker genes, species, tissue context, LLM model
# - **Algorithm**:
#   1. *Annotation agent*: Proposes cell type labels with detailed biological reasoning based on marker gene expression patterns, using a zero-shot chain-of-thought approach that mimics the workflow a computational biologist would follow
#   2. *Validation agent*: Iteratively checks annotations for consistency in marker–cell type alignment
#   3. *Formatting agent*: Summarizes each cell annotation
#   4. *Quality scoring agent*: Assigns a quality score (0–100) based on scientific accuracy and marker balance
#   5. *Reporter agent*: Provides full interpretability with detailed reasoning, quality scores, and refinements
#   6. *Optional agents*:
#     - *Annotation Boost*: Improves low-scoring annotations
#     - *Subclustering*: Resolves mixed cell populations with subtle phenotypic differences
#     - *RAG (retrieval-augmented generation)*: Integrates external knowledge from databases like CellMarker and ontologies
#     - *Uncertainty Quantification*
# - **Output**: Cell type annotation with **quality score (0–100)**, detailed biological reasoning trace, and HTML report with evidence documentation
# - **Supported providers**: OpenAI, Anthropic, OpenRouter (including free open-source models via OpenRouter)
#
# CASSIA is MIT-licensed (commercial-friendly) and provides a [web UI](https://www.cassia.bio/) for interactive use.
#
# *Recommended API KEY: `OPENROUTER_API_KEY`.* We use Ollama for this tutorial. For further information on other models, check out [CASSIA](https://github.com/ElliotXie/CASSIA).

# %%
import CASSIA
from CASSIA import runCASSIA_batch

# %%
#Provider Setup
CASSIA.set_api_key("ollama", provider=f"{OLLAMA_URL}/v1")


# %%
def run_cassia_local(pos_df, neg_df):
    rows = []
    for d, df in [("+", pos_df), ("-", neg_df)]:
        for fac in df.columns:
            genes = df[fac].sort_values(ascending=False).head(llm_top_n_genes).index.tolist()
            rows.append({"cluster": f"{fac.replace(' ', '_')}{d}", "gene": ", ".join(genes)})
    markers_df = pd.DataFrame(rows)
    print(f"Markers: {len(markers_df)} rows, {markers_df['cluster'].nunique()} clusters")

    CASSIA.runCASSIA_batch(
        marker=markers_df,
        output_name="cassia_drvi",
        provider=f"{OLLAMA_URL}/v1",
        model=OLLAMA_MODEL,
        tissue=llm_tissue_context,
        species=llm_species,
        max_workers=4,
        validate_api_key_before_start=False,
    )

    raw = pd.read_csv("cassia_drvi_summary.csv")

    return pd.DataFrame({
        "Factor_ID":         raw["Cluster ID"].str[:-1].str.replace("_", " "),
        "Direction":         raw["Cluster ID"].str[-1],
        "Predicted_General": raw["Predicted General Cell Type"],
        "Predicted_Detailed": raw["Predicted Detailed Cell Type"],
    })


# %%
cassia_results = run_cassia_local(pos_df, neg_df)
n_ok = cassia_results["Predicted_Detailed"].notna().sum()
print(f"CASSIA annotations: {n_ok} / {len(cassia_results)} factor-directions")
display(cassia_results)

# %%
cassia_ct = {f"{r['Factor_ID']}{r['Direction']}": f"CASSIA: {r['Predicted_Detailed']}"
             for _, r in cassia_results.iterrows() if pd.notna(r["Predicted_Detailed"])}

# %% [markdown]
# ## 2. Biological process identification
#
# Factors that do not map to a single cell type often capture biological processes (e.g., interferon response, cell cycle, stress). We use three complementary enrichment approaches, each with different strengths:
#
# | Tool | Method | Input | Strengths |
# |------|--------|-------|-----------|
# | **BlitzGSEA** | Pre-ranked GSEA | Full ranked gene list | Fast; uses entire ranking; uses an analytical null distribution |
# | **GSEApy prerank** | Pre-ranked GSEA (permutation) | Full ranked gene list | Empirical p-values via permutation; provides leading-edge genes |
# | **GSEApy enrichr** | Over-representation (ORA) | Top-N gene list | Fast; access to 200+ Enrichr databases; no ranking needed |
# | **g:Profiler** | Over-representation (ORA) | Ordered gene query | Robust multiple-testing (g:SCS); well-suited for biological pathways and GO terms |
# | **decoupler** | Activity Inference (ULM/MLM) | Gene score matrix + Prior Knowledge | Regression-based; identifies specific regulatory drivers (e.g., TFs) using curated networks |
# | **gs2txt** | LLM-based summarization | Top-N gene list + pathway enrichment | A specialized tool for converting enrichment results into prose via LLMs. |

# %% [markdown]
# ### 2.1 BlitzGSEA
#
# [BlitzGSEA](https://github.com/MaayanLab/blitzgsea) performs pre-ranked Gene Set Enrichment Analysis using an analytical approximation of the null distribution rather than permutations, enabling high-performance enrichment testing across many factors.
#
# - **Input**: Full ranked gene list (genes sorted by their DRVI effect scores, capturing the magnitude and direction of expression change)
# - **Output**: Normalized Enrichment Score (NES) and FDR-adjusted p-values per gene set
# - **Database**: Compatible with any standard .gmt file or Enrichr library (e.g., MSigDB, Reactome)

# %%
# Enrichr library to use. See Appendix for available databases.
# Common choices: "MSigDB_Hallmark_2020", "GO_Biological_Process_2023",
#                 "Reactome_2022", "KEGG_2021_Human"

gsea_db = "GO_Biological_Process_2023"

# %%
signature_lib = blitz.enrichr.get_library(gsea_db)
print(f"Loaded {gsea_db}: {len(signature_lib)} gene sets")

# %%
blitzgsea_rows = []

for fac, factor_label, sign, series in iter_factor_directions():
    # BlitzGSEA expects a DataFrame with columns "i" (gene) and "v" (score)
    signature = series.rename("v").reset_index().rename(columns={"index": "i"})
    signature["v"] = pd.to_numeric(signature["v"], errors="coerce")
    signature = signature.replace([np.inf, -np.inf], np.nan).dropna(subset=["v"])

    try:
        res = blitz.gsea(signature, signature_lib, processes=4)
        sig = res[res["fdr"] < significance_threshold].sort_values("fdr")
        for term, row in sig.head(3).iterrows():
            blitzgsea_rows.append({
                "factor": factor_label,
                "term": term,
                "NES": round(float(row["nes"]), 3),
                "FDR": float(row["fdr"]),
            })
    except Exception as e:
        print(f"BlitzGSEA failed for {factor_label}: {e}")

blitzgsea_results = pd.DataFrame(blitzgsea_rows)


# %%
print(
    f"BlitzGSEA significant directions: {blitzgsea_results['factor'].nunique()} / {len(factor_dir_labels)} "
    f"(with up to 3 terms per direction)"
)
display(blitzgsea_results.sort_values(["factor", "FDR"]))

# %% [markdown]
# ### 2.2 GSEApy

# %% [markdown]
# [GSEApy](https://github.com/zqfang/GSEApy) (Fang et al., Bioinformatics 2023) is the standard Python
# GSEA package. It offers two primary modes for interpreting gene expression data:
#
# 1. **`gseapy.prerank()`** — Classic GSEA algorithm with permutation-based null. Takes the full ranked gene
#    list exactly like BlitzGSEA, but generates an *empirical* p-value from gene score shuffling.
#    Returns NES, nominal p-value, FDR, and **leading-edge genes** (the core driving genes).
# 2. **`gseapy.enrichr()`** — Fast ORA against any Enrichr database via API. No ranking needed, just a
#    top-N gene list. Complements g:Profiler (no ordered query mode, but 200+ databases and instant).
#
# **vs. BlitzGSEA**: different statistical null → use both for cross-validation of hits.  
# **vs. g:Profiler**: no hierarchical GO correction, but broader database coverage and simpler API.
#
# How it works:
# * Input: Full ranked gene list (for prerank) or Top-N gene sets (for enrichr).
# * Output: Normalized Enrichment Scores (NES), FDR-adjusted p-values, and leading-edge gene sets.
# * Database: Extensive access to MSigDB (via prerank) and 200+ Enrichr libraries (via enrichr).
#
# **Key configuration**: Ensure gene_sets is specified (e.g., "GO_Biological_Process_2023", "KEGG_2021_Human") and permutation_num is set sufficiently high (typically 1000) for stable p-values.

# %%
import gseapy 

# %%
gseapy_db          = "GO_Biological_Process_2023"   # any Enrichr library; same as gsea_db or different
gseapy_permutations = 100                            # increase to 1000 for publication; 100 for exploration
gseapy_top_n        = 30                             # genes for enrichr ORA

# Get all available libraries: gseapy.get_library_name()

# %% [markdown]
# #### Preranked GSEA (Permutation-based)

# %%
gseapy_rows = []
for fac, factor_label, sign, series in iter_factor_directions():
    rnk = series.dropna()
    rnk = rnk[~rnk.index.duplicated(keep="first")]
    try:
        pre = gseapy.prerank(
            rnk=rnk,
            gene_sets=gseapy_db,
            outdir=None,
            permutation_num=gseapy_permutations,
            min_size=15,
            max_size=500,
            seed=42,
            verbose=False,
        )
        sig = pre.res2d[pre.res2d["FDR q-val"] < significance_threshold].sort_values("FDR q-val")
        for _, row in sig.head(3).iterrows():
            gseapy_rows.append({
                "factor": factor_label,
                "term": row["Term"],
                "NES": round(float(row["NES"]), 3),
                "FDR": float(row["FDR q-val"]),
                "leading_edge": row.get("Lead_genes", ""),
            })
    except Exception as e:
        print(f"GSEApy prerank failed for {factor_label}: {e}")

gseapy_results = pd.DataFrame(gseapy_rows)

# %%
print(f"GSEApy (prerank) significant directions: {gseapy_results['factor'].nunique()} / {len(factor_dir_labels)}")

display(gseapy_results.sort_values(["factor", "FDR"]))

# %% [markdown]
# #### Enrichr ORA

# %%
# (fast, top-N gene list, 200+ databases)
enrichr_rows = []
for fac, factor_label, sign, series in iter_factor_directions():
    genes = series.head(gseapy_top_n).index.tolist()
    try:
        enr = gseapy.enrichr(
            gene_list=genes,
            gene_sets=gseapy_db,
            outdir=None,
            cutoff=significance_threshold,
        )
        sig = enr.res2d[enr.res2d["Adjusted P-value"] < significance_threshold].sort_values("Adjusted P-value")
        for _, row in sig.head(3).iterrows():
            enrichr_rows.append({
                "factor": factor_label,
                "term": row["Term"],
                "p_adj": float(row["Adjusted P-value"]),
                "overlap": row.get("Overlap", ""),
            })
    except Exception as e:
        print(f"GSEApy enrichr failed for {factor_label}: {e}")

enrichr_results = pd.DataFrame(enrichr_rows)

# %%
print(f"GSEApy (enrichr ORA) significant results: {enrichr_results['factor'].nunique()} / {len(factor_dir_labels)} ")

display(enrichr_results.sort_values(["factor", "p_adj"]))

# %% [markdown]
# ### 2.3 g:Profiler
#
# [g:Profiler](https://biit.cs.ut.ee/gprofiler/) performs Over-Representation Analysis (ORA) using a hypergeometric test. It employs a custom multiple-testing correction (g:SCS), which is specifically optimized to handle the hierarchical and overlapping structure of Gene Ontology terms.
#
# In **ordered query** mode, g:Profiler processes genes sorted by their DRVI effect scores and iteratively tests enrichment at increasing increments. This approach automatically identifies the optimal gene set size for enrichment, making it more sensitive than using a fixed "top-N" cutoff for continuous latent factor scores.
#
# How it works:
# - **Input**: Ordered gene list (genes sorted by absolute or directional traverse effect scores)
# - **Output**: Enriched terms with p-values corrected via g:SCS
# - **Database**: Comprehensive support for GO (BP, MF, CC), Reactome, KEGG, WikiPathways, and regulatory motifs
#
# Many functional annotation collections (for example Gene Ontology, pathway databases, or phenotype ontologies) are hierarchical and redundant. Broad "umbrella" terms tend to be enriched across multiple latent factors, while more specific child terms capture finer-grained biology. Because of this structure, there is rarely a single automatically chosen term that is clearly "the" correct label for a factor.
#
# In this tutorial, we therefore treat g:Profiler as a tool to obtain **shortlists of enriched terms per factor-direction**, not as an automatic source of single-factor labels.
#
# **Key configuration** (set in the next cell): `organism` (e.g. `"hsapiens"`, `"mmusculus"`) and `gp_source` (e.g. `["GO:BP"]`, `["REAC"]`).

# %%
# Source database(s).
# Common choices: ["GO:BP"], ["GO:MF"], ["GO:CC"], ["REAC"], ["KEGG"], ["HP"]
gp_source = ["GO:BP"]

# %%
from gprofiler import GProfiler

gp = GProfiler(return_dataframe=True)


def run_gprofiler_for_factor(genes, factor_label):
    """Run g:Profiler ordered-query ORA for a single factor-direction."""
    genes = pd.Series(genes).dropna().astype(str).drop_duplicates().tolist()
    if not genes:
        return pd.DataFrame()

    res = gp.profile(
        organism=organism,
        query=genes,
        sources=gp_source,
        ordered=True,
        user_threshold=significance_threshold,
        background=all_genes,
    )
    if res is None or res.empty:
        return pd.DataFrame()

    res = res.copy()
    res["factor"] = factor_label
    return res


# %%
gprofiler_parts = []
for fac, factor_label, sign, series in iter_factor_directions():
    genes = series.index.tolist()
    gprofiler_parts.append(run_gprofiler_for_factor(genes, factor_label))

gprofiler_all = pd.concat(
    [x for x in gprofiler_parts if not x.empty], ignore_index=True
) if any(not x.empty for x in gprofiler_parts) else pd.DataFrame()

sig = gprofiler_all[gprofiler_all["p_value"] < significance_threshold].copy()
print(
    f"g:Profiler results: {len(sig)} significant rows across "
    f"{sig['factor'].nunique()} factor-directions "
    f"(g:SCS-corrected p < {significance_threshold})."
)

# %% [markdown]
# #### Explore top pathway terms per factor-direction

# %%
# Example: "DR 2+", "DR 36-", etc.
factor_to_inspect = "DR 2+"
top_n_gp_terms = 10  # or 5, as you prefer

if not gprofiler_all.empty:
    available = sorted(gprofiler_all["factor"].unique())
    print(f"Available factor-directions ({len(available)}): {available[:10]}{' ...' if len(available) > 10 else ''}\n")

    df_fac = gprofiler_all[
        (gprofiler_all["factor"] == factor_to_inspect)
        & (gprofiler_all["p_value"] < significance_threshold)
    ].copy()

    if df_fac.empty:
        print(f"No significant g:Profiler terms for factor {factor_to_inspect} at this threshold.")
    else:
        df_fac = (
            df_fac.sort_values("p_value")
            [["factor", "name", "p_value", "intersection_size", "term_size"]]
            .rename(columns={"name": "term"})
            .head(top_n_gp_terms)
        )
        print(f"Top {len(df_fac)} g:Profiler terms for {factor_to_inspect}:")
        display(df_fac)
else:
    print("gprofiler_all is empty — run the g:Profiler enrichment cell above first.")

# %% [markdown]
# ### 2.4 decoupler
#
# [decoupler](https://decoupler-py.readthedocs.io/) uses regression-based methods (Univariate/Multivariate Linear Models, z-score) and weighted sums to infer the activity of regulators from gene-level scores. Unlike enrichment-based methods, it models the relationship between observed gene scores and a Prior Knowledge Network (PKN), quantifying the specific influence of a regulator.
#
# decoupler provides access to curated regulatory resources from [OmniPath](https://omnipathdb.org/):
#
# - **CollecTRI**: Comprehensive transcription factor (TF) → target gene interactions, well-suited for discovering TF-level drivers of latent factors.
# - **DoRothEA**: TF regulons categorized by confidence levels (A–D) based on supporting evidence; also TF-centric, with tunable stringency.
# - **PROGENy**: Pathway footprints that infer upstream pathway activity (e.g., Hypoxia, EGFR, TGFb) from downstream responsive genes.
#
# For DRVI latent factor annotation, **CollecTRI and DoRothEA are usually the most informative options**, because latent factors can capture TF-driven gene programs or cell identity signatures. **PROGENy can still be used as an exploratory option**, but in practice it may yield few or no strongly significant hits when factors are not dominated by a small set of canonical signaling pathways (as in this immune example).
#
# Multiple decoupler methods are run sequentially and combined via a consensus step to produce robust p-values.
#
# **Runtime-relevant configuration knobs:**
#
# - **`dc_methods`**: methods run sequentially; dropping `"mlm"` gives ~33% speedup with minimal consensus impact.
# - **`dc_min` / `tmin`**: Minimum number of genes from the gene set that must be present in the data for a valid enrichment test
# - **`dc_geneset`**: network size scales runtime linearly. CollecTRI (~1,185 regulators) is slowest; DoRothEA A-B (~500) is ~2x faster at some coverage cost.

# %%
# Gene set / network to use.
# Recommended options for factor annotation: "collectri", "dorothea".
# PROGENy ("progeny") is more pathway-focused and may give few strong hits
# if latent factors are not dominated by canonical signaling pathways.
dc_geneset = "collectri"  # or "dorothea"

dc_methods = ["ulm", "zscore"]
dc_min = 10 

# %%
import decoupler as dc
from statsmodels.stats.multitest import multipletests

# %%
# Load the prior knowledge network for the selected gene set resource.
net_dispatch = {
    "collectri": lambda: dc.op.collectri(organism=dc_organism),
    "dorothea": lambda: dc.op.dorothea(organism=dc_organism, levels=["A", "B", "C"]),
    "progeny": lambda: dc.op.progeny(organism=dc_organism),
}
net = net_dispatch.get(
    dc_geneset.strip().lower(),
    lambda: dc.op.resource(name=dc_geneset, organism=dc_organism),
)()

cols = ["source", "target"] + (["weight"] if "weight" in net.columns else [])
net = net[cols].dropna().drop_duplicates().reset_index(drop=True)
print(f"Network: {len(net)} interactions, {net['source'].nunique()} regulators")


# %%
def run_decouple(df_factors_by_genes):
    """Run decoupler consensus on a factors x genes score matrix.

    Expects a (factor-directions x genes) DataFrame with rows like "DR 1+", "DR 1-".
    """
    mat = df_factors_by_genes.copy()
    mat.columns = mat.columns.astype(str).str.strip()

    net_use = net.copy()
    net_use["target"] = net_use["target"].astype(str).str.strip()

    if UPPERCASE_GENES:
        mat.columns = mat.columns.str.upper()
        net_use["target"] = net_use["target"].str.upper()

    # Keep only genes present in the network
    targets = net_use["target"].unique()
    mat = mat[[g for g in mat.columns if g in targets]]
    mat = mat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    res = dc.mt.decouple(
        data=mat,
        net=net_use,
        methods=dc_methods,
        cons=False,  # We call consensus() manually to extract p-values
        tmin=dc_min,
        verbose=True,
    )
    _, pvals = dc.mt.consensus(res)

    out = pvals.stack().rename("p_value").reset_index()
    out.columns = ["factor", "term", "p_value"]
    _, p_adj, _, _ = multipletests(out["p_value"].values, method="fdr_bh")
    out["p_adj"] = p_adj
    return out[["factor", "term", "p_value", "p_adj"]]


pos_T = pos_std.T.copy()
pos_T.index = [f"{f}+" for f in pos_T.index]
neg_T = neg_std.T.copy()
neg_T.index = [f"{f}-" for f in neg_T.index]
combined_std = pd.concat([pos_T, neg_T], axis=0)

decoupler_all = run_decouple(combined_std)

decoupler_results = (
    decoupler_all[decoupler_all["p_adj"] < significance_threshold]
    .sort_values("p_adj")
    .groupby("factor", as_index=False)
    .first()
    [["factor", "term", "p_adj"]]
)

# %%
print(
    f"decoupler significant regulators for "
    f"{decoupler_results['factor'].nunique()} / {len(factor_dir_labels)} factor-directions "
    f"(top 1 regulator per direction with FDR < {significance_threshold})"
)
display(decoupler_results.sort_values("p_adj"))

# %% [markdown]
# ### 2.5 gs2txt (LLM-based)
#
# [gs2txt](https://github.com/wuys13/gs2txt) uses large language models to generate concise, biologically meaningful descriptions of gene sets. It intelligently combines gene functions with pathway enrichment results to infer the dominant biological process.
#
# gs2txt first runs pathway enrichment on the input gene set (optional), then combines the gene functions with enrichment results into a structured prompt for the LLM, which produces a concise biological process description.
#
# - **Input**: Gene list with scores (top-N genes by DRVI effect score), formatted as a DataFrame with `gene` and `logFC` columns
# - **Algorithm**:
#   1. *Optional enrichment*: Runs pathway enrichment (via GSEApy) to identify enriched GO/KEGG/Reactome terms
#   2. *Prompt construction*: Combines individual gene functions, enrichment results, and optional user context into a structured prompt
#   3. *LLM annotation*: Sends the prompt to the configured LLM for a concise biological process description
# - **Output**: Natural language description of the dominant biological process (e.g., "DNA damage response and cell cycle regulation")
# - **Supported providers**: OpenAI, Anthropic, LiteLLM or custom.
#
# We configure gs2txt to use Ollama. Custom prompts can be tailored for domain-specific output (e.g., "focus on immune cell biology").

# %%
from gs2txt import GeneSetAnnotator
from gs2txt.llm import OpenAIProvider

# %%
gs2txt_annotator = GeneSetAnnotator(
    llm_provider=OpenAIProvider(
        api_key="ollama",
        model_id=OLLAMA_MODEL,
        temperature=0.1,
        base_url=f"{OLLAMA_URL}/v1",
    ),
    enrichment_method="pathway",
    organism="human",              # must be lowercase
)
print(f"gs2txt configured: {OLLAMA_MODEL} @ {OLLAMA_URL}/v1")


# %%
def run_gs2txt(pos_ranked, neg_ranked):
    rows = []
    for fac, factor_label, sign, series in iter_factor_directions(): 
        genes_series = series.head(llm_top_n_genes)
        deg_df = pd.DataFrame({
            "gene": genes_series.index,
            "logFC": genes_series.values,
        })
        try:
            annotation = gs2txt_annotator.annotate(
                deg_df,
                max_gene_num=llm_top_n_genes,
                additional_context=f"DRVI factor {factor_label} — {llm_tissue_context}",
            )
        except Exception as e:
            annotation = f"Error: {e}"
        rows.append({
            "Factor_ID": fac, "Direction": sign,
            "Tool_Name": "gs2txt", "Model_Used": OLLAMA_MODEL,
            "Predicted_Process": annotation, "Confidence_Score": None,
            "Evidence_Reasoning": "Gene functions + pathway enrichment summarized by LLM",
        })
    return pd.DataFrame(rows).reindex(columns=LLM_COLS)



# %%
gs2txt_results = run_gs2txt(pos_ranked, neg_ranked)
with pd.option_context("display.max_colwidth", None):
    display(gs2txt_results)

# %% [markdown]
# ## 3. Curation Table and Export

# %% [markdown]
# ### 3.1 Curation Table 
#
# This is the final deliverable of the tutorial. We merge evidence from all tools into a single table where each row is one factor-direction. The table includes the top hits from each tool (when significant) and the top 10 genes, helping you to assign a biological label.
#
# **Recommended workflow:**
# 1. Run the cells below to build the table and export a CSV.
# 2. Edit `MANUAL_LABELS` in the helper cell to define your final annotations per factor-direction.
# 3. Re-run the curation and export cells to update the CSV.
# 4. Run the re-import cell to store final annotations in the embedding object.
#

# %%
# Top-N hits per tool in curation table
CURATION_TOP_N = {
    "known_annotation": 1,
    "celltypist": 1,
    "celltype_ora": 3,
    "blitzgsea": 5,
    "gseapy_prerank": 3,
    "gseapy_enrichr": 3,
    "gprofiler": 5,
    "decoupler": 5,
}

# %%
# ─── Base table: all factor-directions with top 10 genes ─────────────────────
curation_rows = []
for fac, factor_label, sign, series in iter_factor_directions():
    top_genes = ", ".join(series.head(10).index.tolist())
    curation_rows.append({"factor": factor_label, "top_genes": top_genes})

curation = pd.DataFrame(curation_rows)

# ─── Helper to build a column from grouped results ───────────────────────────
def build_column(df, factor_col, sort_col, top_n, fmt_fn):
    if df is None or df.empty:
        return pd.Series("", index=curation.index)
    mapping = {}
    for fac, grp in df.sort_values(sort_col).groupby(factor_col):
        top = grp.head(top_n)
        mapping[fac] = " | ".join(fmt_fn(r) for _, r in top.iterrows())
    return curation["factor"].map(mapping).fillna("")

# ─── 1.1 Known annotation SMI ────────────────────────────────────────────────
known_map = {}
if "smi_top_matches" in dir() and len(smi_top_matches):
    for fac, grp in smi_top_matches.sort_values("value", ascending=False).groupby("title"):
        row = grp.head(CURATION_TOP_N["known_annotation"]).iloc[0]
        known_map[fac] = f"{row['variable']} || SMI={row['value']:.2f}"
curation["known_annotation"] = curation["factor"].map(known_map).fillna("")

# ─── 1.2 CellTypist SMI ──────────────────────────────────────────────────────
ct_map = {}
if "ct_significant" in dir() and len(ct_significant):
    for _, row in ct_significant.iterrows():
        ct_map[row["factor"]] = f"{row['cell_type']} || SMI={row['smi']:.2f}"
curation["celltypist"] = curation["factor"].map(ct_map).fillna("")

# ─── 1.3 Cell Type ORA ───────────────────────────────────────────────────────
curation["celltype_ora"] = build_column(
    celltype_ora_results if "celltype_ora_results" in dir() else None,
    "factor", "FDR", CURATION_TOP_N["celltype_ora"],
    lambda r: f"{r['cell_type']} ({r['database']}) || NES={r['NES']:.2f} || FDR={r['FDR']:.2e}"
)

# ─── 2.1 BlitzGSEA ───────────────────────────────────────────────────────────
curation["blitzgsea"] = build_column(
    blitzgsea_results if "blitzgsea_results" in dir() else None,
    "factor", "FDR", CURATION_TOP_N["blitzgsea"],
    lambda r: f"{r['term']} || NES={r['NES']:.2f} || FDR={r['FDR']:.2e}"
)

# ─── 2.2 GSEApy prerank ──────────────────────────────────────────────────────
curation["gseapy_prerank"] = build_column(
    gseapy_results if "gseapy_results" in dir() else None,
    "factor", "FDR", CURATION_TOP_N["gseapy_prerank"],
    lambda r: f"{r['term']} || NES={r['NES']:.2f} || FDR={r['FDR']:.2e}"
)

# ─── 2.2 GSEApy enrichr ORA ──────────────────────────────────────────────────
curation["gseapy_enrichr"] = build_column(
    enrichr_results if "enrichr_results" in dir() else None,
    "factor", "p_adj", CURATION_TOP_N["gseapy_enrichr"],
    lambda r: f"{r['term']} || p_adj={r['p_adj']:.2e}"
)

# ─── 2.3 g:Profiler ──────────────────────────────────────────────────────────
sig_gp = (
    gprofiler_all[gprofiler_all["p_value"] < significance_threshold].copy()
    if "gprofiler_all" in dir() and not gprofiler_all.empty else pd.DataFrame()
)
curation["gprofiler"] = build_column(
    sig_gp, "factor", "p_value", CURATION_TOP_N["gprofiler"],
    lambda r: f"{r['name']} || p={r['p_value']:.2e}"
)

# ─── 2.4 decoupler ───────────────────────────────────────────────────────────
dc_sig = (
    decoupler_all[decoupler_all["p_adj"] < significance_threshold]
    if "decoupler_all" in dir() else pd.DataFrame()
)
curation["decoupler"] = build_column(
    dc_sig, "factor", "p_adj", CURATION_TOP_N["decoupler"],
    lambda r: f"{r['term']} || p_adj={r['p_adj']:.2e}"
)

# ─── LLM cell type (AnnDictionary + CASSIA) ──────────────────────────────────
anndict_ct = (
    {f"{r['Factor_ID']}{r['Direction']}": r["Predicted_Celltype"]
     for _, r in df_anndictionary.iterrows() if r["Predicted_Celltype"]}
    if "df_anndictionary" in dir() else {}
)
curation["anndictionary_celltype"] = curation["factor"].map(anndict_ct).fillna("")

if "cassia_results" in dir():
    cassia_general = {f"{r['Factor_ID']}{r['Direction']}": r["Predicted_General"]
                      for _, r in cassia_results.iterrows() if pd.notna(r["Predicted_General"])}
    cassia_detailed = {f"{r['Factor_ID']}{r['Direction']}": r["Predicted_Detailed"]
                       for _, r in cassia_results.iterrows() if pd.notna(r["Predicted_Detailed"])}
else:
    cassia_general, cassia_detailed = {}, {}

curation["cassia_general"] = curation["factor"].map(cassia_general).fillna("")
curation["cassia_detailed"] = curation["factor"].map(cassia_detailed).fillna("")

# ─── LLM biological process (AnnDictionary + gs2txt) ─────────────────────────
anndict_bp = (
    {f"{r['Factor_ID']}{r['Direction']}": f"AnnDict: {r['Predicted_Process']}"
     for _, r in df_anndictionary.iterrows() if r["Predicted_Process"]}
    if "df_anndictionary" in dir() else {}
)
gs2txt_bp = (
    {f"{r['Factor_ID']}{r['Direction']}": f"gs2txt: {r['Predicted_Process']}"
     for _, r in gs2txt_results.iterrows() if r["Predicted_Process"]}
    if "gs2txt_results" in dir() else {}
)
curation["llm_bioprocess"] = curation["factor"].map(
    lambda f: " | ".join(filter(None, [anndict_bp.get(f), gs2txt_bp.get(f)]))
).replace("", pd.NA).fillna("")

# ─── Manual curation columns ─────────────────────────────────────────────────
curation["manual_label"] = ""
curation["manual_notes"] = ""


# %%
# ─── Display & export ─────────────────────────────────────────────────────────
display(curation)
curation_path = io_dir / "factor_annotation_curation.csv"
curation.to_csv(curation_path, index=False)
print(f"Curation table exported to: {curation_path}")

print("\nEdit MANUAL_LABELS in the helper cell above, then re-run the curation and export cells.")
print("Then run the cells below to finalize and store annotations in embed.var.")

# %% [markdown]
# #### Manual labeling helper
#
# Edit the dictionaries below to define your final annotations. Use factor-direction labels (e.g. `"DR 2+"`, `"DR 36-"`) as keys. After editing, re-run the curation table cell above and the export cell to update the CSV with your labels.

# %%
# Define your final annotations here. Keys are factor-direction labels (e.g. "DR 2+", "DR 36-").
# Example: MANUAL_LABELS = {"DR 2+": "T cell activation", "DR 36-": "Monocyte differentiation"}
MANUAL_LABELS = {"DR 36+": "Monocyte progenitor"}

# Optional: free-text notes per factor-direction
MANUAL_NOTES = {}


def set_label(factor, label, notes=None):
    """Helper to add or update a label for a factor-direction."""
    MANUAL_LABELS[factor] = label
    if notes is not None:
        MANUAL_NOTES[factor] = notes
    print(f"Set {factor} -> {label}")


# %% [markdown]
# #### Re-import and finalize
#
# Re-import the curation CSV and store final annotations in the embedding object. The `final_label` is taken **only** from `manual_label` (as populated from `MANUAL_LABELS` in the helper cell). No automatic fallback to tool-based columns.

# %%
curation_edited = pd.read_csv(curation_path)

# Apply manual labels from MANUAL_LABELS (overrides any CSV edits)
curation_edited["manual_label"] = curation_edited["factor"].map(MANUAL_LABELS).fillna("")

curation_edited["final_label"] = curation_edited["manual_label"].str.strip()
curation_edited["label_source"] = curation_edited["final_label"].apply(
    lambda x: "manual" if x else "none"
)

display(curation_edited[["factor", "final_label", "label_source"]].head(20))

# %%
# Store annotations in embed.var for persistence
# Map factor-direction labels back to factor base names for embed.var
# We take the "+" direction label as the primary annotation per factor
annot_by_factor = {}
for _, row in curation_edited.iterrows():
    fac_base = row["factor"][:-1]  # strip +/- suffix
    direction = row["factor"][-1]
    if direction == "+" and row["final_label"]:
        annot_by_factor[fac_base] = (row["final_label"], row["label_source"])
    elif fac_base not in annot_by_factor and row["final_label"]:
        annot_by_factor[fac_base] = (row["final_label"], row["label_source"])

embed.var["annotation_final"] = embed.var["title"].map(
    lambda t: annot_by_factor.get(t, ("", ""))[0]
)
embed.var["annotation_source"] = embed.var["title"].map(
    lambda t: annot_by_factor.get(t, ("", ""))[1]
)

print("Annotations stored in embed.var:")
display(embed.var[["title", "vanished", "annotation_final", "annotation_source"]].head(20))

# %%
embed.write_h5ad(embed_path)
print(f"Updated embedding saved to: {embed_path}")

# %% [markdown]
# ## 4. Visual validation
#
# Sanity-check a few annotated factors by visualizing their activity on the UMAP alongside the preannotated cell type.
#
# > These plots require a per-cell annotation column in `adata.obs` (configured as `annot_col`, e.g. `final_annotation`). If your dataset does not have such annotations, you can skip this section.

# %%
example_dims = ["DR 36"]  # Replace with factor-directions you want to visualize

if annot_col is not None and example_dims:
    drvi_factors_df = pd.DataFrame(embed_nv.X, index=embed_nv.obs_names, columns=factor_ids)
    for dim in example_dims:
        if dim in drvi_factors_df.columns:
            adata.obs["_factor_check"] = drvi_factors_df[dim].reindex(adata.obs_names).values
            annot = embed.var.set_index("title").loc[dim, "annotation_final"]
            print(f"\n{dim} — annotated as: {annot}")

            sc.pl.umap(
                adata,
                color=["_factor_check", annot_col],
                ncols=2,
                frameon=False,
                title=[f"Factor: {dim}", annot_col],
            )


# %% [markdown]
# ## Appendix: Database reference
#
# The table below lists curated databases available for factor annotation, organized by domain. You can swap any of the tool-specific config variables above (e.g., `gsea_db`, `gp_source`, `dc_geneset`) to use different databases.
#
# ### Cell type marker databases (BlitzGSEA ORA only)
#
# | Database | Description | BlitzGSEA name |
# |----------|-------------|----------------|
# | CellMarker 2.0 | Curated cell type markers from literature | `CellMarker_2024` |
# | PanglaoDB | Curated markers from scRNA-seq studies | `PanglaoDB_Augmented_2021` |
# | ARCHS4 Cell Lines | Gene expression signatures from cell lines | `ARCHS4_Cell-lines` |
# | Tabula Muris | Cell type markers from single-cell mouse atlas | `Tabula_Muris` |
# | Tabula Sapiens | Cell type markers from single-cell human atlas | `Tabula_Sapiens` |
# | Human Gene Atlas | Tissue/cell type expression profiles | `Human_Gene_Atlas` |
#
#
#
# ### Biological process databases
#
# | Database | Description | BlitzGSEA| g:Profiler | 
# |----------|-------------|-------------------|------------|
# | MSigDB Hallmark | 50 well curated non redundant biological states | `MSigDB_Hallmark_2020` | — |
# | GO Biological Process | Comprehensive hierarchical processes | `GO_Biological_Process_2025` | `GO:BP` |
# | GO Cellular Component | Subcellular localization | `GO_Cellular_Component_2025` | `GO:CC` |
# | GO Molecular Function | Molecular activities | `GO_Molecular_Function_2025` | `GO:MF` |
# | Reactome | Detailed, reaction-based pathways | `Reactome_Pathways_2024` | `REAC` |
# | KEGG | Classic metabolic/signaling maps | `KEGG_2026` | `KEGG` | 
# | WikiPathways | Community-curated biological maps | `WikiPathways_2024_Human` | `WP` |
#
# ### Regulatory networks (decoupler only)
#
# | Network | Description | decoupler name | Notes|
# |---------|-------------|---------------|---------|
# | CollecTRI | TF → target gene regulons| `collectri` | Recommended for identifying TF drivers of a factor |
# | PROGENy | Pathway-responsive genes signatures | `progeny` | Best for signaling (TGFb, MAPK, etc.) activity |
# | DoRothEA |  TF → target gene interactions | `dorothea` |  Curated resource for Transcription Factor (TF) activity; uses confidence levels (A-D) |
#
# ### Clinical and Disease Phenotypes
#
# | Database | Description | BlitzGSEA| g:Profiler |
# |----------|-------------|-------------------|------------|
# | Human Phenotype Ont. | Genes linked to clinical signs | Human_Phenotype_Ontology | HP |
# | OMIM | Human genes and genetic disorders | OMIM_Disease | OMIM
#
