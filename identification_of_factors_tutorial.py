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
#     display_name: python_apptainer
#     language: python
#     name: python_apptainer
# ---

# %% [markdown]
# # Identification of factors

# %% [markdown]
# In this notebook, we use some supervised information and enrichment tools on an already trained DRVI model on the immune dataset to identify biological processes captured by latent factors. We combine multiple complementary strategies:
#
# 1. **Cell type annotation** — match factors to known cell types. For that you can use:
#     * Existing labels
#     * Pre-trained classifiers (e.g. CellTypist)
# 3. **Annotation of Biological Processes** using
#     * **Gene set enrichment analysis (GSEA)** — identify enriched pathways from ranked gene lists (BlitzGSEA)
#     * **Over-representation analysis (ORA)** — test for enriched gene sets using ordered queries (g:Profiler)
#     * **Regulator activity inference** — infer transcription factor or pathway activity using a statistical framework (decoupler) integrated with prior knowledge 
#
# **We always advise examination by a biologist and validation against published literature for any identified processes.**

# %% [markdown]
# ## Intro
#
# This notebook assumes that you have already trained a DRVI model and computed the interpretability scores (via `model.calculate_interpretability_scores` in the general pipeline).
#
# Please refer to the [General training and interpretability pipeline](./general_pipeline.html) tutorial.
#
# While we use the immune dataset as a running example, all code is dataset-agnostic. Configuration variables at the top of each section indicate what to change for your own data.

# %% [markdown]
# ### Adapting this notebook to your own dataset
#
# To reuse this notebook on a different dataset or DRVI model:
#
# - Update the `0. Load a previously trained model` section accorfingly.
#   - Update `io_dir` to point to your project directory.
#   - Make sure the following files exist under `io_dir` with your data (or change the code):
#     - `immune_all.h5ad` (or your equivalent full-gene data) in the parent directory.
#     - `adata_preprocesses.h5ad` (preprocessed AnnData with HVGs and UMAP)
#     - `drvi_model/` (trained DRVI model directory)
#     - `embed.h5ad`
#
# ### Config Overview
# - The notebook contains 5 independent ways to annotate processes. Each having its own config. Some important configs are:
#   - **Section 1.1** For identification based on user annotations you have to set `annot_col` to the corresponding column in `adata.obs`.
#   - **Section 1.2** For identification based on a pre-trained model. If you use CellTypist you have to choose a model via `ct_model` that matches your tissue / species (e.g. `"Immune_All_Low.pkl"` for PBMC, `"Developing_Mouse_Brain.pkl"` for mouse brain)
#   - **Section 2.1** For BlitzGSEA, set:
#     - `gsea_db` (e.g `"GO_Biological_Process_2023"`)
#     - `fdr_threshold` to keep significant results.
#   - **Section 2.2** For g:Profiler, set:
#     - `organism` (e.g. `"hsapiens"`, `"mmusculus"`).
#     - `gp_source` to the GO / pathway collections you care about (e.g. `["GO:BP"]`, `["REAC"]`).
#     - `pval_threshold` to keep significant results.
#   - **Section 2.3** For decoupler, set:
#     - `dc_geneset` (e.g. `"collectri"`, `"dorothea"`, `"progeny"` or another resource name).
#     - `dc_organism` to match your species (e.g. `"human"`, `"mouse"`).
#     - `fdr_threshold` to keep significant results.
#

# %% [markdown]
# ## Requirements
#
# This notebook requires the following python packages:
# ```
# celltypist
# blitzgsea
# gprofiler-official
# decoupler
# ```

# %% [markdown]
# ## Contact
#
# For questions and help requests, you can reach out in the [scverse discourse](https://discourse.scverse.org/).
#
# If you found a bug, please use the [issue tracker](https://github.com/theislab/drvi/issues).

# %% [markdown]
# ## Install
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
                           "celltypist", "blitzgsea", "gprofiler-official", "decoupler"])

# %% [markdown]
# ## Imports

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

import scvi
import drvi
from pathlib import Path
from drvi.model import DRVI
from drvi.utils.misc import hvg_batch

# %%
print("Last run with scvi-tools version:", scvi.__version__)
print("Last run with DRVI version:", drvi.__version__)

# %%
# Plot defaults
sc.settings.set_figure_params(dpi=100, frameon=False)
sc.set_figure_params(figsize=(3, 3))
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (3, 3)

# %% [markdown]
# ## 0. Load a previously trained model

# %% [markdown]
# ### Config

# %%
# Set input output directory
# We use tmp_io/ directory in the same place as this notebook. Update accordingly.
io_dir = Path("./tmp_io/drvi_immune_128").resolve()
print(f"Using directory: {io_dir}")

# %% [markdown]
# ### Load Data

# %%
raw_data_path = io_dir.parent / "immune_all.h5ad"
adata_full = sc.read_h5ad(raw_data_path)
adata_full

# %%
# The adata which DRVI model is trained on (HVG selected)
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
# ## 1. Identifying Cell-Type Specific Processes
#
# Some latent factors capture cell-type identity. We can identify these using:
# - **Known annotations**: This way we measure alignment between factors and annotated cell types via a similarity measurement like Scaled Mutual Information (SMI). We note that DRVI dimensions may be more fine grained than these annotations, resemble a process covering multiple cell types, or showing general shared processes. However, this approach allows identification of large proportion of processes. For this we need some supervised info such as:
#   - **User annotations:** Where user has annotated its data.
#   - **Annotation tools**: User can also use pre-trained models such as CellTypist or Foundational Models to classify cells.
# - **Enrichment**: Using GSEA/ORA methods with Cell Type databases. This is not described in this tutorial in detail. but same models in **Biological Process Identification** section can be used.

# %% [markdown]
# ### 1.1 Identification based on user annotations
#
# If your dataset has existing cell type annotations, Scaled Mutual Information (SMI) measures how well each latent factor aligns with each annotated category. SMI is normalized to [0, 1], where 1 indicates perfect correspondence between a factor and a cell type.
#
# **Skip this section if your dataset does not have cell type annotations.**

# %% [markdown]
# ####  Imports

# %%
import networkx as nx
from drvi.utils.metrics import DiscreteDisentanglementBenchmark

# %% [markdown]
# #### Config
#
# In this dataset we have annotations stored in `adata.obs["final_annotation"]`.
#

# %%
# Column in adata.obs containing cell type labels. Set to None if not available.
annot_col = "final_annotation" 

# Minimum SMI score between factor and cell-type probability profiles to consider a factor as associated with a cell type. Adjust as needed.
smi_threshold = 0.7

# %% [markdown]
# #### Visualize with a Heatmap

# %%
drvi.utils.pl.plot_latent_dims_in_heatmap(
    embed, 
    annot_col, 
    title_col="title", 
    sort_by_categorical=True,
)

# %% [markdown]
# We observe very good one-to-one relationship between some factors and cell types. Let's identify them

# %% [markdown]
# #### Prepare data
#
# We prepare a matrix where columns are positive and negative nonvanished latent factors and rows are cells

# %%
embed_pos = embed[:, ~embed.var['vanished_positive_direction']].copy()
embed_neg = embed[:, ~embed.var['vanished_negative_direction']].copy()
embed_pos.var.index = embed_pos.var['title'] + '+'
embed_neg.var.index = embed_neg.var['title'] + '-'
embed_pos.X = embed_pos.X.clip(min=0)
embed_neg.X = -embed_neg.X.clip(max=0)
embed_directional_df = pd.concat([embed_pos.to_df(), embed_neg.to_df()], axis=1).loc[embed.obs.index]
embed_directional_df[:3]

# %% [markdown]
# #### Calculation of Scaled Mutual Information
#
# We use `DiscreteDisentanglementBenchmark` class that we use for evaluation of models. This class calculated pairwise similarity function between latent factors and supervised targets.

# %%
# Compute SMI between each factor-direction (+/-) and each annotated cell type.
benchmark = DiscreteDisentanglementBenchmark(
    embed_directional_df.values,
    dim_titles=embed_directional_df.columns,
    discrete_target=embed.obs[annot_col],
    metrics=["SMI-disc"],
    aggregation_methods=[],
)
benchmark.evaluate()
smi_similarity = benchmark.get_results_details()["SMI-disc"]
smi_similarity.index.name = "title"

# %%
print(f"SMI matrix shape: {smi_similarity.shape} (factor-directions x cell types)")
smi_similarity[:3]

# %%
# Reshape the SMI matrix from wide to long format, then keep only pairs above the threshold.
smi_top_matches = (
    smi_similarity.reset_index()
    .melt(id_vars="title", value_vars=smi_similarity.columns)
    .query("value >= @smi_threshold")
    .reset_index(drop=True)
    .sort_values("value", ascending=False)
)
print(f"Factor–cell type pairs with SMI >= {smi_threshold}: {len(smi_top_matches)}")
smi_top_matches


# %% [markdown]
# #### Helper function for relationship Visualization

# %%
def plot_packed_network(df, title_col="title", var_col="variable", val_col="value", figsize=(14, 10)):
    """Visualizes factor–cell type associations as a network with edge weights."""
    G = nx.from_pandas_edgelist(df, title_col, var_col, edge_attr=val_col)

    pos = {}
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    cols = 3
    for i, nodes in enumerate(components):
        sub_pos = nx.spring_layout(G.subgraph(nodes), weight=val_col, k=0.5, seed=42)
        r, c = divmod(i, cols)
        for n, (x, y) in sub_pos.items():
            pos[n] = (x + c * 3, y - r * 3)

    plt.figure(figsize=figsize)
    titles = set(df[title_col])
    weights = [d[val_col] for u, v, d in G.edges(data=True)]
    nx.draw(
        G, pos,
        with_labels=True, font_size=8, font_weight="bold", node_size=600,
        node_color=["#A0CBE2" if n in titles else "#FF9E9E" for n in G.nodes()],
        width=[w * 4 for w in weights],
        edge_color=weights, edge_cmap=plt.cm.Oranges, alpha=0.6,
    )
    edge_labels = {(u, v): f"{d[val_col]:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.axis("off")
    plt.show()


# %% [markdown]
# #### Plot Packed Visualization

# %%
plot_packed_network(smi_top_matches, figsize=(20, 20))

# %% [markdown]
# #### Store summary in var

# %%
first_match = smi_top_matches.drop_duplicates(subset=['title'])
first_match['direction'] = first_match['title'].str[-1:]
first_match['title'] = first_match['title'].str[:-1]
first_match_pos = first_match.query("direction == '+'")
first_match_neg = first_match.query("direction == '-'")
embed.var.set_index("title", drop=False, inplace=True)
embed.var[f'positive_direction_match_with_{annot_col}'] = None
embed.var[f'negative_direction_match_with_{annot_col}'] = None
embed.var[f'positive_direction_match_with_{annot_col}'][first_match_pos['title']] = first_match_pos['variable']
embed.var[f'negative_direction_match_with_{annot_col}'][first_match_neg['title']] = first_match_neg['variable']
embed.var.index = embed.var["original_dim_id"].astype(int).astype(str)
embed.var.index.name = None

(
    embed.var[f'positive_direction_match_with_{annot_col}'].unique(),
    embed.var[f'negative_direction_match_with_{annot_col}'].unique(),
)

# %% [markdown]
# #### Store results in uns

# %%
embed.uns[f'best_smi_matching_{annot_col}_results'] = smi_top_matches

# %%

# %% [markdown]
# ### 1.2 Identification based on pre-trained model
#
# Here, we use [CellTypist](https://www.celltypist.org/) as a pre-trained logistic regression model trained on large-scale annotated atlases to classify individual cells. We calculate the Similarity Mutual Information (SMI) between the CellTypist probability matrix (cells × cell types) and the DRVI factor activity matrix (cells × factors) to identify which factors correspond to which cell types.
#
# **Note 1: Users can skip this if they already have good annotations.**
#
# **Note 2: Only the annotation part is different from the previous section and everything is duplicate from `prepare data` section.**

# %% [markdown]
# #### Imports

# %%
import celltypist
import networkx as nx
from drvi.utils.metrics import DiscreteDisentanglementBenchmark

# %% [markdown]
# #### Config

# %%
# Minimum SMI score between factor and cell-type probability profiles to consider a factor as associated with a cell type. Adjust as needed.
smi_threshold = 0.7

# We use "celltypist_majority" from cell typisy outputs
annot_col = "celltypist_majority"

# %%
# CellTypist Model

# Run celltypist.models.models_description() to see all available models. Choose one matching your tissue. 
ct_model = "Immune_All_Low.pkl"  # e.g., "Developing_Mouse_Brain.pkl" for mouse brain
celltypist.models.download_models(force_update=False, model=ct_model)

ct_model = celltypist.models.Model.load(model=ct_model)
# Run print(ct_model.cell_types) to see available cell types


# %% [markdown]
# #### CellTypist Annotation

# %% [markdown]
# Each cell receives a predicted label via logistic regression based on its transcriptomic profile. Setting majority_voting=True refines these labels by assigning the most frequent label within a cell's local neighborhood (kNN), reducing technical noise. The resulting per-cell labels are stored in adata.obs.

# %%
# We use full anndata (all genes) see Ce;;Typist docs for more info
adata_full.X = adata_full.layers['counts'].copy()
sc.pp.normalize_total(adata_full, target_sum=1e4)
sc.pp.log1p(adata_full)

predictions = celltypist.annotate(adata_full, model=ct_model, majority_voting=True)

# %%
embed.obs["celltypist_labels"] = predictions.predicted_labels["predicted_labels"].loc[embed.obs.index]
embed.obs["celltypist_majority"] = predictions.predicted_labels["majority_voting"].loc[embed.obs.index]
embed.obsm["celltypist_probs"] = predictions.probability_matrix[embed.obs["celltypist_majority"].cat.categories].loc[embed.obs.index]

# %% [markdown]
# #### Visualize with a Heatmap

# %%
drvi.utils.pl.plot_latent_dims_in_heatmap(
    embed, 
    annot_col, 
    title_col="title", 
    sort_by_categorical=True,
)

# %% [markdown]
# We observe very good one-to-one relationship between some factors and cell types. Let's identify them

# %% [markdown]
# #### Prepare data
#
# We prepare a matrix where columns are positive and negative nonvanished latent factors and rows are cells

# %%
embed_pos = embed[:, ~embed.var['vanished_positive_direction']].copy()
embed_neg = embed[:, ~embed.var['vanished_negative_direction']].copy()
embed_pos.var.index = embed_pos.var['title'] + '+'
embed_neg.var.index = embed_neg.var['title'] + '-'
embed_pos.X = embed_pos.X.clip(min=0)
embed_neg.X = -embed_neg.X.clip(max=0)
embed_directional_df = pd.concat([embed_pos.to_df(), embed_neg.to_df()], axis=1).loc[embed.obs.index]
embed_directional_df[:3]

# %% [markdown]
# #### Calculation of Scaled Mutual Information
#
# We use `DiscreteDisentanglementBenchmark` class that we use for evaluation of models. This class calculated pairwise similarity function between latent factors and supervised targets.

# %%
# Compute SMI between each factor-direction (+/-) and each cell type probability.
benchmark = DiscreteDisentanglementBenchmark(
    embed_directional_df.values,
    dim_titles=embed_directional_df.columns,
    discrete_target=embed.obs[annot_col],
    metrics=["SMI-disc"],
    aggregation_methods=[],
)
benchmark.evaluate()
smi_similarity = benchmark.get_results_details()["SMI-disc"]
smi_similarity.index.name = "title"

# %%
print(f"SMI matrix shape: {smi_similarity.shape} (factor-directions x cell types)")
smi_similarity[:3]

# %%
# Reshape the SMI matrix from wide to long format, then keep only pairs above the threshold.
smi_top_matches = (
    smi_similarity.reset_index()
    .melt(id_vars="title", value_vars=smi_similarity.columns)
    .query("value >= @smi_threshold")
    .reset_index(drop=True)
    .sort_values("value", ascending=False)
)
print(f"Factor–cell type pairs with SMI >= {smi_threshold}: {len(smi_top_matches)}")
smi_top_matches


# %% [markdown]
# #### Helper function for relationship Visualization

# %%
def plot_packed_network(df, title_col="title", var_col="variable", val_col="value", figsize=(14, 10)):
    """Visualizes factor–cell type associations as a network with edge weights."""
    G = nx.from_pandas_edgelist(df, title_col, var_col, edge_attr=val_col)

    pos = {}
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    cols = 3
    for i, nodes in enumerate(components):
        sub_pos = nx.spring_layout(G.subgraph(nodes), weight=val_col, k=0.5, seed=42)
        r, c = divmod(i, cols)
        for n, (x, y) in sub_pos.items():
            pos[n] = (x + c * 3, y - r * 3)

    plt.figure(figsize=figsize)
    titles = set(df[title_col])
    weights = [d[val_col] for u, v, d in G.edges(data=True)]
    nx.draw(
        G, pos,
        with_labels=True, font_size=8, font_weight="bold", node_size=600,
        node_color=["#A0CBE2" if n in titles else "#FF9E9E" for n in G.nodes()],
        width=[w * 4 for w in weights],
        edge_color=weights, edge_cmap=plt.cm.Oranges, alpha=0.6,
    )
    edge_labels = {(u, v): f"{d[val_col]:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.axis("off")
    plt.show()


# %% [markdown]
# #### Plot Packed Visualization

# %%
plot_packed_network(smi_top_matches, figsize=(20, 20))

# %% [markdown]
# #### Store summary in var

# %%
first_match = smi_top_matches.drop_duplicates(subset=['title'])
first_match['direction'] = first_match['title'].str[-1:]
first_match['title'] = first_match['title'].str[:-1]
first_match_pos = first_match.query("direction == '+'")
first_match_neg = first_match.query("direction == '-'")
embed.var.set_index("title", drop=False, inplace=True)
embed.var[f'positive_direction_match_with_{annot_col}'] = None
embed.var[f'negative_direction_match_with_{annot_col}'] = None
embed.var[f'positive_direction_match_with_{annot_col}'][first_match_pos['title']] = first_match_pos['variable']
embed.var[f'negative_direction_match_with_{annot_col}'][first_match_neg['title']] = first_match_neg['variable']
embed.var.index = embed.var["original_dim_id"].astype(int).astype(str)
embed.var.index.name = None

(
    embed.var[f'positive_direction_match_with_{annot_col}'].unique(),
    embed.var[f'negative_direction_match_with_{annot_col}'].unique(),
)

# %% [markdown]
# #### Store results in uns

# %%
embed.uns[f'best_smi_matching_{annot_col}_results'] = smi_top_matches

# %%

# %% [markdown]
# ## 2. Biological process identification
#
# Factors that do not map to a single cell type often capture biological processes (e.g., interferon response, cell cycle, stress). We use three complementary enrichment approaches, each with different strengths:
#
# | Tool | Method | Input | Strengths |
# |------|--------|-------|-----------|
# | **BlitzGSEA** | Pre-ranked GSEA | Full ranked gene list | Fast; uses entire ranking; uses an analytical null distribution |
# | **g:Profiler** | Over-representation (ORA) | Ordered gene query | Robust multiple-testing (g:SCS); well-suited for biological pathways and GO terms |
# | **decoupler** | Activity Inference (ULM/MLM) | Gene score matrix + Prior Knowledge | Regression-based; identifies specific regulatory drivers (e.g., TFs) using curated networks |

# %% [markdown]
# Each tool operates on the gene-level interpretability scores produced by DRVI's built-in scoring API (`model.get_interpretability_scores`). DRVI provides two complementary scoring approaches that can be selected via config variables later:
#
# - **OOD (Out-of-Distribution)**: Uses decoder reconstructions to calculate per-gene effect scores. This is our suggested method to consider for finding cell-types and most specific genes of a program. This is stored with `"OOD_combined"` key.
# - **IND (withIN-Distribution)**: Iterates over all cells to compute weighted mean effects. Captures broader mechanistic effects including shared genes. This is stored with `"IND_linear_weighted_mean"` key.
#
# All tools in this notebook are **guiding tools**: they summarize large gene-level patterns into interpretable scores, but they do **not** provide definitive labels. Their outputs should always be interpreted in context, compared across methods, and validated against known biology and the original data.

# %%

# %% [markdown]
# ### 2.1 BlitzGSEA
#
# [BlitzGSEA](https://github.com/MaayanLab/blitzgsea) performs pre-ranked Gene Set Enrichment Analysis using an analytical approximation of the null distribution rather than permutations, enabling high-performance enrichment testing across many factors.
#
# - **Input**: Full ranked gene list (genes sorted by their DRVI effect scores, capturing the magnitude and direction of expression change)
# - **Output**: Normalized Enrichment Score (NES) and FDR-adjusted p-values per gene set
# - **Database**: Compatible with any standard .gmt file or Enrichr library (e.g., MSigDB, Reactome)

# %%
import blitzgsea as blitz

# %% [markdown]
# #### Config

# %%
# Enrichr library to use. See Appendix for available databases.
# Common choices: "MSigDB_Hallmark_2020", "GO_Biological_Process_2023",
#                 "Reactome_2022", "KEGG_2021_Human"

gsea_db = "GO_Biological_Process_2023"

# Interpretability method
# "OOD_combined" — Out-of-distribution: uses decoder reconstructions (more specific genes, shared genes are penalized)
# "IND_linear_weighted_mean" — Within-distribution: iterates over all cells (captures broader effects and includes shared genes)

score_key = "OOD_combined"

# Significance threshold
fdr_threshold = 0.05

print(f"GSEA DB: {gsea_db}")
print(f"Score key: {score_key}")
print(f"FDR threshold: {fdr_threshold}")

# %%
signature_lib = blitz.enrichr.get_library(gsea_db)
print(f"Loaded {gsea_db}: {len(signature_lib)} gene sets")

# %% [markdown]
# #### Prepare data

# %%
# once again: you can set score_key in main config of the notebook
scores_df = model.get_interpretability_scores(embed, adata, key=score_key)
scores_df

# %% [markdown]
# #### Enrichment

# %%
blitzgsea_rows = []

for factor_label in scores_df.columns:
    series = scores_df[factor_label]

    # BlitzGSEA expects a DataFrame with columns "i" (gene) and "v" (score)
    signature = series.rename("v").reset_index().rename(columns={"index": "i"})
    signature["v"] = pd.to_numeric(signature["v"], errors="coerce")
    signature = signature.replace([np.inf, -np.inf], np.nan).dropna(subset=["v"])

    try:
        res = blitz.gsea(signature, signature_lib, processes=4)
        sig = res[res["fdr"] < fdr_threshold].sort_values("fdr")
        if len(sig):
            # Keep up to the top 3 most significant terms per factor-direction
            top_sig = sig.head(3)
            for term, row in top_sig.iterrows():
                blitzgsea_rows.append({
                    "factor": factor_label,
                    "term": term,
                    "NES": round(float(row["nes"]), 3),
                    "FDR": float(row["fdr"]),
                })
    except Exception as e:
        print(f"BlitzGSEA failed for {factor_label}: {e}")

# %%
blitzgsea_results = pd.DataFrame(blitzgsea_rows)
print(
    f"BlitzGSEA significant directions: {blitzgsea_results['factor'].nunique()} / {scores_df.shape[1]} "
    f"(with up to 3 terms per direction)"
)
blitzgsea_results.sort_values(["factor", "FDR"])

# %% [markdown]
# #### Store results

# %%
# dtype conversion is to be able to write as h5ad
embed.uns[f'blitzgsea_{gsea_db}_results'] = blitzgsea_results.convert_dtypes(convert_integer=False, convert_floating=False)

# %%

# %% [markdown]
# ### 2.2 g:Profiler
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
from gprofiler import GProfiler

# %% [markdown]
# #### Config

# %%
# Organism string. Common values: "hsapiens", "mmusculus", "drerio"
organism = "hsapiens"

# Source database(s).
# Common choices: ["GO:BP"], ["GO:MF"], ["GO:CC"], ["REAC"], ["KEGG"], ["HP"]
gp_source = ["GO:BP"]

# Interpretability method
# "OOD_combined" — Out-of-distribution: uses decoder reconstructions (more specific genes, shared genes are penalized)
# "IND_linear_weighted_mean" — Within-distribution: iterates over all cells (captures broader effects and includes shared genes)

score_key = "OOD_combined"
# Gene-set filtering criteria
drvi_score_cutoff = 0.5  # Practically 0.1 for OOD and 0.5 for IND should be good values
drvi_n_top_genes = 100  # maximum top genes per program to consider

# Significance threshold
pval_threshold = 0.05

print(f"Organism: {organism}")
print(f"GP Sources: {gp_source}")
print(f"Score key: {score_key}")
print(f"Cutoff - min score : {drvi_score_cutoff}")
print(f"Cutoff - maximum # of top genes: {drvi_n_top_genes}")
print(f"P-val culoff threshold: {pval_threshold}")

# %% [markdown]
# #### Prepare data

# %%
# once again: you can set score_key in main config of the notebook
scores_df = model.get_interpretability_scores(embed, adata, key=score_key)
scores_df

# %% [markdown]
# #### Enrichment

# %%
gp = GProfiler(return_dataframe=True)

def run_gprofiler_for_factor(genes, background, factor_label):
    """Run g:Profiler ordered-query ORA for a single factor-direction."""
    genes = pd.Series(genes).dropna().astype(str).drop_duplicates().tolist()
    if not genes:
        return pd.DataFrame()

    res = gp.profile(
        organism=organism,
        query=genes,
        sources=gp_source,
        ordered=True,
        user_threshold=pval_threshold,
        background=background,
    )
    if res is None or res.empty:
        return pd.DataFrame()

    res = res.copy()
    res["factor"] = factor_label
    return res


# %%
gprofiler_parts = []

for factor_label in scores_df.columns:
    series = scores_df[factor_label].copy().sort_values(ascending=False)
    top_genes = series[series > drvi_score_cutoff][:drvi_n_top_genes].index.to_list()
    if len(top_genes) < 3:
        continue
    background = series.index.to_list()
    gprofiler_parts.append(run_gprofiler_for_factor(top_genes, background, factor_label))

# %%
gprofiler_results = pd.concat(
    [x for x in gprofiler_parts if not x.empty], ignore_index=True
) if any(not x.empty for x in gprofiler_parts) else pd.DataFrame()
gprofiler_results = gprofiler_results[gprofiler_results["p_value"] < pval_threshold].copy()

print(
    f"Profiler significant directions: {gprofiler_results['factor'].nunique()} / {scores_df.shape[1]}"
)
gprofiler_results['parents'] = gprofiler_results['parents'].astype(str)
gprofiler_results.sort_values(["factor", "p_value"])

# %% [markdown]
# #### Store results

# %%
embed.uns[f'gprofiler_results'] = gprofiler_results.convert_dtypes(convert_integer=False, convert_floating=False)

# %%

# %% [markdown]
# ### 2.3 decoupler
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
import decoupler as dc
from statsmodels.stats.multitest import multipletests

# %% [markdown]
# #### Config

# %%
# Gene set / network to use.
# Recommended options for factor annotation: "collectri", "dorothea".
# PROGENy ("progeny") is more pathway-focused and may give few strong hits
# if latent factors are not dominated by canonical signaling pathways.
dc_geneset = "collectri"  # or "dorothea"

# Organism. Must match ORGANISM above: "human" for hsapiens, "mouse" for mmusculus
dc_organism = "human"

dc_methods = ["ulm", "zscore"]
dc_min = 10 

# Interpretability method
# "OOD_combined" — Out-of-distribution: uses decoder reconstructions (more specific genes, shared genes are penalized)
# "IND_linear_weighted_mean" — Within-distribution: iterates over all cells (captures broader effects and includes shared genes)

score_key = "OOD_combined"
# Gene-set filtering criteria
drvi_score_cutoff = 0.05  # Practically 0.01 for OOD and 0.05 for IND should be good values. Values below this will be zerod.

# Significance threshold
fdr_threshold = 0.05

print(f"Organism: {dc_organism}")
print(f"Gene-set: {dc_geneset}")
print(f"Score key: {score_key}")
print(f"FDR threshold: {fdr_threshold}")

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

# %% [markdown]
# #### Prepare data

# %%
# once again: you can set score_key in main config of the notebook
scores_df = model.get_interpretability_scores(embed, adata, key=score_key)
scores_df


# %% [markdown]
# #### Run

# %%
def run_decouple(df_factors_by_genes):
    """Run decoupler consensus on a factors x genes score matrix.

    Expects a (factor-directions x genes) DataFrame with rows like "DR 1+", "DR 1-".
    """
    mat = df_factors_by_genes.copy()
     # Standardize gene names to uppercase to match the PKN, and keep only genes present in the network.
    mat.columns = mat.columns.astype(str).str.strip().str.upper()
    targets = net["target"].astype(str).str.strip().str.upper().unique()
    keep_cols = [g for g in mat.columns if g in targets]
    mat = mat[keep_cols]
    mat = mat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    net_use = net.copy()
    net_use["target"] = net_use["target"].astype(str).str.strip().str.upper()

    res = dc.mt.decouple(
        data=mat,
        net=net_use,
        methods=dc_methods,
        cons=False,
        tmin=dc_min,
        verbose=True,
    )
    _, pvals = dc.mt.consensus(res)

    out = pvals.stack().rename("p_value").reset_index()
    out.columns = ["factor", "term", "p_value"]

    _, p_adj, _, _ = multipletests(out["p_value"].values, method="fdr_bh")
    out["p_adj"] = p_adj
    return out[["factor", "term", "p_value", "p_adj"]]


# %%
input_df = scores_df.copy()
input_df[input_df < drvi_score_cutoff] = 0
decoupler_all = run_decouple(input_df.T)

# %%
# Keep the most significant regulator per factor-direction for a summary view.
decoupler_results = (
    decoupler_all[decoupler_all["p_adj"] < fdr_threshold]
    .sort_values("p_adj")
    .groupby("factor", as_index=False)
    .first()
    [["factor", "term", "p_adj"]]
)

print(
    f"decoupler significant regulators for "
    f"{decoupler_results['factor'].nunique()} / {scores_df.shape[1]} factor-directions "
    f"(top 1 regulator per direction with FDR < {fdr_threshold})"
)
decoupler_results.sort_values("p_adj")

# %% [markdown]
# #### Store results

# %%
embed.uns[f'decoupler_{dc_geneset}_results'] = decoupler_results.convert_dtypes(convert_integer=False, convert_floating=False)

# %%

# %% [markdown]
# ## Write back to latent anndata

# %%
ad.settings.allow_write_nullable_strings = True

embed.write_h5ad(embed_path)
print(f"Updated embedding saved to: {embed_path}")

# %%

# %%

# %% [markdown]
# ## Appendix: Database reference
#
# The table below lists curated databases available for factor annotation, organized by domain. You can swap any of the tool-specific config variables above (e.g., `gsea_db`, `gp_source`, `dc_geneset`) to use different databases.
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
