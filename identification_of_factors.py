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
#     display_name: drvi_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Identification of factors

# %% [markdown]
# In this notebook, we use the already trained DRVI model on the immune dataset to show different ways to identify a factor. In summary we can:
#
# - Identify based on available annotation
# - Identify using GSEA
# - Identify using Language models
# - Looking into the annotation databases
#
# **We always advise examination by a biologist or looking into the published litereture for validation of the identified processes.**

# %% [markdown]
# ## Intro
#
# In this notrbook, we assume that the user has already trained DRVI on Immune data.
#
# Please refer to [General training and interpretability pipeline](./general_pipeline.html) tutorial.

# %% [markdown]
# ## Contact

# %% [markdown]
# For questions and help requests, you can reach out in the [scverse discourse](https://discourse.scverse.org/).
#
# If you found a bug, please use the [issue tracker](https://github.com/theislab/drvi/issues).

# %% [markdown]
# ## Install

# %% [markdown]
# If you try DRVI on colab, next cell will install dependencies.
#
# Please remove this part if your environment is already setup.

# %%
import sys

# if branch is stable, will install via pypi, else will install from source
branch = "latest"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB and branch == "stable":
    # !pip install multigrate[tutorials]
elif IN_COLAB and branch != "stable":
    # !pip install git+https://github.com/theislab/drvi.git#egg=drvi[tutorials]

# %% [markdown]
# ## Imports

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import anndata as ad
import scanpy as sc

import scvi
import drvi
from pathlib import Path
from drvi.model import DRVI
from drvi.utils.misc import hvg_batch

# %%
print("Last run with scvi-tools version:", scvi.__version__)
print("Last run with DRVI version:", drvi.__version__)

# %% [markdown]
# ## Config

# %%
# Set this to false if you already trained your model and do not like to retrain.
overwrite = False
SEED = 1  # Set to None if you don't want to set seed

# Set input output directory
# We use tmp_io/ directory in the same place as this notebook. Update accordingly.
io_dir = Path("./tmp_io/drvi_immune_128/")
io_dir.mkdir(parents=True, exist_ok=True)
io_dir

# %% [markdown]
# ## Load Data

# %%
# We already saved pre-processed data in previous notebook
adata = sc.read_h5ad(io_dir / "adata_preprocesses.h5ad")

# %% [markdown]
# ## Load DRVI ouputs

# %%
model_path = io_dir / "drvi_model"
embed_path = io_dir / "embed.h5ad"
traverse_adata_path = io_dir / "traverse_adata.h5ad"

model = DRVI.load(model_path, adata)
embed = sc.read_h5ad(embed_path)
traverse_adata = sc.read_h5ad(traverse_adata_path)

# %% [markdown]
# ## Identify based on available annotations

# %% [markdown]
# In this dataset we have annotations stored in `adata.obs["final_annotation"]`.
#
# We first measure Scaled Mutual Information (SMI) between each latent dimension and each category using DRVI built-in functions.

# %%
annot_col = "final_annotation"

# %% [markdown]
# ### Specific Imports

# %%
import math
import networkx as nx
from drvi.utils.metrics import DiscreteDisentanglementBenchmark


# %% [markdown]
# ### Helper functions

# %%
def plot_packed_network(df, title_col='title', var_col='variable', val_col='value'):
    """
    Visualizes network with edge weights shown to 2 decimal places.
    """
    # Create Graph
    G = nx.from_pandas_edgelist(df, title_col, var_col, edge_attr=val_col)
    
    # Custom Grid Layout Logic
    pos = {}
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    cols = math.ceil(len(components)**0.5)
    
    for i, nodes in enumerate(components):
        sub_pos = nx.spring_layout(G.subgraph(nodes), weight=val_col, k=0.5, seed=42)
        r, c = divmod(i, cols)
        for n, (x, y) in sub_pos.items():
            pos[n] = (x + c * 3, y - r * 3)

    plt.figure(figsize=(14, 10))
    titles = set(df[title_col])
    
    # Draw Nodes & Edges
    nx.draw(G, pos, 
            with_labels=True, font_size=8, font_weight='bold', node_size=600,
            node_color=['#A0CBE2' if n in titles else '#FF9E9E' for n in G.nodes()],
            width=[d[val_col] * 4 for u, v, d in G.edges(data=True)], 
            edge_color='grey', alpha=0.6)
    
    # Draw Edge Labels (Weights rounded to 2 decimals)
    edge_labels = {(u, v): f"{d[val_col]:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            
    plt.axis('off')
    plt.show()


# %% [markdown]
# ### Code

# %%
# Remove vanished dimensions
embed_nv = embed[:, embed.var['vanished'] == False].copy()
embed_nv

# %%
benchmark = DiscreteDisentanglementBenchmark(
    embed_nv.X, dim_titles=embed_nv.var['title'], discrete_target=embed.obs[annot_col],
    metrics=["SMI-disc", "SPN"], aggregation_methods=["LMS"],
)
benchmark.evaluate()
# You can optionally save benchmark object if you want.
# benchmark.save(filename)
# benchmark = DiscreteDisentanglementBenchmark.load(filename, embed_nv.X, dim_titles=embed_nv.var['title'], discrete_target=embed.obs["final_annotation"], metrics=["SMI-disc", "SPN"], aggregation_methods=["LMS"])

# %%
smi_similarity = benchmark.get_results_details()["SMI-disc"]
smi_similarity[:5]  # only showing 5 rows

# %%
filtering_threshold = 0.5

top_matches = (
    smi_similarity.reset_index()
    .melt(id_vars='title', value_vars=smi_similarity.columns)
    .query("value >= @filtering_threshold")
    .reset_index(drop=True)
)
top_matches

# %%
plot_packed_network(top_matches)

# %%

# %% [markdown]
# #### Heatmap

# %%
drvi.utils.pl.plot_latent_dims_in_heatmap(embed, "final_annotation", title_col="title")

# %% [markdown]
# It is possible to sort dimensions based on the top relevance with respect to a categoricals variable

# %%
drvi.utils.pl.plot_latent_dims_in_heatmap(embed, "final_annotation", title_col="title", sort_by_categorical=True)

# %%


# %% [markdown]
# ## Identification of programs

# %% [markdown]
# Once we identify the top relevant genes, we can determine some programs through supervised external information, such as:
# - existing annotations
# - examination by biologists
# - gene-set enrichment analysis (GSEA)
# - scientific literature
# - automated tools based on language models
#
# **Please refer to this tutorial for some tools that we found useful for identification of programs**
#
# It is worth mentioning that since such supervised information is not given to the model, the quality of the derived signatures is neither affected nor biased by it. Unidentified processes with high gene scores are promising candidates for further literature search, additional analysis, and even experimental design.

# %% [markdown]
# 1. Input: 
# -Gene list?
# -Ranked list?
# Expression Correlation with known Zelltype Markers
# 2. Reference Databases: 
# -Classic Datatabases: Gene Ontology, Reactome, MSigDB? 
# -Celltype specific Data (Cell Typist)
# -LLMs: gsai?
# 3. Annotation Method:
# -Over-representation
# -Regressions based: How well does a set of genes explain a factor
#
#
#

# %% [markdown]
# # Exploration Roadmap: DRVI Factor Annotation Pipeline

# %% [markdown]
# ## General Roadmap Settings
#

# %% [markdown]
# ### General Configurations

# %%
filter_vanished = True
factor_id_col = "title"   # keeps factor labels stable across tools (e.g. "DR 36")

# Comparison thresholds
corr_threshold = 0.4
spec_threshold = 0.1
gsea_fdr_threshold = 0.05


# %% [markdown]
# This roadmap outlines the systematic evaluation of tools for annotating latent factors. The goal is to move from abstract dimensions to interpretable biological processes using the immune dataset as a pilot.

# %% [markdown]
# ### Shared Data Setup

# %%
import celltypist
from celltypist import models
import blitzgsea as blitz
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# %% [markdown]
# ### Sync Cells
#

# %%
# make sure indices of adata and embed are the same
common_cells = adata.obs_names.intersection(embed_full.obs_names)
adata = adata[common_cells].copy()
embed_full = embed_full[common_cells].copy()
print(f"Cells: {adata.n_obs}")


# %% [markdown]
# ### Factor Filter
#

# %%
if filter_vanished and 'vanished' in embed_full.var.columns:
    mask = ~embed_full.var['vanished'].astype(bool).values
else:
    mask = np.ones(embed_full.n_vars, dtype=bool)
embed = embed_full[:, mask].copy()
print(f"Factors used: {embed.n_vars} | filter_vanished={filter_vanished}")


# %% [markdown]
# ### Standardized Factor IDs

# %%
if factor_id_col not in embed.var.columns:
    embed.var[factor_id_col] = [f"Factor_{i}" for i in range(embed.n_vars)]
factor_ids = embed.var[factor_id_col].astype(str).tolist()


# %% [markdown]
# ## 1. Statistical Annotation & Similarity 

# %% [markdown]
# Goal: Map latent factors to known cell types using existing annotations and atlases.
#
# Tools to Compare:
# * CellTypist: Utilizing the Immune_All_Low.pkl or High models for automated labeling.
# * GSEA: 
#     * Azimuth 2023 or Azimuth_Cell_Types_2021 (Reference Marker lists from Human Biomolecular Atlas Program)
#     * PanglaoDB_Augmented_2021 (Mouse and Human) --> clean markers lists
#     * HuBMAP_ASCTplusB_augmented_2022 (Anatomical Structures, Cell Typs, Biomarkers) --> very well curated, strict anatomical hierarchy
#     * Descartes_Cell_Types_and_Tissue_2021 (Human Cell Atlas) --> broad level annotation
#     * Tabula Sapiens (one of the biggest Single Cell Atlases, 24 Organs)
#     * ...
#
#
#
#
# Key Metrics:
# * Correlation
# * Specificity
#

# %% [markdown]
# ### 1.1 Cell Typist

# %% [markdown]
# * Input: normalized gene expression matrix of cells
# * Reference: Pre-trained Logistic Regression models trained on millions of annotated cells in different tissues
# * Algorithm: Linear Classification --> Calculates Decision Scores via linear combination of scaled expression and model coefficients, followed by a Maximum Score selection for identity.
# * Output: 
#     * Predicted Labels: Final call for each cell
#     * Decision Matrix: Raw classification scores
#     * Probability Matrix: Sigmoid-transformed scores (0 to 1)
#
#
#
#

# %% [markdown]
# #### CellTypist Model

# %%
#download celltypist model
print(models.models_description())
model_name = 'Immune_All_Low.pkl'
models.download_models(force_update=True, model=model_name)
ct_model = models.Model.load(model=model_name)

#load recommended model for immune cells
ct_model = models.Model.load(model=model_name)
print(model.cell_types)

# %% [markdown]
# #### CellTypist Annotation
#

# %%
# Annotate cell types using CellTypist
predictions = celltypist.annotate(adata, model=model_name, majority_voting=True)

adata.obs['celltypist_labels'] = predictions.predicted_labels['predicted_labels']

adata.obs['celltypist_majority'] = predictions.predicted_labels['majority_voting']

# %% [markdown]
# #### Extract Probability Matrix

# %%
# The Probability Matrix contains sigmoid-transformed scores (0 to 1)
# This is the primary input for the factor correlation analysis
prob_matrix = predictions.probability_matrix
prob_matrix.index = adata.obs_names

# Quick check: How many cell types did the model find?
print(f"Modell knows {prob_matrix.shape[1]} different Immune cell types.")

# %% [markdown]
# #### CellTypist Factor Correlation 

# %%
drvi_factors = pd.DataFrame(embed.X, index=embed.obs_names, columns=factor_ids)

# correlation: factor vs CellTypist class probabilities
eval_matrix = pd.DataFrame(
    np.corrcoef(drvi_factors.T, prob_matrix.T)[:drvi_factors.shape[1], drvi_factors.shape[1]:],
    index=drvi_factors.columns,
    columns=prob_matrix.columns,
)


# %%
#Visualize the correlation matrix as a heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(eval_matrix, cmap='RdBu_r', center=0)
plt.title("Concordance between DRVI factors and CellTypist probabilities")
plt.show()


# %% [markdown]
# #### CellTypist Validation

# %%
factor_to_check = "DR 36"
adata.obs["factor_check"] = drvi_factors[factor_to_check].reindex(adata.obs_names)

# %% [markdown]
# UMAP

# %%
sc.pl.umap(
    adata,
    color=["factor_check", "celltypist_majority"],
    ncols=2,
    frameon=False,
    cmap="viridis",
)


# %% [markdown]
# Violin Plot

# %%
sc.pl.violin(
    adata,
    keys="factor_check",
    groupby="celltypist_majority",
    rotation=90
)

# %% [markdown]
# #### CellTypist Summary

# %%
# top and second-best label per factor
top_1 = eval_matrix.idxmax(axis=1)
top_1_val = eval_matrix.max(axis=1)
tmp = eval_matrix.copy()
for i, c in enumerate(top_1):
    tmp.iloc[i, tmp.columns.get_loc(c)] = -1

# Calculate specificity as the difference between top and second-best correlation
specificity = top_1_val - tmp.max(axis=1)


#Summarize results in a DataFrame
celltypist_summary = pd.DataFrame({
    'Factor': eval_matrix.index,
    'Top_CellType': top_1.values,
    'Correlation': top_1_val.values,
    'Specificity': specificity.values,
})

celltypist_summary['Tool'] = 'celltypist'
celltypist_summary['Label_std'] = celltypist_summary['Top_CellType'].astype(str).str.lower().str.replace(r'[^a-z0-9]+', '_', regex=True).str.strip('_')
celltypist_summary['Significant'] = (
    (celltypist_summary['Correlation'] >= corr_threshold)
    & (celltypist_summary['Specificity'] >= spec_threshold)
)
display(celltypist_summary[celltypist_summary['Significant']].head(20))


# %% [markdown]
# ## 2. Gene Set Enrichment Analysis (Functional Identity)

# %% [markdown]
# Goal: Identify biological processes (e.g., "Interferon Response", "Cell Cycle") for factors that do not map 1-to-1 to a cell type.
#
# Tools to Compare:
# * gProfiler (gprofiler-official): The benchmark used in the DRVI preprint
# * Gseapy: Python implementation for local Enrichment analysis (Enrichr/MSigDB)
# * Decoupler: A fast framework for footprint-based enrichment (e.g., PROXIMA)
# * Blitzqseq
#
# Key Metrics:

# %% [markdown]
# ### 2.1 Blitzqseq

# %% [markdown]
# How it works:
# * Input: Ranked gene list (list of all genes sorted by loadings for a specific factor)
# * Reference: Database on Enrichr
#     * Key Collections: 
#         * e.g MgSigDB
#             * H (Hallmark): Broad biological states (e.g., "Hypoxia", "Inflammatory Response").
#             * C5 (GO): Highly specific Gene Ontology terms.
#             * C2 (CP): Curated pathways from Reactome or KEGG.
#         * Cellmarker/Azimuth
#         * Reactome/KEGG
# * Algorithm: 
#     * Pre-ranking: Unlike standard GSEA which compares groups of cells, this version only looks at the ranking of genes
#     * It calculates an Enrichment Score (ES) that increases when genes from a specific pathway appear at the top of your ranked list (high loadings)
#     * Speed Optimization: BlitzGSEA uses a probability distribution approximation to estimate the null model. Instead of running thousands of slow permutations for every gene set, it uses mathematical shortcuts to calculate p-values almost instantly
# * Output: 
#     * NES (Normalized Enrichment Score): A high positive NES indicates that the biological process is strongly represented by that factor
#     * p-value & FDR (q-value): Statistical significance. Usually, you filter for FDR<0.05
#     * Leading Edge Genes: The specific subset of genes within a pathway that actually drove the enrichment score

# %% [markdown]
# #### BlitzGSEA Library

# %%
# MSigDB Hallmark is the gold standard for the first evaluation
signature_lib = blitz.enrichr.get_library("MSigDB_Hallmark_2020")

# %% [markdown]
# #### BlitzGSEA Scores (from latent dimensions)

# %%
# Use full rankings, both directions
pos_df = traverse_adata.varm["max_possible_traverse_effect_pos"].copy()
neg_df = traverse_adata.varm["max_possible_traverse_effect_neg"].copy()
pos_df.columns = pos_df.columns.astype(str)
neg_df.columns = neg_df.columns.astype(str)

#Name mapping from factor IDs to titles (e.g. "DR 36")
dimid_to_title = (
    embed.var[["original_dim_id", factor_id_col]]
    .assign(original_dim_id=lambda d: d["original_dim_id"].astype(str),
            factor_id=lambda d: d[factor_id_col].astype(str))
    .drop_duplicates("original_dim_id")
    .set_index("original_dim_id")["factor_id"]
    .to_dict()
)
# only keep factors that are in both pos and neg and have a title
raw_ids = [rid for rid in pos_df.columns if rid in neg_df.columns and rid in dimid_to_title] 

# Combine pos and neg scores, and rename columns to include direction and factor titles
scores_df = pd.concat([pos_df[raw_ids].add_suffix("+"), neg_df[raw_ids].add_suffix("-")], axis=1)
scores_df.columns = [f"{dimid_to_title[c[:-1]]}{c[-1]}" for c in scores_df.columns if c[:-1] in dimid_to_title]
print(f"GSEA inputs: {scores_df.shape[1]} factor-directions")

print(f"GSEA inputs from DRVI framework: {len(gsea_inputs)} factor-directions")

# %% [markdown]
# ##### GSEA

# %%
blitzgsea_results = [] # to store results
for factor_dir in scores_df.columns:
    try:
        signature = scores_df[factor_dir].rename("v").reset_index().rename(columns={"index": "i"}) # blitzgsea expects columns "i" and "v"
        signature["v"] = pd.to_numeric(signature["v"], errors="coerce") # ensure numeric and coerce errors to NaN
        signature = signature.replace([np.inf, -np.inf], np.nan).dropna(subset=["v"])
        res = blitz.gsea(signature, signature_lib, processes=4) # run GSEA with parallelization
        sig = res[res["fdr"] < gsea_fdr_threshold].sort_values("fdr") # filter significant results and sort by FDR
        if len(sig): # if there are significant results, take the top one, else record as no significant enrichment
            blitzgsea_results.append({
                "FactorDir": factor_dir,
                "Factor": factor_dir[:-1],
                "Direction": factor_dir[-1],
                "Term": sig.index[0],
                "NES": float(sig.iloc[0]["nes"]),
                "FDR": float(sig.iloc[0]["fdr"]),
            })
        else:
            blitzgsea_results.append({"FactorDir": factor_dir, "Factor": factor_dir[:-1], "Direction": factor_dir[-1], "Term": "No significant enrichment", "NES": 0.0, "FDR": 1.0})
    except Exception as e: # if any error occurs (e.g. due to bad input), record it and continue with the next factor
        blitzgsea_results.append({"FactorDir": factor_dir, "Factor": factor_dir[:-1], "Direction": factor_dir[-1], "Term": f"ERROR: {type(e).__name__}", "NES": 0.0, "FDR": 1.0})

blitzgsea_summary = pd.DataFrame(blitzgsea_results) 
display(blitzgsea_summary.head(60))


# %% [markdown]
# #### BlitzGSEA Summary

# %%
# Create a summary of significant GSEA results
annotated = blitzgsea_summary[blitzgsea_summary["FDR"] < gsea_fdr_threshold].copy()

#Calculate key performance indicators

# how many percent of latent factors got at least one significant annotation?
coverage = 100 * len(annotated) / len(blitzgsea_summary) if len(blitzgsea_summary) else 0 
print(f"Coverage (FDR<{gsea_fdr_threshold}): {coverage:.2f}% ({len(annotated)}/{len(blitzgsea_summary)})")

# how many unique terms were found? This indicates the diversity of biological processes captured by the factors.
print(f"Unique terms: {annotated['Term'].nunique()}")

# Normalized Enrichment Score. A value above 2 is generally considered strong evidence of enrichment. The median NES gives a sense of the overall strength of the annotations.
print(f"Median NES: {annotated['NES'].median():.2f}" if len(annotated) else "Median NES: n/a")


# %% [markdown]
# #### Preparation Cross Tool Comparison

# %%
blitzgsea_tool_summary = blitzgsea_summary.copy()
blitzgsea_tool_summary["Tool"] = "blitzgsea"
blitzgsea_tool_summary["Label"] = blitzgsea_tool_summary["Term"].astype(str)
blitzgsea_tool_summary["Label_std"] = blitzgsea_tool_summary["Label"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
blitzgsea_tool_summary["Significant"] = blitzgsea_tool_summary["FDR"] < gsea_fdr_threshold

display(annotated.sort_values("NES", ascending=False).head(20))

# %% [markdown]
# #### Cross Tool Comparison

# %%
# Preparation of Cell-Typist Data for Comparison
ct_sig = celltypist_summary[celltypist_summary['Significant']][['Factor', 'Label_std', 'Correlation', 'Specificity']]
ct_sig = ct_sig.rename(columns={'Label_std': 'CellTypist_Label'})

#Filter significant GSEA results
gs_sig = (
    blitzgsea_tool_summary[blitzgsea_tool_summary['Significant']]
    .sort_values(['FDR', 'NES'], ascending=[True, False])
    .drop_duplicates('Factor')
    [['Factor', 'Direction', 'Label_std', 'NES', 'FDR']]
    .rename(columns={'Label_std': 'BlitzGSEA_Label'})
)

# Merge CellTypist and BlitzGSEA results on Factor
comparison = ct_sig.merge(gs_sig, on='Factor', how='outer')
comparison['Relationship'] = np.where(
    comparison['CellTypist_Label'].isna(),
    'blitzgsea_only', # if CellTypist label is missing but GSEA gives a result it's a BlitzGSEA-only annotation
    np.where(
        comparison['BlitzGSEA_Label'].isna(),
        'celltypist_only', #factor is a cell type
        np.where(comparison['CellTypist_Label'] == comparison['BlitzGSEA_Label'], 'same_label', 'complementary') #both tools give an annotation, if they match it's "same_label", else it's "complementary"
    )
)

print("Cross-tool relationship counts:")
display(comparison['Relationship'].value_counts()) 
display(comparison.sort_values(['Relationship', 'Factor']).head(50))


# %% [markdown]
# ### 2.2 Gseapy (Preranking Module)

# %% [markdown]
# How it works:
# * Input: 
#     * Ranked gene list (list of all genes sorted by loadings for a specific factor) for Preranked Module [Important: Gene names have to be in capital letters]
#     * Gene expression matrix and group annotation for classical module
# * Reference: Database on Enrichr
#     * Key Collections: 
#         * e.g MgSigDB
#             * H (Hallmark): Broad biological states (e.g., "Hypoxia", "Inflammatory Response").
#             * C5 (GO): Highly specific Gene Ontology terms.
#             * C2 (CP): Curated pathways from Reactome or KEGG.
#         * Cellmarker/Azimuth
#         * Reactome/KEGG
# * Algorithm: 
#     * Pre-ranking: Unlike standard GSEA which compares groups of cells, this version only looks at the ranking of genes
#     * It calculates an Enrichment Score (ES) that increases when genes from a specific pathway appear at the top of your ranked list (high loadings)
#     * Permutation-based Null Model: It randomly reassigns gene labels many times (default is 1000 iterations) to see how often a similar ES occurs by pure chance
# * Output: 
#     * NES (Normalized Enrichment Score): A high positive NES indicates that the biological process is strongly represented by that factor
#     * FDR (q-value): Statistical significance. Usually, you filter for FDR<0.05
#     * Lead_genes: The specific subset of genes within a pathway that actually drove the enrichment score

# %% [markdown]
# #### Gseapy Import
#

# %%
import gseapy as gp

# %% [markdown]
# #### Choosing Gseapy Database

# %%
gseapy_db = "MSigDB_Hallmark_2020"
print("Using DB:", gseapy_db)

# %% [markdown]
# #### Get Gene Library

# %%
gseapy_lib= blitz.enrichr.get_library(gseapy_db)
print("Gene sets:", len(gseapy_lib))

# %% [markdown]
# #### Preparing Data

# %%
scores_df.index = scores_df.index.str.upper()

# %% [markdown]
# #### Run Gseapy Prerank Loop
#

# %%
gseapy_results = []

for factor_dir in scores_df.columns:
    signature = (
        scores_df[factor_dir]
        .rename("score")
        .reset_index()
        .rename(columns={"index": "gene"})
    )

    signature["score"] = pd.to_numeric(signature["score"], errors="coerce")
    signature = signature.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
    signature = signature.drop_duplicates(subset=["gene"]).sort_values("score", ascending=False)

    try:
        pre_res = gp.prerank(
            rnk=signature[["gene", "score"]],
            gene_sets=gseapy_lib, 
            outdir=None, #to avoid file output
            min_size=5, #minimum gene set size to consider (to avoid very small sets that are hard to interpret)
            max_size=500, #maximum gene set size to consider (to avoid very broad terms)
            permutation_num=100, #number of permutations for significance testing
            seed=42, #random seed
            verbose=False, #to see progress and potential warnings during GSEA runs
        )

        res_df = pre_res.res2d.copy().sort_values("FDR q-val", ascending=True)

        if len(res_df):
            top = res_df.iloc[0]
            term_name = top["Term"] if "Term" in res_df.columns else res_df.index[0]

            gseapy_results.append({
                "FactorDir": factor_dir,
                "Factor": factor_dir[:-1],
                "Direction": factor_dir[-1],
                "Term": str(term_name),
                "NES": float(top["NES"]),
                "FDR": float(top["FDR q-val"]),
            })
        else:
            gseapy_results.append({
                "FactorDir": factor_dir, "Factor": factor_dir[:-1], "Direction": factor_dir[-1],
                "Term": "No significant enrichment", "NES": 0.0, "FDR": 1.0
            })

    except Exception as e:
        gseapy_results.append({
            "FactorDir": factor_dir, "Factor": factor_dir[:-1], "Direction": factor_dir[-1],
            "Term": f"ERROR: {type(e).__name__}", "NES": 0.0, "FDR": 1.0
        })

gseapy_summary = pd.DataFrame(gseapy_results)



# %%
#Visualize GSEA results
display(
    gseapy_summary[gseapy_summary["FDR"] < gsea_fdr_threshold]
    .sort_values(["FDR", "NES"], ascending=[True, False])
)

# %% [markdown]
# #### Gseapy Summary
#

# %%
gseapy_tool_summary = gseapy_summary.copy()
gseapy_tool_summary["Tool"] = "gseapy"
gseapy_tool_summary["Label"] = gseapy_tool_summary["Term"].astype(str)
gseapy_tool_summary["Label_std"] = (
    gseapy_tool_summary["Label"]
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)
gseapy_tool_summary["Significant"] = gseapy_tool_summary["FDR"] < gsea_fdr_threshold

annotated_gseapy = gseapy_tool_summary[gseapy_tool_summary["Significant"]].copy()
coverage_gseapy = (100 * len(annotated_gseapy) / len(gseapy_tool_summary)) if len(gseapy_tool_summary) else 0

print(f"Gseapy coverage (FDR<{gsea_fdr_threshold}): {coverage_gseapy:.2f}% ({len(annotated_gseapy)}/{len(gseapy_tool_summary)})")
print(f"Unique terms: {annotated_gseapy['Term'].nunique()}")
print(f"Median NES: {annotated_gseapy['NES'].median():.2f}" if len(annotated_gseapy) else "Median NES: n/a")

display(annotated_gseapy.sort_values("FDR", ascending=True).head(20))


# %% [markdown]
# ### 3. Language Model Based Identification (Advanced Annotation)

# %% [markdown]
#

# %% [markdown]
# Goal: Automate "narrative" annotation and validation using LLMs.
#
# Tools to Compare:
# * gsai (Gene Set AI): Specialized LLM tool for gene list interpretation.
# * Direct LLM Prompting: Using GPT-4/Claude via API to summarize factor-defining genes.
# * OpenScholar: For literature-backed validation of the proposed factor names.
#
# Key Metrics:

# %% [markdown]
# ### 4. Final Integration & Verification

# %% [markdown]
# Goal: Create the final notebook structure to guide the user.
#
# Implementation:
# * Unified API: A single function to call all three categories and present a summary table.
# * Verify Cells: Visual sanity checks (UMAPs + Dotplots) to verify that a factor's activity matches the assigned name.
# * Filtering: Automating the exclusion of "Vanished Factors" (max absolute value < 1) to reduce noise.
