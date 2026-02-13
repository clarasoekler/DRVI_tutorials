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
# This roadmap outlines the systematic evaluation of tools for annotating latent factors. The goal is to move from abstract dimensions to interpretable biological processes using the immune dataset as a pilot.

# %% [markdown]
# ### 1. Statistical Annotation & Similarity 

# %% [markdown]
# Goal: Map latent factors to known cell types using existing annotations and atlases.
#
# Tools to Compare:
# * CellTypist: Utilizing the Immune_All_Low.pkl or High models for automated labeling.
# * Single R
# * Reference Mapping (Scanpy Ingest/scArches): Projecting query data onto a high-quality PBMC/Immune atlas.
# * Direct Regression: Using a Logistic Regression classifier to see if factors linearly predict known cell types.
#
# Key Metrics:
# * LMS-SMI (Scaled Mutual Information): Measures the exclusivity of a factor for a specific label. High SMI = high disentanglement.
# * LMS-SPN (Same-Process Neighbors): Evaluates if cells with the same annotation are closer in the latent space.
# * Diagonalization: Visual assessment via Heatmaps to see if factors have 1-to-1 mappings to cell types.

# %% [markdown]
# #### 1.1 Cell Typist

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
# ##### Importing Libraries

# %%
import celltypist
from celltypist import models
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ##### Setup and Load data

# %%
#make sure indices of adata and embed are the same, so we can transfer factors to adata object

#Load data
embed = sc.read_h5ad("/Users/clara.sanchez/Documents/code/drvi_project/drvi_tutorials/tmp_io/drvi_immune_128/embed.h5ad")
adata = sc.read_h5ad("/Users/clara.sanchez/Documents/code/drvi_project/drvi_tutorials/tmp_io/drvi_immune_128/adata_preprocesses.h5ad")  

#Synchronize barcodes
common_cells = adata.obs_names.intersection(embed.obs_names)
adata = adata[common_cells].copy()
embed = embed[common_cells].copy()

#Transfer factors
for i in range(embed.X.shape[1]):
    adata.obs[f'DRVI_F_{i}'] = embed.X[:, i]

print(f"Synchronized! Both objects have {adata.n_obs} cells.")

# %% [markdown]
# ##### Download Celltypist Model

# %%
#download celltypist model
print(models.models_description())
model_name = 'Immune_All_Low.pkl'
model = models.download_models(force_update=True, model=model_name)


#download recommended model for immune cells
model = models.Model.load(model = 'Immune_All_Low.pkl')

print(model.cell_types)


# %% [markdown]
# ##### Annotate Cells

# %%
#Annotation
predictions = celltypist.annotate(adata, model = 'Immune_All_Low.pkl', majority_voting = True)
adata.obs['celltypist_labels'] = predictions.predicted_labels['predicted_labels']
adata.obs['celltypist_majority'] = predictions.predicted_labels['majority_voting']

# %% [markdown]
# ##### Extract Probability Matrix

# %%
# The Decision Matrix contains raw logit scores (useful for confidence checks)
decision_matrix = predictions.decision_matrix

# The Probability Matrix contains sigmoid-transformed scores (0 to 1)
# This is the primary input for the factor correlation analysis
prob_matrix = predictions.probability_matrix
prob_matrix.index = adata.obs_names

# Quick check: How many cell types did the model find?
print(f"Modell knows {prob_matrix.shape[1]} different Immune cell types.")

# %% [markdown]
# ##### Quantitative Evaluation (Correlation with DRVI factors)

# %%
# Extract DRVI factors from the latent representation
## The .X matrix  contains the latent representations
adata.obsm['X_drvi'] = embed.X

drvi_factors = pd.DataFrame(adata.obsm['X_drvi'], index=adata.obs_names)
drvi_factors.columns = [f'Factor_{i}' for i in range(drvi_factors.shape[1])]

# Calculate Correlation between each Factor and each CellTypist Identity
eval_matrix = pd.DataFrame(
    np.corrcoef(drvi_factors.T, prob_matrix.T)[:drvi_factors.shape[1], drvi_factors.shape[1]:],
    index=drvi_factors.columns,
    columns=prob_matrix.columns
)

# Plotting the Evaluation Heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(eval_matrix, cmap='RdBu_r', center=0)
plt.title("Evaluation: Concordance between DRVI Factors and CellTypist Probabilities")
plt.show()

# %% [markdown]
# #### Factor Assignment Table with a Specificity Score

# %%
# 1. Extract the Top Match and the second-best match using 'eval_matrix'
top_1 = eval_matrix.idxmax(axis=1)
top_1_val = eval_matrix.max(axis=1)

# To find the 2nd best, we temporarily mask the top one
temp_matrix = eval_matrix.copy()
for i, col in enumerate(top_1):
    temp_matrix.loc[temp_matrix.index[i], col] = -1
top_2_val = temp_matrix.max(axis=1)

# 2. Calculate Specificity Score
specificity = top_1_val - top_2_val

# 3. Create the Quantification Table
quant_df = pd.DataFrame({
    'Factor': eval_matrix.index,
    'Top_CellType': top_1.values,
    'Correlation': top_1_val.values,
    'Specificity': specificity.values
}).sort_values('Correlation', ascending=False)

# 4. Filter for "Identified" factors (Correlation > 0.4)
identified_factors = quant_df[quant_df['Correlation'] > 0.4]

print(f"Quantification Summary:")
print(f"- Total Factors evaluated: {len(quant_df)}")
print(f"- Successfully assigned (Corr > 0.4): {len(identified_factors)}")
print(f"- Highly Specific (> 0.2 Gap): {len(quant_df[quant_df['Specificity'] > 0.2])}")

quant_df.head(10)

# %% [markdown]
# ### 2. Gene Set Enrichment Analysis (Functional Identity)

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
# ### 3. Language Model Based Identification (Advanced Annotation)

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
