# Script must be run from root directory of the totalVI_journal repo.
from scvi.dataset import Dataset10X
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import doubletdetection

import sys

sys.path.append("./utils/")
from utils import seurat_v3_highly_variable_genes

# Load data
save_path = "data/raw_data/"

dataset = Dataset10X(
    dataset_name="malt_10k_protein_v3",
    save_path=save_path,
    measurement_names_column=1,
    dense=True,
)

# Filter control proteins
non_control_proteins = []
for i, p in enumerate(dataset.protein_names):
    if not p.startswith("IgG"):
        non_control_proteins.append(i)
    else:
        print(p)
dataset.protein_expression = dataset.protein_expression[:, non_control_proteins]
dataset.protein_names = dataset.protein_names[non_control_proteins]

# Make anndata object
adata = anndata.AnnData(dataset.X)
adata.var.index = dataset.gene_names
adata.var_names_make_unique()
adata.obs.index = dataset.barcodes
adata.obsm["protein_expression"] = dataset.protein_expression
adata.uns["protein_names"] = dataset.protein_names

# Filter doublets called by DoubletDetection
try:
    doublets = np.load("data/metadata/malt_doublets.npy")
except FileNotFoundError:
    clf = doubletdetection.BoostClassifier(
        n_iters=25, use_phenograph=True, verbose=True, standard_scaling=False
    )
    doublets = clf.fit(adata.X).predict(p_thresh=1e-7, voter_thresh=0.8) == 1
    print("{} doublet rate".format(np.sum(doublets) / adata.X.shape[0]))
    np.save("data/metadata/malt_doublets.npy", doublets)

adata = adata[~doublets.astype(np.bool)]

# Filter cells by min_genes
sc.pp.filter_cells(adata, min_genes=200)

# Filter cells by mitochondrial reads, n_genes, n_counts
mito_genes = adata.var_names.str.startswith("MT-")
adata.obs["percent_mito"] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(
    adata.X, axis=1
)
adata.obs["n_counts"] = adata.X.sum(axis=1)

adata = adata[adata.obs["percent_mito"] < 0.15, :]
adata = adata[adata.obs["n_genes"] < 5000, :]
adata = adata[adata.obs["n_counts"] < 30000, :]

# filtering by protein library size (these were manually selected based on histogram)
pro_lib_size = adata.obsm["protein_expression"].sum(axis=1)
keep_protein_cells = np.logical_and(pro_lib_size >= 400, pro_lib_size <= 20000)
adata = adata[keep_protein_cells]

# Filter genes
sc.pp.filter_genes(adata, min_cells=4)

# Find highly variable genes (really adds a mask for genes)
adata_hvg = adata.copy()
seurat_v3_highly_variable_genes(adata_hvg, n_top_genes=4000)
# sc.pp.normalize_per_cell(adata_hvg, counts_per_cell_after=1e4)
# sc.pp.log1p(adata_hvg)
# sc.pp.highly_variable_genes(adata_hvg, n_top_genes=4000)

adata.var["highly_variable"] = adata_hvg.var["highly_variable"]

adata.write("data/malt_10k_protein_v3.h5ad", compression="gzip")
