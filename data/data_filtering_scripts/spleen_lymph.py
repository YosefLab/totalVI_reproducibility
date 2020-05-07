# Script must be run from root directory of the totalVI_journal repo.

import anndata
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

import sys

sys.path.append("./utils/")
from utils import seurat_v3_highly_variable_genes

sc.settings.verbosity = 4

# Load data and arrange protein data
save_path_111 = "/data/yosef2/users/zsteier/TotalSeq/20190814_BioLegend_ZRS08/cellranger_aggr/ZRS08_spleenLN111_aggr/outs/filtered_feature_bc_matrix/"
save_path_206 = "/data/yosef2/users/zsteier/TotalSeq/20190814_BioLegend_ZRS08/cellranger_aggr/ZRS08_spleenLN206_aggr/outs/filtered_feature_bc_matrix/"

dataset_111 = sc.read_10x_mtx(save_path_111, gex_only=False)
dataset_111.obsm["protein_expression"] = dataset_111[
    :, dataset_111.var["feature_types"] == "Antibody Capture"
].X.A
dataset_111.uns["protein_names"] = np.array(
    dataset_111.var_names[dataset_111.var["feature_types"] == "Antibody Capture"]
)
dataset_111 = dataset_111[
    :, dataset_111.var["feature_types"] != "Antibody Capture"
].copy()
dataset_111.X = dataset_111.X.A
dataset_111.var_names_make_unique()

dataset_206 = sc.read_10x_mtx(save_path_206, gex_only=False)
dataset_206.obsm["protein_expression"] = dataset_206[
    :, dataset_206.var["feature_types"] == "Antibody Capture"
].X.A
dataset_206.uns["protein_names"] = np.array(
    dataset_206.var_names[dataset_206.var["feature_types"] == "Antibody Capture"]
)
dataset_206 = dataset_206[
    :, dataset_206.var["feature_types"] != "Antibody Capture"
].copy()
dataset_206.X = dataset_206.X.A
dataset_206.var_names_make_unique()

# Load metadata
meta_111 = pd.read_csv("data/metadata/spleenLN111/hashtag_labels.csv")
meta_206 = pd.read_csv("data/metadata/spleenLN206/hashtag_labels.csv")
pro_names_matched = pd.read_csv(
    "data/metadata/protein_metadata/TotalSeq_ADT206_ADT111_matched.csv"
)
protein_to_rna_table = pd.read_csv(
    "data/metadata/protein_metadata/TotalSeq_ADT111_barcodes_RNA_20190717.csv"
)

# Subset on proteins included in the 111 panel
subset_111 = pro_names_matched["name_206"][
    ~pro_names_matched["name_111"].isnull()
].values
keep = []
for i, p in enumerate(dataset_111.uns["protein_names"]):
    if p in subset_111:
        keep.append(i)
dataset_111.obsm["protein_expression"] = dataset_111.obsm["protein_expression"][:, keep]
dataset_111.uns["protein_names"] = dataset_111.uns["protein_names"][keep]
# fix mislabeling of hashtag in 111
dataset_111.uns["protein_names"][-2] = "HTO_B6_spl_r4_206_A0301"
dataset_111.uns["protein_names"][-1] = "HTO_B6_LN_r4_206_A0302"

# calculate protein library size
dataset_111.obs["n_protein_counts"] = dataset_111.obsm["protein_expression"].sum(axis=1)
dataset_206.obs["n_protein_counts"] = dataset_206.obsm["protein_expression"].sum(axis=1)

# calculate number of proteins
dataset_111.obs["n_proteins"] = (dataset_111.obsm["protein_expression"] > 0).sum(axis=1)
dataset_206.obs["n_proteins"] = (dataset_206.obsm["protein_expression"] > 0).sum(axis=1)

# Some genes encode multiple protein so it's ok to overwrite key in this case as it will be included downstream
protein_111_names = pro_names_matched["name_111"][
    ~pro_names_matched["name_111"].isnull()
].values
g_to_p = {}
for p1, p2 in zip(protein_111_names, subset_111):
    ind = np.where(protein_to_rna_table["name"] == p1)[0]
    g_to_p[str(protein_to_rna_table.loc[ind, "RNA"].values[0])] = p2


# prepare anndata object for demultiplexing
hash_id_111 = np.array(["todo"] * len(dataset_111), dtype=object)
dataset_111.obs["seurat_hash_id"] = meta_111["Hash_ID"].values
dataset_111.obs["batch_indices"] = [
    int(b.split("-")[1]) - 1 for b in dataset_111.obs.index
]
hash_id_206 = np.array(["todo"] * len(dataset_206), dtype=object)
dataset_206.obs["seurat_hash_id"] = meta_206["Hash_ID"].values
dataset_206.obs["batch_indices"] = [
    int(b.split("-")[1]) - 1 + 2 for b in dataset_206.obs.index
]

# Demultiplexing by GMM
hto_names = dataset_111.uns["protein_names"][-2:]
hto_names = np.array([h.split("_")[2] for h in hto_names])
for b in np.unique(dataset_111.obs["batch_indices"]):
    log_hto_counts = np.log1p(
        dataset_111.obsm["protein_expression"][dataset_111.obs["batch_indices"] == b][
            :, -2:
        ]
    )
    post_probs = []
    gmm_classifier = []
    for col in range(log_hto_counts.shape[1]):
        gmm = GaussianMixture(n_components=2)
        probs = gmm.fit(log_hto_counts[:, col].reshape(-1, 1)).predict_proba(
            log_hto_counts[:, col].reshape(-1, 1)
        )

        # Make sure lower mode is first mean
        means = gmm.means_.ravel()
        mean_order = np.argsort(means)
        print(means[mean_order])
        probs = probs[:, mean_order]
        # Only need the probability of being "on"
        post_probs.append(probs[:, 1])
        labels = np.argmax(probs, axis=1).flatten()
        gmm_classifier.append(labels)
    gmm_classifier = np.array(gmm_classifier).T
    post_probs = np.array(post_probs).T
    classification = (post_probs > 0.5).sum(axis=1).astype(str)
    classification[classification == "2"] = "Doublet"
    classification[classification == "0"] = "Negative"
    classification[classification == "1"] = hto_names[
        np.argmax(post_probs[classification == "1"], axis=1)
    ]
    hash_id_111[(dataset_111.obs["batch_indices"] == b).values.ravel()] = classification
hash_id_111[hash_id_111 == "spl"] = "Spleen"
hash_id_111[hash_id_111 == "LN"] = "Lymph Node"
dataset_111.obs["hash_id"] = hash_id_111

hto_names = dataset_206.uns["protein_names"][-2:]
hto_names = np.array([h.split("_")[2] for h in hto_names])
for b in np.unique(dataset_206.obs["batch_indices"]):
    log_hto_counts = np.log1p(
        dataset_206.obsm["protein_expression"][dataset_206.obs["batch_indices"] == b][
            :, -2:
        ]
    )
    post_probs = []
    gmm_classifier = []
    for col in range(log_hto_counts.shape[1]):
        gmm = GaussianMixture(n_components=2)
        probs = gmm.fit(log_hto_counts[:, col].reshape(-1, 1)).predict_proba(
            log_hto_counts[:, col].reshape(-1, 1)
        )

        # Make sure lower mode is first mean
        means = gmm.means_.ravel()
        mean_order = np.argsort(means)
        print(means[mean_order])
        probs = probs[:, mean_order]
        # Only need the probability of being "on"
        post_probs.append(probs[:, 1])
        labels = np.argmax(probs, axis=1).flatten()
        gmm_classifier.append(labels)
    gmm_classifier = np.array(gmm_classifier).T
    post_probs = np.array(post_probs).T
    classification = (post_probs > 0.5).sum(axis=1).astype(str)
    classification[classification == "2"] = "Doublet"
    classification[classification == "0"] = "Negative"
    classification[classification == "1"] = hto_names[
        np.argmax(post_probs[classification == "1"], axis=1)
    ]
    hash_id_206[(dataset_206.obs["batch_indices"] == b).values.ravel()] = classification
hash_id_206[hash_id_206 == "spl"] = "Spleen"
hash_id_206[hash_id_206 == "LN"] = "Lymph Node"
dataset_206.obs["hash_id"] = hash_id_206

# Filter doublets called by Seurat
keep_cells = dataset_111.obs["seurat_hash_id"] != "Doublet"
dataset_111 = dataset_111[keep_cells]
keep_cells = dataset_111.obs["seurat_hash_id"] != "Negative"
dataset_111 = dataset_111[keep_cells]

keep_cells = dataset_206.obs["seurat_hash_id"] != "Doublet"
dataset_206 = dataset_206[keep_cells]
keep_cells = dataset_206.obs["seurat_hash_id"] != "Negative"
dataset_206 = dataset_206[keep_cells]

# Filter cells by min_genes
sc.pp.filter_cells(dataset_111, min_genes=200)
sc.pp.filter_cells(dataset_206, min_genes=200)

# filtering by protein library size (these were manually selected based on histogram)
keep_protein_cells = np.logical_and(
    dataset_111.obs["n_protein_counts"] >= 400,
    dataset_111.obs["n_protein_counts"] <= 10000,
)
dataset_111 = dataset_111[keep_protein_cells]

keep_protein_cells = np.logical_and(
    dataset_206.obs["n_protein_counts"] >= 400,
    dataset_206.obs["n_protein_counts"] <= 10000,
)
dataset_206 = dataset_206[keep_protein_cells]

# filtering by number of proteins (these were manually selected based on histogram)
keep_protein_cells = dataset_111.obs["n_proteins"] >= 90
dataset_111 = dataset_111[keep_protein_cells]

keep_protein_cells = dataset_206.obs["n_proteins"] >= 170
dataset_206 = dataset_206[keep_protein_cells]

# Filter cells by mitochondrial reads
mito_genes = dataset_111.var_names.str.startswith("mt-")
dataset_111.obs["percent_mito"] = np.sum(dataset_111[:, mito_genes].X, axis=1) / np.sum(
    dataset_111.X, axis=1
)
dataset_111 = dataset_111[dataset_111.obs["percent_mito"] < 0.15, :]

mito_genes = dataset_206.var_names.str.startswith("mt-")
dataset_206.obs["percent_mito"] = np.sum(dataset_206[:, mito_genes].X, axis=1) / np.sum(
    dataset_206.X, axis=1
)
dataset_206 = dataset_206[dataset_206.obs["percent_mito"] < 0.15, :]

# Filter genes that only a few cells express
for b in np.unique(dataset_111.obs["batch_indices"]):
    n_cell_per_gene = (dataset_111[dataset_111.obs["batch_indices"] == b].X > 0).sum(
        axis=0
    )
    dataset_111 = dataset_111[:, n_cell_per_gene >= 4]
    dataset_206 = dataset_206[:, n_cell_per_gene >= 4]

for b in np.unique(dataset_206.obs["batch_indices"]):
    n_cell_per_gene = (dataset_206[dataset_206.obs["batch_indices"] == b].X > 0).sum(
        axis=0
    )
    dataset_111 = dataset_111[:, n_cell_per_gene >= 4]
    dataset_206 = dataset_206[:, n_cell_per_gene >= 4]

assert (dataset_111.var_names == dataset_206.var_names).all()

adata_total = anndata.AnnData.concatenate(dataset_111, dataset_206)

# Find highly variable genes (really adds a mask for genes)
adata_hvg = adata_total.copy()
adata_hvg.obs["batch"] = [
    str(b) for b in np.array(adata_hvg.obs["batch_indices"]).ravel()
]
seurat_v3_highly_variable_genes(adata_hvg, n_top_genes=4000)
# sc.pp.normalize_per_cell(adata_hvg, counts_per_cell_after=1e4)
# sc.pp.log1p(adata_hvg)
# sc.pp.highly_variable_genes(
#     adata_hvg, n_top_genes=4000, batch_key="batch", flavor="seurat", n_bins=8
# )

dataset_111.var["highly_variable"] = adata_hvg.var["highly_variable"]
dataset_111.var["highly_variable_mean_variance"] = adata_hvg.var[
    "highly_variable_mean_variance"
]
dataset_206.var["highly_variable"] = adata_hvg.var["highly_variable"]
dataset_206.var["highly_variable_mean_variance"] = adata_hvg.var[
    "highly_variable_mean_variance"
]

dataset_206.obs["batch_indices"] = dataset_206.obs["batch_indices"] - 2

# Keep genes that encode proteins
encode = []
for g in dataset_111.var.index:
    if g in g_to_p.keys():
        encode.append(g_to_p[g])
    else:
        encode.append(None)
encode = np.array(encode)
dataset_111.var["encode"] = encode
dataset_206.var["encode"] = encode

# highly variable and encode protein
dataset_111.var["hvg_encode"] = np.logical_or(
    (dataset_111.var["highly_variable"]).values, (encode != None)
)
dataset_206.var["hvg_encode"] = np.logical_or(
    (dataset_206.var["highly_variable"]).values, (encode != None)
)

# remove lowly expressed proteins
keep_proteins = dataset_111.obsm["protein_expression"].sum(axis=0) >= 1000
print("111 removing")
print(dataset_111.uns["protein_names"][~keep_proteins])
dataset_111.uns["protein_names"] = dataset_111.uns["protein_names"][keep_proteins]
dataset_111.obsm["protein_expression"] = dataset_111.obsm["protein_expression"][
    :, keep_proteins
]

keep_proteins = dataset_206.obsm["protein_expression"].sum(axis=0) >= 1000
print("206 removing")
print(dataset_206.uns["protein_names"][~keep_proteins])
dataset_206.uns["protein_names"] = dataset_206.uns["protein_names"][keep_proteins]
dataset_206.obsm["protein_expression"] = dataset_206.obsm["protein_expression"][
    :, keep_proteins
]

print("hvg_encode contains {} genes".format(np.sum(dataset_111.var["hvg_encode"])))

dataset_111.uns["version"] = {"version": 1.1}
dataset_206.uns["version"] = {"version": 1.1}
dataset_111.write("data/spleen_lymph_111.h5ad", compression="lzf")
dataset_206.write("data/spleen_lymph_206.h5ad", compression="lzf")
