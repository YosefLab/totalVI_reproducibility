import anndata
import pandas as pd

save_path = "/data/yosef2/users/adamgayoso/projects/totalVI_journal/data/"


post_adata = anndata.read("sln_all_intersect_post_adata.h5ad")

new_annots = pd.read_csv("annotations/annotations_subclustered.csv")

annotations = {
    "0": "CD4 T",
    "1": "Mature B",
    "2": "CD8 T",
    "3": "Transitional B",
    "4": "Mature B",
    "5": "Mature B",
    "6": "CD122+ CD8 T",
    "7": "Mature B",
    "8": "Ifit3-high B",
    "9": "Mature B",
    "10,0": "Tregs",
    "10,1": "ICOS-high Tregs",
    "10,2": "Tregs",
    "11": "MZ B",
    "12,0": "NKT",
    "12,1": "NK",
    "13": "B1 B",
    "14": "Ifit3-high CD4 T",
    "15,0": "cDC2s",
    "15,1": "cDC1s",
    "15,2": "Migratory DCs",
    "16,0": "B-macrophage doublets",
    "16,1": "MZ/Marco-high macrophages",
    "16,2": "Red-pulp macrophages",
    "16,3": "Erythrocytes",
    "17": "Low quality B cells",
    "18": "Ifit3-high CD8 T",
    "19": "Cycling B/T cells",
    "20,0": "Ly6-high mono",
    "20,1": "Ly6-low mono",
    "21": "B doublets",
    "22": "GD T",
    "23": "T doublets",
    "24,0": "B-CD8 T cell doublets",
    "24,1": "Erythrocytes",
    "24,2": "Low quality T cells",
    "25": "B-CD4 T cell doublets",
    "26": "Neutrophils",
    "27": "Activated CD4 T",
    "28": "pDCs",
    "29": "Low quality T cells",
    "30": "Neutrophils",
    "31": "Plasma B",
}
post_adata.obs["leiden_subclusters"] = pd.Categorical(new_annots["leiden_subclusters"].values.ravel())
post_adata.obs["labels"] = pd.Categorical([annotations[a] for a in new_annots["leiden_subclusters"].values.ravel()])


anndataset_111 = anndata.read(save_path + "spleen_lymph_111.h5ad")
anndataset_206 = anndata.read(save_path + "spleen_lymph_206.h5ad")

anndataset_111.obs["leiden_subclusters"] = post_adata.obs["leiden_subclusters"].values[
    : anndataset_111.X.shape[0]
]
anndataset_206.obs["leiden_subclusters"] = post_adata.obs["leiden_subclusters"].values[
    anndataset_111.X.shape[0] :
]

try:
    del anndataset_111.obs["labels"]
except KeyError:
    print("Nothing in labels field")

try:
    del anndataset_206.obs["labels"]
except KeyError:
    print("Nothing in labels field")

anndataset_111.obs["cell_types"] = post_adata.obs["labels"].values[
    : anndataset_111.X.shape[0]
]
anndataset_206.obs["cell_types"] = post_adata.obs["labels"].values[
    anndataset_111.X.shape[0] :
]

anndataset_111.write(save_path + "spleen_lymph_111.h5ad", compression="lzf")
anndataset_206.write(save_path + "spleen_lymph_206.h5ad", compression="lzf")
