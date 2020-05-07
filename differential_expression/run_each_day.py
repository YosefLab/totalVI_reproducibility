import os
import seaborn as sns
import numpy as np
import torch

from scvi.dataset import CellMeasurement, AnnDatasetFromAnnData
from scvi.models import TOTALVI
from scvi.inference import TotalPosterior, TotalTrainer

import anndata

from scvi import set_seed

sns.set(context="notebook", font_scale=1.15, style="ticks")
save_path = "/data/yosef2/users/adamgayoso/projects/totalVI_journal/data/"
colors = ["#3B7EA1", "#FDB515", "#D9661F", "#859438", "#EE1F60", "#00A598", "#CFDD45"]
sns.set_palette(sns.color_palette(colors))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Load anndata
anndataset_111 = anndata.read(save_path + "spleen_lymph_111.h5ad")
anndataset_206 = anndata.read(save_path + "spleen_lymph_206.h5ad")

# Remove hashtags
keep_pro_111 = np.array(
    [not p.startswith("HTO") for p in anndataset_111.uns["protein_names"]]
)
anndataset_111.obsm["protein_expression"] = anndataset_111.obsm["protein_expression"][
    :, keep_pro_111
]
anndataset_111.uns["protein_names"] = anndataset_111.uns["protein_names"][keep_pro_111]
keep_pro_206 = np.array(
    [not p.startswith("HTO") for p in anndataset_206.uns["protein_names"]]
)
anndataset_206.obsm["protein_expression"] = anndataset_206.obsm["protein_expression"][
    :, keep_pro_206
]
anndataset_206.uns["protein_names"] = anndataset_206.uns["protein_names"][keep_pro_206]

adatas = []
for b in np.unique(anndataset_111.obs["batch_indices"]):
    adatas.append(anndataset_111[anndataset_111.obs["batch_indices"] == b, :].copy())
    adatas[-1].obs["batch_indices"] *= 0
for b in np.unique(anndataset_206.obs["batch_indices"]):
    adatas.append(anndataset_206[anndataset_206.obs["batch_indices"] == b, :].copy())
    adatas[-1].obs["batch_indices"] *= 0

names = ["111_d1", "111_d2", "206_d1", "206_d2"]

# Iterate over datasets
for n, adata in zip(names, adatas):
    hvg = adata.var["hvg_encode"]

    dataset = AnnDatasetFromAnnData(ad=adata[:, hvg])
    protein_data = CellMeasurement(
        name="protein_expression",
        data=adata.obsm["protein_expression"].astype(np.float32),
        columns_attr_name="protein_names",
        columns=adata.uns["protein_names"],
    )
    dataset.initialize_cell_measurement(protein_data)
    dataset.gene_names = adata[:, hvg].var_names.values
    
    set_seed(0)

    model = TOTALVI(dataset.nb_genes, dataset.protein_expression.shape[1], n_latent=20,)
    use_cuda = True
    lr = 4e-3
    early_stopping_kwargs = {
        "early_stopping_metric": "elbo",
        "save_best_state_metric": "elbo",
        "patience": 45,
        "threshold": 0,
        "reduce_lr_on_plateau": True,
        "lr_patience": 30,
        "lr_factor": 0.6,
        "posterior_class": TotalPosterior,
    }

    trainer = TotalTrainer(
        model,
        dataset,
        train_size=0.9,
        test_size=0.1,
        use_cuda=use_cuda,
        frequency=1,
        data_loader_kwargs={"batch_size": 256, "pin_memory": False},
        early_stopping_kwargs=early_stopping_kwargs,
    )
    trainer.train(lr=lr, n_epochs=500)
    # create posterior on full data
    full_posterior = trainer.create_posterior(
        model, dataset, indices=np.arange(len(dataset)), type_class=TotalPosterior,
    )

    torch.save(
        trainer.model.state_dict(), "differential_expression/saved_models/" + n + ".pt"
    )
