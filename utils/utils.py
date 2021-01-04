import scanpy as sc
from scvi.inference import TotalPosterior
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
import scipy
import torch
from tqdm.auto import tqdm
import statsmodels.api as sm
import phenograph
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    silhouette_samples,
)
import hotspot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def rank_genes_groups_totalVI(
    adata: sc.AnnData,
    scvi_posterior: TotalPosterior,
    n_samples: int = None,
    M_permutation: int = None,
    n_features: int = 25,
    protein_only: bool = True,
    gene_only: bool = False,
    label_name: str = "louvain_scvi",
) -> pd.DataFrame:
    """
    Rank genes for characterizing groups.
    Computes Bayes factor for each cluster against the others to test for differential expression.
    See Nature article (https://rdcu.be/bdHYQ)

    :param adata: sc.AnnData object non-normalized
    :param scvi_posterior:
    :param n_samples:
    :param M_permutation:
    :param n_genes:
    :param label_name: The groups tested are taken from adata.obs[label_name] which can be computed
                       using clustering like Louvain (Ex: sc.tl.louvain(adata, key_added=label_name) )
    :return: Summary of Bayes factor per gene, per cluster
    """

    # Call scvi function
    per_cluster_de, cluster_id = scvi_posterior.one_vs_all_degenes(
        cell_labels=np.asarray(adata.obs[label_name].values).astype(int).ravel(),
        min_cells=1,
        n_samples=n_samples,
        M_permutation=M_permutation,
    )

    # convert to ScanPy format -- this is just about feeding scvi results into a format readable by ScanPy
    markers = []
    scores = []
    names = []
    for i, x in enumerate(per_cluster_de):
        subset_de = x[:n_features]
        markers.append(subset_de)
        scores.append(tuple(subset_de["bayes_factor"].values))
        names.append(tuple(subset_de.index.values))

    markers = pd.concat(markers)
    dtypes_scores = [(str(i), "<f4") for i in range(len(scores))]
    dtypes_names = [(str(i), "<U50") for i in range(len(names))]
    scores = np.array([tuple(row) for row in np.array(scores).T], dtype=dtypes_scores)
    scores = scores.view(np.recarray)
    names = np.array([tuple(row) for row in np.array(names).T], dtype=dtypes_names)
    names = names.view(np.recarray)

    adata.uns["rank_genes_groups_totalVI"] = {
        "params": {
            "groupby": "",
            "reference": "rest",
            "method": "",
            "use_raw": True,
            "corr_method": "",
        },
        "scores": scores,
        "names": names,
    }
    return markers


def entropy_batch_mixing(
    latent_space, batches, n_neighbors=50, n_pools=50, n_samples_per_pool=100
):
    def entropy(hist_data):
        n_batches = len(np.unique(hist_data))
        if n_batches > 2:
            raise ValueError("Should be only two clusters for this metric")
        frequency = np.mean(hist_data == 1)
        if frequency == 0 or frequency == 1:
            return 0
        return -frequency * np.log(frequency) - (1 - frequency) * np.log(1 - frequency)

    def neg_kl(hist_data, global_freq):
        n_batches = len(np.unique(hist_data))
        if n_batches > 2:
            raise ValueError("Should be only two clusters for this metric")
        frequency = np.mean(hist_data == 1)
        if frequency == 0 or frequency == 1:
            return 0
        return -(
            frequency * np.log(frequency / global_freq)
            + (1 - frequency) * np.log((1 - frequency) / (1 - global_freq))
        )

    n_neighbors = min(n_neighbors, len(latent_space) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(latent_space)
    kmatrix = nne.kneighbors_graph(latent_space) - scipy.sparse.identity(
        latent_space.shape[0]
    )

    global_freq = np.mean(batches)
    print(global_freq)
    score = 0
    for t in range(n_pools):
        indices = np.random.choice(
            np.arange(latent_space.shape[0]), size=n_samples_per_pool
        )
        score += np.mean(
            [
                neg_kl(
                    batches[  # the batches of cell i's neighbors
                        kmatrix[indices].nonzero()[
                            1
                        ][  # the neighbors of cell i (columns in row i)
                            kmatrix[indices].nonzero()[0] == i  # the row of cell i
                        ]
                    ],
                    global_freq,
                )
                for i in range(n_samples_per_pool)
            ]
        )
    return score / float(n_pools)


def clustering_metric_silhoutte(
    adata1,
    adata2,
    adata,
    batchid,
    metric="euclidean",
    k=30,
    use_rep="X_pca",
    resolution=0.8,
    n_clusters=25,
):

    sc.pp.neighbors(adata1, n_neighbors=k, use_rep=use_rep, metric=metric)
    sc.pp.neighbors(adata2, n_neighbors=k, use_rep=use_rep, metric=metric)
    adata_joint_1 = adata[batchid == 0].copy()
    adata_joint_2 = adata[batchid == 1].copy()
    sc.pp.neighbors(adata_joint_1, n_neighbors=k, use_rep=use_rep, metric=metric)
    sc.pp.neighbors(adata_joint_2, n_neighbors=k, use_rep=use_rep, metric=metric)

    sc.tl.leiden(adata1, key_added="leiden_clus_metric", resolution=resolution)
    sc.tl.leiden(adata2, key_added="leiden_clus_metric", resolution=resolution)

    cluster1 = adata1.obs["leiden_clus_metric"].astype(int).values
    cluster2 = adata2.obs["leiden_clus_metric"].astype(int).values

    batch1_score = silhouette_samples(
        adata_joint_1.uns["neighbors"]["distances"], cluster1
    )
    batch2_score = silhouette_samples(
        adata_joint_2.uns["neighbors"]["distances"], cluster2
    )

    batch1_score_same = silhouette_score(adata1.uns["neighbors"]["distances"], cluster1)

    batch2_score_same = silhouette_score(adata2.uns["neighbors"]["distances"], cluster2)

    return (
        np.mean(batch1_score - batch1_score_same),
        np.mean(batch2_score - batch2_score_same),
    )


def hotspot_score(
    full_adata,
    latent1,
    latent2,
    latent_joint,
    batches,
    k=30,
    subset_features=None,
    weighted_graph=True,
):
    if subset_features is not None:
        full_adata = full_adata[:, subset_features].copy()
    scaler = StandardScaler()
    joint_data = scaler.fit_transform(full_adata.X).T
    joint_data = pd.DataFrame(joint_data, index=full_adata.var_names)
    b1_data = pd.DataFrame(
        scaler.fit_transform(full_adata.X[batches == 0]).T, index=full_adata.var_names
    )
    b2_data = pd.DataFrame(
        scaler.fit_transform(full_adata.X[batches == 1]).T, index=full_adata.var_names
    )

    hs_joint_1 = hotspot.Hotspot(
        b1_data, model="none", latent=pd.DataFrame(latent_joint[batches == 0]),
    )

    hs_joint_1.create_knn_graph(
        weighted_graph=weighted_graph, n_neighbors=k,
    )
    hs_joint_1_results = hs_joint_1.compute_autocorrelations(jobs=10)

    hs_joint_2 = hotspot.Hotspot(
        b2_data, model="none", latent=pd.DataFrame(latent_joint[batches == 1]),
    )

    hs_joint_2.create_knn_graph(
        weighted_graph=weighted_graph, n_neighbors=k,
    )
    hs_joint_2_results = hs_joint_2.compute_autocorrelations(jobs=10)

    hs_1 = hotspot.Hotspot(b1_data, model="none", latent=pd.DataFrame(latent1),)

    hs_1.create_knn_graph(
        weighted_graph=weighted_graph, n_neighbors=k,
    )
    hs_1_results = hs_1.compute_autocorrelations(jobs=10)

    hs_2 = hotspot.Hotspot(b2_data, model="none", latent=pd.DataFrame(latent2),)

    hs_2.create_knn_graph(
        weighted_graph=weighted_graph, n_neighbors=k,
    )
    hs_2_results = hs_2.compute_autocorrelations(jobs=10)

    hs_1_results = hs_1_results.loc[hs_joint_1_results.index]
    hs_2_results = hs_2_results.loc[hs_joint_2_results.index]

    res1 = np.mean((hs_joint_1_results - hs_1_results)["Z"])
    print(np.var((hs_joint_1_results - hs_1_results)["Z"]))
    print(np.var((hs_joint_2_results - hs_2_results)["Z"]))
    res2 = np.mean((hs_joint_2_results - hs_2_results)["Z"])

    return 0.5 * (res1 + res2), hs_joint_1_results, hs_1_results


def seurat_v3_highly_variable_genes(adata, n_top_genes=4000, use_lowess=False):
    norm_gene_vars = []
    del_batch = False
    if "batch" not in adata.obs_keys():
        del_batch = True
        adata.obs["batch"] = np.zeros((adata.X.shape[0]))
    for b in np.unique(adata.obs["batch"]):
        var = adata[adata.obs["batch"] == b].X.var(0)
        print(var.shape)
        mean = adata[adata.obs["batch"] == b].X.mean(0)
        estimat_var = np.zeros((adata.X.shape[1]))

        y = np.log10(var)
        x = np.log10(mean)
        if use_lowess is True:
            lowess = sm.nonparametric.lowess
            # output is sorted by x
            v = lowess(y, x, frac=0.15)
            estimat_var[np.argsort(x)] = v[:, 1]
        else:
            estimat_var = loess(y, x)

        norm_values = (adata[adata.obs["batch"] == b].X - mean) / np.sqrt(
            10 ** estimat_var
        )
        print(norm_values.shape)
        # as in seurat paper, clip max values
        norm_values = np.clip(
            norm_values, None, np.sqrt(np.sum(adata.obs["batch"] == b))
        )
        norm_gene_var = norm_values.var(0)
        norm_gene_vars.append(norm_gene_var.reshape(1, -1))

    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    ranked_norm_gene_vars = np.argsort(np.argsort(norm_gene_vars, axis=1), axis=1)
    mean_norm_gene_vars = np.mean(norm_gene_vars, axis=0)
    median_ranked = np.median(ranked_norm_gene_vars, axis=0)

    num_batches_high_var = np.sum(
        ranked_norm_gene_vars >= (adata.X.shape[1] - n_top_genes), axis=0
    )
    df = pd.DataFrame(index=np.array(adata.var_names))
    df["highly_variable_n_batches"] = num_batches_high_var
    df["highly_variable_median_rank"] = median_ranked

    df["highly_variable_mean_variance"] = mean_norm_gene_vars
    df.sort_values(
        ["highly_variable_n_batches", "highly_variable_median_rank"],
        ascending=False,
        na_position="last",
        inplace=True,
    )
    df["highly_variable"] = False
    df.loc[:n_top_genes, "highly_variable"] = True
    df = df.loc[adata.var_names]

    if del_batch is True:
        del adata.obs["batch"]

    adata.var["highly_variable"] = df["highly_variable"].values
    adata.var["highly_variable_n_batches"] = df["highly_variable_n_batches"].values
    adata.var["highly_variable_mean_variance"] = df[
        "highly_variable_mean_variance"
    ].values


def loess(y, x, span=0.3):
    from rpy2.robjects import r
    import rpy2.robjects as robjects

    a, b = robjects.FloatVector(x), robjects.FloatVector(y)
    df = robjects.DataFrame({"a": a, "b": b})
    loess_fit = r.loess("b ~ a", data=df, span=span)

    return np.array(loess_fit[loess_fit.names.index("fitted")])


def glm_de(
    protein_adata,
    cell_type_1,
    cell_type_2,
    cell_type_key="cell_types",
    batch_key="batch",
    use_raw=True,
    family="Gaussian",
):
    group1 = protein_adata.obs[cell_type_key] == cell_type_1
    group2 = protein_adata.obs[cell_type_key] == cell_type_2
    groups = np.logical_or(group1, group2).ravel()

    adata_sub = protein_adata.raw.X[groups]
    labels_sub = protein_adata.obs[cell_type_key].values[groups]
    batch_sub = protein_adata.obs[batch_key].values[groups].ravel()
    cell_types_sub = labels_sub == cell_type_1
    pvals, coefs = _glm_fit(adata_sub, batch_sub, cell_types_sub, family=family)

    _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    df = pd.DataFrame(
        index=protein_adata.var_names, columns=["pvals", "pvals_adj", "coeff"]
    )
    df["pvals"] = pvals
    df["pvals_adj"] = pvals_adj
    df["coeff"] = coefs
    df = df.sort_values("coeff", ascending=False)

    return df


def _glm_fit(protein_data, batch, cell_types, family="Gaussian"):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # const = np.ones((len(protein_data), 1))
    df = pd.DataFrame()
    df["cell_type_1"] = cell_types
    # ct = cell_types.reshape(-1, 1)
    df["batch"] = batch.ravel() + 1

    p_vals = []
    coefs = []
    for p in tqdm(range(protein_data.shape[1])):
        # b = pd.get_dummies(batch).values
        # exog = np.concatenate([ct, b, const], axis=1)
        # print(exog.shape)
        if family == "Gaussian":
            f = sm.families.Gaussian()
            y = protein_data[:, p]
        elif family == "Poisson":
            f = sm.families.Poisson()
            y = np.expm1(protein_data[:, p])
        else:
            raise ValueError("incorrect family")
        df["expression"] = y

        # model = sm.GLM(y, exog, family=f)
        model = smf.glm("expression ~ C(cell_type_1) + C(batch)", data=df, family=f)
        model_results = model.fit()
        p_vals.append(model_results.pvalues[1])
        coefs.append(model_results.params[1])

    return np.array(p_vals), np.array(coefs)
