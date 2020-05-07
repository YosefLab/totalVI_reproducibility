from typing import Dict, Optional
import numpy as np
from scvi.inference import TotalPosterior, Posterior
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import pandas as pd
from tqdm.auto import tqdm
import sparse
import tempfile
import collections
import os


class FileDict(collections.abc.MutableMapping):
    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir="./")
        self.temp_dir_name = self.temp_dir.name
        self.store = dict()

    def __getitem__(self, key):
        save_path = self._get_save_path(key)
        val = sparse.load_npz(save_path)
        return val

    def __setitem__(self, key, val):
        save_path = self._get_save_path(key)
        self.store[key] = save_path
        sparse.save_npz(save_path, val)

    def __delitem__(self, key):
        save_path = self._get_save_path(key)
        del self.store[key]
        os.remove(save_path)

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __del__(self):
        self.temp_dir.cleanup()

    def _get_save_path(self, key):
        save_path = self.temp_dir_name + "/" + key + ".npz"
        return save_path


class TotalPosteriorPredictiveCheck:
    """Posterior predictive checks for comparing totalVI models
    """

    def __init__(
        self,
        posteriors_dict: Dict[str, TotalPosterior],
        scvi_posts_dict: Optional[Dict[str, Posterior]] = None,
        n_samples: int = 10,
        batch_size=32,
    ):
        """
        Args:
            posteriors_dict (Dict[str, Posterior]): dictionary of Posterior objects fit on the same dataset
            n_samples (int, optional): Number of posterior predictive samples. Defaults to 10.
        """
        self.posteriors = posteriors_dict
        self.dataset = posteriors_dict[next(iter(posteriors_dict.keys()))].gene_dataset
        self.raw_counts = None
        self.posterior_predictive_samples = FileDict()
        self.n_samples = n_samples
        self.models = {}
        self.metrics = {}
        self.raw_metrics = {}
        self.batch_size = batch_size
        self.scvi_posts_dict = scvi_posts_dict

        self.store_posterior_samples()

    def store_posterior_samples(self):
        """Samples from the Posterior objects and sets raw_counts
        """

        for m, post in self.posteriors.items():
            pp_counts, original = post.sequential().generate(
                n_samples=self.n_samples, batch_size=self.batch_size
            )
            self.posterior_predictive_samples[m] = sparse.COO(pp_counts)
        self.raw_counts = sparse.COO(original)

        if self.scvi_posts_dict is not None:
            for m, post in self.scvi_posts_dict.items():
                pp_counts, original = post.sequential().generate(
                    n_samples=self.n_samples, batch_size=self.batch_size
                )
                self.posterior_predictive_samples[m] = sparse.COO(pp_counts)

    def coeff_of_variation(self, cell_wise: bool = False):
        """Calculate the coefficient of variation

        Args:
            cell_wise (bool, optional): Calculate for each cell across genes. Defaults to True.
                                        If False, calculate for each gene across cells.
        """
        axis = 1 if cell_wise is True else 0
        if cell_wise is True:
            raise NotImplementedError("cell_wise=True is not yet implemented")
        identifier = "cv_cell" if cell_wise is True else "cv_gene"
        temp_raw_counts = self.raw_counts.todense()
        df = pd.DataFrame()
        for m in self.posterior_predictive_samples:
            samples = self.posterior_predictive_samples[m].todense()
            cv = np.nanmean(
                np.std(samples, axis=axis) / np.mean(samples, axis=axis), axis=-1
            )
            # make all zeros have 0 cv
            cv = np.nan_to_num(cv)
            if samples.shape[1] == self.dataset.nb_genes:
                cv = np.concatenate(
                    [
                        cv.ravel(),
                        np.nan
                        * np.zeros((self.raw_counts.shape[1] - self.dataset.nb_genes)),
                    ]
                )

            df[m] = cv.ravel()

        df["raw"] = np.std(temp_raw_counts, axis=axis) / np.mean(
            temp_raw_counts, axis=axis
        )
        df["raw"] = np.nan_to_num(df["raw"])

        self.metrics[identifier] = df

    def median_absolute_error(self, point_estimate="mean"):
        df = pd.DataFrame()
        temp_raw_counts = self.raw_counts.todense()
        for m in self.posterior_predictive_samples:
            samples = self.posterior_predictive_samples[m].todense()
            if point_estimate == "mean":
                point_sample = np.mean(samples, axis=-1)
            else:
                point_sample = np.median(samples, axis=-1)
            mad_gene = np.median(
                np.abs(
                    point_sample[:, : self.dataset.nb_genes]
                    - temp_raw_counts[:, : self.dataset.nb_genes]
                )
            )
            # For samples with protein data
            if point_sample.shape[1] != self.dataset.nb_genes:
                mad_pro = np.median(
                    np.abs(
                        point_sample[:, self.dataset.nb_genes :]
                        - temp_raw_counts[:, self.dataset.nb_genes :]
                    )
                )
            else:
                mad_pro = np.nan
            df[m] = [mad_gene, mad_pro]

        df.index = ["genes", "proteins"]
        self.metrics["mae"] = df

    def mean_squared_error(self, point_estimate="mean"):
        df = pd.DataFrame()
        temp_raw_counts = self.raw_counts.todense()
        for m in self.posterior_predictive_samples:
            samples = self.posterior_predictive_samples[m].todense()
            if point_estimate == "mean":
                point_sample = np.mean(samples, axis=-1)
            else:
                point_sample = np.median(samples, axis=-1)
            mse_gene = np.mean(
                np.square(
                    point_sample[:, : self.dataset.nb_genes]
                    - temp_raw_counts[:, : self.dataset.nb_genes]
                )
            )
            # For samples with protein data
            if point_sample.shape[1] != self.dataset.nb_genes:
                mse_pro = np.mean(
                    np.square(
                        point_sample[:, self.dataset.nb_genes :]
                        - temp_raw_counts[:, self.dataset.nb_genes :]
                    )
                )
            else:
                mse_pro = np.nan
            df[m] = [mse_gene, mse_pro]

        df.index = ["genes", "proteins"]
        self.metrics["mse"] = df

    def mean_absolute_error(self, point_estimate="mean"):
        df = pd.DataFrame()
        temp_raw_counts = self.raw_counts.todense()
        for m in self.posterior_predictive_samples:
            samples = self.posterior_predictive_samples[m].todense()
            if point_estimate == "mean":
                point_sample = np.mean(samples, axis=-1)
            else:
                point_sample = np.median(samples, axis=-1)
            mse_gene = np.mean(
                np.abs(
                    point_sample[:, : self.dataset.nb_genes]
                    - temp_raw_counts[:, : self.dataset.nb_genes]
                )
            )
            # For samples with protein data
            if point_sample.shape[1] != self.dataset.nb_genes:
                mse_pro = np.mean(
                    np.abs(
                        point_sample[:, self.dataset.nb_genes :]
                        - temp_raw_counts[:, self.dataset.nb_genes :]
                    )
                )
            else:
                mse_pro = np.nan
            df[m] = [mse_gene, mse_pro]

        df.index = ["genes", "proteins"]
        self.metrics["mean_ae"] = df

    def mean_squared_log_error(self, point_estimate="mean"):
        df = pd.DataFrame()
        temp_raw_counts = self.raw_counts.todense()
        for m in self.posterior_predictive_samples:
            samples = self.posterior_predictive_samples[m].todense()
            if point_estimate == "mean":
                point_sample = np.mean(samples, axis=-1)
            else:
                point_sample = np.mean(samples, axis=-1)
            mad_gene = np.mean(
                np.square(
                    np.log(point_sample[:, : self.dataset.nb_genes] + 1)
                    - np.log(temp_raw_counts[:, : self.dataset.nb_genes] + 1)
                )
            )
            if point_sample.shape[1] != self.dataset.nb_genes:
                mad_pro = np.mean(
                    np.square(
                        np.log(point_sample[:, self.dataset.nb_genes :] + 1)
                        - np.log(temp_raw_counts[:, self.dataset.nb_genes :] + 1)
                    )
                )
            else:
                mad_pro = np.nan
            df[m] = [mad_gene, mad_pro]

        df.index = ["genes", "proteins"]
        self.metrics["msle"] = df

    def dropout_ratio(self):
        """Fraction of zeros in raw_counts for a specific gene
        """
        df = pd.DataFrame()
        for m in self.posterior_predictive_samples:
            samples = self.posterior_predictive_samples[m]
            dr = np.mean(np.mean(samples == 0, axis=0), axis=-1)
            df[m] = dr.ravel()

        df["raw"] = np.mean(np.mean(self.raw_counts == 0, axis=0), axis=-1)

        self.metrics["dropout_ratio"] = df

    def store_fa_samples(
        self,
        train_data,
        train_indices,
        test_indices,
        key="Factor Analysis",
        normalization="log",
        clip_zero=True,
        **kwargs
    ):
        # reconstruction
        if normalization == "log":
            train_data = np.log(train_data + 1)
            data = np.log(self.raw_counts.todense() + 1)
            key += " (Log)"
        elif normalization == "log_rate":
            train_data_rna = train_data[:, : self.dataset.nb_genes]
            train_data_pro = train_data[:, self.dataset.nb_genes :]
            train_data_rna = np.log(
                10000 * train_data_rna / train_data_rna.sum(axis=1)[:, np.newaxis] + 1
            )
            train_data_pro = np.log(
                10000 * train_data_pro / train_data_pro.sum(axis=1)[:, np.newaxis] + 1
            )
            train_data = np.concatenate([train_data_rna, train_data_pro], axis=1)
            lib_size_rna = self.raw_counts.todense()[:, : self.dataset.nb_genes].sum(
                axis=1
            )[:, np.newaxis]
            lib_size_pro = self.raw_counts.todense()[:, self.dataset.nb_genes :].sum(
                axis=1
            )[:, np.newaxis]

            data = np.concatenate(
                [
                    np.log(
                        10000
                        * self.raw_counts.todense()[:, : self.dataset.nb_genes]
                        / lib_size_rna
                        + 1
                    ),
                    np.log(
                        10000
                        * self.raw_counts.todense()[:, self.dataset.nb_genes :]
                        / lib_size_pro
                        + 1
                    ),
                ],
                axis=1,
            )
            key += " (Log Rate)"
        else:
            train_data = train_data
            data = self.raw_counts.todense()
        fa = FactorAnalysis(**kwargs)
        fa.fit(train_data)
        self.models[key] = fa

        # transform gives the posterior mean
        z_mean = fa.transform(data)
        Ih = np.eye(len(fa.components_))
        # W is n_components by n_features, code below from sklearn implementation
        Wpsi = fa.components_ / fa.noise_variance_
        z_cov = linalg.inv(Ih + np.dot(Wpsi, fa.components_.T))

        # sample z's
        z_samples = np.random.multivariate_normal(
            np.zeros(fa.n_components, dtype=np.float32),
            cov=z_cov,
            size=(self.raw_counts.shape[0], self.n_samples),
        )
        # cells by n_components by posterior samples
        z_samples = np.swapaxes(z_samples, 1, 2)
        # add mean to all samples
        z_samples += z_mean[:, :, np.newaxis]

        x_samples = np.zeros(
            (self.raw_counts.shape[0], self.raw_counts.shape[1], self.n_samples),
            dtype=np.float32,
        )
        for i in range(self.n_samples):
            x_mean = np.matmul(z_samples[:, :, i], fa.components_)
            x_sample = np.random.normal(x_mean, scale=np.sqrt(fa.noise_variance_))
            # add back feature means
            x_samples[:, :, i] = x_sample + fa.mean_

        reconstruction = x_samples

        if normalization == "log":
            reconstruction = np.exp(reconstruction) - 1
        if normalization == "log_rate":
            reconstruction = np.concatenate(
                [
                    lib_size_rna[:, :, np.newaxis]
                    / 10000
                    * (np.exp(reconstruction[:, : self.dataset.nb_genes]) - 1),
                    lib_size_pro[:, :, np.newaxis]
                    / 10000
                    * (np.exp(reconstruction[:, self.dataset.nb_genes :]) - 1),
                ],
                axis=1,
            )
        if clip_zero is True:
            reconstruction[reconstruction < 0] = 0

        self.posterior_predictive_samples[key] = sparse.COO(reconstruction)

    def store_pca_samples(self, key="PCA", normalization="log", **kwargs):
        # reconstruction
        if normalization == "log":
            data = np.log(self.raw_counts + 1)
            key += " (Log)"
        else:
            data = self.raw_counts
        pca = PCA(**kwargs)
        pca.fit(data)
        self.models[key] = pca

        # Using Bishop notation section 12.2, M is comp x comp
        # W is fit using MLE, samples generated using posterior predictive
        M = (
            np.matmul(pca.components_, pca.components_.T)
            + np.identity(pca.n_components) * pca.noise_variance_
        )
        z_mean = np.matmul(
            np.matmul(linalg.inv(M), pca.components_), (self.raw_counts - pca.mean_).T
        ).T
        z_cov = linalg.inv(M) * pca.noise_variance_

        # sample z's
        z_samples = np.random.multivariate_normal(
            np.zeros(pca.n_components),
            cov=z_cov,
            size=(self.raw_counts.shape[0], self.n_samples),
        )
        # cells by n_components by posterior samples
        z_samples = np.swapaxes(z_samples, 1, 2)
        # add mean to all samples
        z_samples += z_mean[:, :, np.newaxis]

        x_samples = np.zeros(
            (self.raw_counts.shape[0], self.raw_counts.shape[1], self.n_samples)
        )
        for i in range(self.n_samples):
            x_mean = np.matmul(z_samples[:, :, i], pca.components_)
            x_sample = np.random.normal(x_mean, scale=np.sqrt(pca.noise_variance_))
            # add back feature means
            x_samples[:, :, i] = x_sample + pca.mean_

        reconstruction = x_samples

        if normalization == "log":
            reconstruction = np.clip(reconstruction, -1000, 20)
            reconstruction = np.exp(reconstruction - 1)

        self.posterior_predictive_samples[key] = reconstruction

    def protein_gene_correlation(self, n_genes=1000):
        self.gene_set = np.random.choice(
            self.dataset.nb_genes, size=n_genes, replace=False
        )
        model_corrs = {}
        for m in tqdm(self.posterior_predictive_samples):
            samples = self.posterior_predictive_samples[m].todense()
            correlation_matrix = np.zeros((n_genes, len(self.dataset.protein_names)))
            for i in range(self.n_samples):
                sample = StandardScaler().fit_transform(samples[:, :, i])
                gene_sample = sample[:, self.gene_set]
                protein_sample = sample[:, self.dataset.nb_genes :]

                correlation_matrix += np.matmul(gene_sample.T, protein_sample)
            correlation_matrix /= self.n_samples
            correlation_matrix /= self.raw_counts.shape[0] - 1
            model_corrs[m] = correlation_matrix.ravel()

        scaled_raw_counts = StandardScaler().fit_transform(self.raw_counts.todense())
        scaled_genes = scaled_raw_counts[:, self.gene_set]
        scaled_proteins = scaled_raw_counts[:, self.dataset.nb_genes :]
        raw_count_corr = np.matmul(scaled_genes.T, scaled_proteins)
        raw_count_corr /= self.raw_counts.shape[0] - 1
        model_corrs["raw"] = raw_count_corr.ravel()

        model_corrs["protein_names"] = list(self.dataset.protein_names) * n_genes
        model_corrs["gene_names"] = np.repeat(
            self.dataset.gene_names[self.gene_set], len(self.dataset.protein_names)
        )

        df = pd.DataFrame.from_dict(model_corrs)
        self.metrics["all protein-gene correlations"] = df

    def calibration_error(self, confidence_intervals=None):
        if confidence_intervals is None:
            ps = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 82.5, 85, 87.5, 90, 92.5, 95, 97.5]
        else:
            ps = confidence_intervals
        reverse_ps = ps[::-1]
        model_cal = {}
        temp_raw_counts = self.raw_counts.todense()
        for m in self.posterior_predictive_samples:
            samples = self.posterior_predictive_samples[m].todense()
            percentiles = np.percentile(samples, ps, axis=2)
            reverse_percentiles = percentiles[::-1]
            cal_error_genes = 0
            cal_error_proteins = 0
            cal_error_total = 0
            for i, j, truth, reverse_truth in zip(
                percentiles, reverse_percentiles, ps, reverse_ps
            ):
                if truth > reverse_truth:
                    break
                true_width = (100 - truth * 2) / 100
                # For gene only model
                if samples.shape[1] == self.dataset.nb_genes:
                    ci = np.logical_and(
                        temp_raw_counts[:, : self.dataset.nb_genes] >= i,
                        temp_raw_counts[:, : self.dataset.nb_genes] <= j,
                    )
                    cal_error_proteins = np.nan
                    cal_error_total = np.nan
                else:
                    ci = np.logical_and(temp_raw_counts >= i, temp_raw_counts <= j)
                    pci_proteins = np.mean(ci[:, self.dataset.nb_genes :])
                    pci_total = np.mean(ci)
                    cal_error_proteins += (pci_proteins - true_width) ** 2
                    cal_error_total += (pci_total - true_width) ** 2
                pci_genes = np.mean(ci[:, : self.dataset.nb_genes])
                cal_error_genes += (pci_genes - true_width) ** 2
            model_cal[m] = {
                "genes": cal_error_genes,
                "proteins": cal_error_proteins,
                "total": cal_error_total,
            }
        self.metrics["calibration"] = pd.DataFrame.from_dict(model_cal)
