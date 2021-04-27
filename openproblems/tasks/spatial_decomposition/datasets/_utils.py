from .._utils import obs_means
from anndata import AnnData
from itertools import accumulate
from scipy.sparse import csr_matrix
from typing import Optional
from typing import Union

import numpy as np
import random
import scanpy as sc


def multinomial(N: int, pvals: Union[np.ndarray, list]):
    """Multinomial sampling function

    Parameters
    ----------
    N : int
        Number of draws
    pvals : Union[np.ndarray,list]
        probability vector

    Returns:
    -------
    Numpy array representing a
    sample from Mult(n,p)

    """

    n = len(pvals)
    res = [0] * n

    def ss(x, y):
        return x + y

    pv = list(accumulate(pvals))

    for i in range(N):
        r = random.random() * pv[-1]
        for j in range(n):
            if pv[j] >= r:
                res[j] += 1
                break

    return np.array(res)


# pass the reference data
def generate_synthetic_dataset(
    adata: AnnData,
    sim_type: str = "avg",
    seed: int = 42,
    bead_depth: int = 1000,
    num_of_beads: Optional[int] = None,
):
    """Create cell-aggregate samples for ground-truth spatial decomposition task.

    Parameters
    ----------
    adata : AnnData
        Anndata object.
    sim_type : str
        Simulation type: either average `'avg'` or per cell `'cell'`.
    seed: int
        Seed for rng.

    Returns
    -------
    AnnData with:
        - `adata_spatial.obsm["proportions_true"]`: true proportion values.
        - `adata_spatial.X`: simulated counts (aggregate of sc dataset).
        - `adata_spatial.uns["sc_reference"]`: original sc adata for reference.
    """

    rng = np.random.default_rng(seed)
    random.seed(seed)

    adata.obs["label"] = adata.obs.label.astype("category")

    if isinstance(adata.X, csr_matrix):
        adata.X = adata.X.todense()

    n_genes = adata.shape[1]
    n_cells = adata.shape[0]
    n_types = len(set(adata.obs["label"].values))

    if num_of_beads is None:
        num_of_beads = n_cells * 2

    # generate proportion values
    props = rng.dirichlet(np.ones(n_types), num_of_beads)

    true_proportion = np.zeros((num_of_beads, n_types))
    bead_to_gene_matrix = np.zeros((num_of_beads, n_genes))

    # if sim_type avg
    # generate from avg profiles
    if sim_type == "avg":
        profile_mean = obs_means(adata, "label")
        sc.pp.normalize_total(profile_mean, target_sum=1, inplace=True)
        # run for each bead
        for bead_index in range(num_of_beads):
            # allocation = rng.multinomial(bead_depth, props[bead_index, :], size=1)[0]
            allocation = multinomial(bead_depth, props[bead_index, :])
            true_proportion[bead_index, :] = allocation.copy()
            for j in range(n_types):
                gene_exp = multinomial(allocation[j], profile_mean.X[j, :])
                bead_to_gene_matrix[bead_index, :] += gene_exp

    elif sim_type == "cell":
        # generate from cells
        # assign beads to actual cells
        # cell_ids with this cluster
        cells_to_sample_from_celltype = []
        grouped = adata.obs.groupby("label")
        for idx in grouped.indices.values():
            cells_to_sample_from_celltype += [idx]

        # Actual cells assigned randomly
        cell_association = np.zeros((num_of_beads, n_types)).astype(np.int)
        for j in range(n_types):
            cell_association[:, j] = rng.integers(
                low=0, high=len(cells_to_sample_from_celltype[j]), size=num_of_beads
            )

        counts = np.array(adata.X)
        rowSums = counts.sum(axis=1, keepdims=True)
        X_norm_prof = np.divide(counts, rowSums, where=rowSums > 0)
        for bead_index in range(num_of_beads):
            allocation = multinomial(bead_depth, props[bead_index, :])
            true_proportion[bead_index, :] = allocation
            for j in range(n_types):
                cell_index = cells_to_sample_from_celltype[j][
                    cell_association[bead_index, j]
                ]
                gene_exp = multinomial(allocation[j], X_norm_prof[cell_index, :])
                bead_to_gene_matrix[bead_index, :] += gene_exp
    else:
        raise ValueError(f"{sim_type} is not a valid key for `sim_type`.")

    bead_barcodes = np.arange(num_of_beads)

    adata_spatial = AnnData(
        bead_to_gene_matrix,
        obs=dict(obs_names=bead_barcodes),
        var=dict(var_names=adata.var_names),
    )

    true_proportion = true_proportion / true_proportion.sum(1)[:, np.newaxis].astype(
        "float64"
    )

    # fake coordinates
    adata_spatial.obsm["spatial"] = rng.random((adata_spatial.shape[0], 2))
    adata_spatial.obsm["proportions_true"] = true_proportion

    adata_spatial.uns["sc_reference"] = adata.copy()

    return adata_spatial
