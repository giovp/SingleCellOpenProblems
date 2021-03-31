from .._utils import obs_means
from anndata import AnnData

import numpy as np
import scanpy as sc


# pass the reference data
def generate_synthetic_dataset(adata: AnnData, sim_type: str = "avg"):

    rng = np.random.default_rng(42)

    n_genes = adata.shape[1]
    n_cells = adata.shape[0]
    n_types = len(set(adata.obs["label"].values))

    # TODO(make these arguments)
    bead_depth = 1000
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
        clipped_x = np.clip(profile_mean.X.copy(), 0, 1)
        profile_mean.X = clipped_x
        # run for each bead
        for bead_index in range(num_of_beads):
            allocation = rng.multinomial(bead_depth, props[bead_index, :], size=1)[0]
            true_proportion[bead_index, :] = allocation.copy()
            for j in range(n_types):
                gene_exp = rng.multinomial(allocation[j], profile_mean.X[j, :], size=1)[
                    0
                ]
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
            cell_association[:, j] = rng.randint(
                0, len(cells_to_sample_from_celltype[j]), num_of_beads
            )

        X_norm_prof = (adata.X / adata.X.sum(1)[:, np.newaxis]).astype("float64")
        for bead_index in range(num_of_beads):
            allocation = rng.multinomial(bead_depth, props[bead_index, :], size=1)[0]
            true_proportion[bead_index, :] = allocation.copy()
            for j in range(n_types):
                cell_index = cells_to_sample_from_celltype[j][
                    cell_association[bead_index, j]
                ]
                gene_exp = rng.multinomial(
                    allocation[j], X_norm_prof[cell_index, :], size=1
                )[0]
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

    return adata
