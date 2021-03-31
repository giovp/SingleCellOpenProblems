from .._utils import obs_normalize
from anndata import AnnData

import numpy as np
import random


def generate_seq(length=14):
    return "".join(random.choice("CGTA") for _ in range(length))


# pass the reference data
def generate_synthatic_data(adata, sim_type="avg"):
    n_genes = adata.n_var
    n_cells = adata.n_obs
    n_types = len(set(adata.obs["label"].values))

    # TODO(make these arguments)
    bead_depth = 1000
    num_of_beads = n_cells * 2
    # generate proportion values
    props = np.random.dirichlet(np.ones(n_types), num_of_beads)

    true_proportion = np.zeros((num_of_beads, n_types))
    bead_to_gene_matrix = np.zeros((num_of_beads, n_genes))

    # if sim_type avg
    # generate from avg profiles
    if sim_type == "avg":
        profile_mean = obs_normalize(adata, "label")
        # run for each bead
        for bead_index in range(num_of_beads):
            allocation = np.random.multinomial(
                bead_depth, props[bead_index, :], size=1
            )[0]
            true_proportion[bead_index, :] = allocation.copy()
            for j in range(n_types):
                gene_exp = np.random.multinomial(
                    allocation[j], profile_mean.X[j, :], size=1
                )[0]

                bead_to_gene_matrix[bead_index, :] += gene_exp.copy()
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
            cell_association[:, j] = np.random.randint(
                0, len(cells_to_sample_from_celltype[j]), num_of_beads
            )

        X_norm_prof = (adata.X / adata.X.sum(1)[:, np.newaxis]).astype("float64")
        for bead_index in range(num_of_beads):
            allocation = np.random.multinomial(
                bead_depth, props[bead_index, :], size=1
            )[0]
            true_proportion[bead_index, :] = allocation.copy()
            for j in range(n_types):
                cell_index = cells_to_sample_from_celltype[j][
                    cell_association[bead_index, j]
                ]
                gene_exp = np.random.multinomial(
                    allocation[j], X_norm_prof[cell_index, :], size=1
                )[0]
                bead_to_gene_matrix[bead_index, :] += gene_exp.copy()

    bead_barcodes = np.array([generate_seq() for _ in range(num_of_beads)])

    adata_spatial = AnnData(
        bead_to_gene_matrix,
        obs=dict(obs_names=bead_barcodes),
        var=dict(var_names=adata.var_names),
    )

    true_proportion = true_proportion / true_proportion.sum(1)[:, np.newaxis].astype(
        "float64"
    )

    # fake coordinates
    adata_spatial.obsm["spatial"] = np.random.random((adata_spatial.shape[0], 2))
    adata_spatial.obsm["proportions_true"] = true_proportion

    adata_spatial.uns["sc_reference"] = adata.copy()

    return adata
