import anndata
import numpy as np
import pandas as pd
import random


def generate_seq(length=14):
    return "".join(random.choice("CGTA") for _ in range(length))


def grouped_obs_mean(adata, group_key, layer=None):
    if layer is not None:

        def getX(x):
            return x.layers[layer]

    else:

        def getX(x):
            return x.X

    grouped = adata.obs.groupby(group_key)
    out = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=adata.var_names,
    )

    for group, idx in grouped.indices.items():
        X = getX(adata[idx])
        out.loc[:, group] = np.ravel(X.mean(axis=0, dtype=np.float64))
    return out.T


# pass the reference data
def generate_synthatic_data(adata, sim_type="avg"):
    n_genes = adata.n_var
    n_cells = adata.n_obs
    n_types = len(set(adata.obs["label"].values))

    # TODO make these arguments
    bead_depth = 1000
    num_of_beads = n_cells * 2
    # generate proportion values
    props = np.random.dirichlet(np.ones(n_types), num_of_beads)

    true_proportion = np.zeros((num_of_beads, n_types))
    bead_to_gene_matrix = np.zeros((num_of_beads, n_genes))

    # if sim_type avg
    # generate from avg profiles
    if sim_type == "avg":
        profile_mean = grouped_obs_mean(adata, "label").values
        # run for each bead
        for bead_index in range(num_of_beads):
            allocation = np.random.multinomial(
                bead_depth, props[bead_index, :], size=1
            )[0]
            true_proportion[bead_index, :] = allocation.copy()
            for j in range(n_types):
                gene_exp = np.random.multinomial(
                    allocation[j], profile_mean[j, :], size=1
                )[0]

                bead_to_gene_matrix[bead_index, :] += gene_exp.copy()
    else:
        # generate from cells
        # assign beads to actual cells
        # cell_ids with this cluster
        cells_to_sample_from_celltype = []
        grouped = adata.obs.groupby("label")
        for _, idx in grouped.indices.items():
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

    sc_adata = anndata.AnnData(
        bead_to_gene_matrix,
        obs=dict(obs_names=bead_barcodes),
        var=dict(var_names=adata.var_names),
    )

    # fake coordinates
    sc_adata.obsm["spatial"] = np.random.random((sc_adata.shape[0], 2))
    sc_adata.obsm["proportions_true"] = true_proportion

    adata.uns["sc_reference"] = adata.copy()

    return adata
