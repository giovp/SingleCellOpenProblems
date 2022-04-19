from typing import Dict
from typing import Sequence
from typing import Union

import anndata as ad
import numpy as np
import pandas as pd


def generate_synthetic_dataset(
    adata: ad.AnnData,
    type_column: str = "label",
    alpha: Union[float, Sequence] = 1.0,
    n_obs: int = 1000,
    cell_lb: int = 10,
    cell_ub: int = 30,
    umi_lb: int = 1000,
    umi_ub: int = 5000,
    seed: int = 42,
) -> ad.AnnData:
    """Create cell-aggregate samples for ground-truth spatial decomposition task.

    Parameters
    ----------
    adata: AnnData
        Anndata object.
    type_column: str
        name of column in `adata.obs` where cell type labels are gives
    alpha: Union[float,Sequence]
        alpha value in dirichlet distribution. If single number then all alpha_i values
        will be set to this value. Default value is 1.
    n_obs: int
        number of spatial observations to generate. Default value is 1000.
    cell_lb: int
        lower bound for number of cells at each spot. Default value is 10.
    cell_ub: int
        upper bound for number of cells at each spot. Default value is 30.
    umi_lb: int
        lower bound for number of UMIs at each spot. Default value is 10.
    umi_ub: int
        upper bound for number of UMIs at each spot. Default value is 30.
    seed: int
        Seed for rng.

    Returns
    -------
    AnnData with:
        - `adata_spatial.X`: simulated counts (aggregate of sc dataset).
        - `adata_spatial.uns["sc_reference"]`: original sc adata for reference.
        - `adata_spatial.obsm["proportions_true"]`: true proportion values.
        - `adata_spatial.obsm["n_cells"]`: number of cells from each type at every location # noqa: 501
        - `adata_spatial.obs["proportions_true"]`: total number of cells at each location # noqa: 501

    The cell type labels are stored in adata_sc.obs["label"].

    rng = np.random.default_rng(seed)

    X = adata.X
    labels = adata.obs[type_column].values
    uni_labs = np.unique(labels)
    n_labs = len(uni_labs)
    n_genes = adata.shape[1]

    label_indices = dict()

    for label in uni_labs:
        label_indices[label] = np.where(labels == label)[0]

    if not hasattr(alpha, "__len__"):
        alpha = np.ones(n_labs) * alpha

    sp_props = rng.dirichlet(alpha, size=n_obs)
    n_cells = rng.integers(cell_lb, cell_ub, size=n_obs)

    sp_x = np.zeros((n_obs, n_genes))
    sp_p = np.zeros((n_obs, n_labs))
    sp_c = np.zeros(sp_p.shape)

    for s in range(n_obs):
        n_umis = rng.integers(umi_lb, umi_ub)

        raw_s = rng.multinomial(n_cells[s], pvals=sp_props[s, :])
        sp_c[s, :] = raw_s
        prop_s = raw_s / n_cells[s]
        sp_p[s, :] = prop_s

        pool_s = np.zeros(n_genes)

        for l, n in enumerate(raw_s):
            idx_sl = rng.choice(label_indices[uni_labs[l]], size=n)
            pool_s += X[idx_sl, :].sum(axis=0)

        pool_s /= pool_s.sum()
        sp_x[s, :] = rng.multinomial(
            n_umis,
            pool_s,
        )

    obs_names = ["spatial_{}".format(x) for x in range(n_obs)]
    adata_spatial = ad.AnnData(
        sp_x,
        obs=dict(obs_names=obs_names),
        var=dict(var_names=adata.var_names),
    )

    # fake coordinates
    adata_spatial.obsm["spatial"] = rng.random((adata_spatial.shape[0], 2))
    adata_spatial.obsm["proportions_true"] = pd.DataFrame(
        sp_p,
        index=obs_names,
        columns=uni_labs,
    )
    adata_spatial.obs["n_cells"] = n_cells
    adata_spatial.obsm["n_cells"] = pd.DataFrame(
        sp_c,
        index=obs_names,
        columns=uni_labs,
    )

    adata_spatial.uns["sc_reference"] = dict(
        counts=adata.to_df(),
        label=adata.obs[type_column],
    )

    return adata_spatial


def reconstruct_sc_adata(
    sc_dict: Dict[str, Union[pd.DataFrame, pd.Series]]
) -> ad.AnnData:
    sc_adata = ad.AnnData(sc_dict["counts"])
    sc_adata.obs["label"] = sc_dict["label"]
    return sc_adata
