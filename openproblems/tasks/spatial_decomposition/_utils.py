import anndata as ad
import numpy as np

# from typing import TypedDict, Tuple
from typing_extensions import TypedDict
from typing import Tuple


class SpScBC(TypedDict):
    spatial: str
    single_cell: str


def merge_sc_and_sp(
    adata_sc: ad.AnnData,
    adata_sp: ad.AnnData,
    batch_key: str = "modality",
    batch_categories: SpScBC = dict(spatial="sp", single_cell="sc"),
) -> ad.AnnData:

    # merge single cell and spatial data
    adata_merged = adata_sp.concatenate(
        adata_sc,
        batch_key=batch_key,
        batch_categories=(batch_categories["spatial"], batch_categories["single_cell"]),
        join="outer",
        index_unique=None,
    )

    # get attributes from single cell and spatial data
    sc_obs = set(adata_sc.obs.columns.tolist())
    sp_obs = set(adata_sp.obs.columns.tolist())
    sc_obsm = set(list(adata_sc.obsm.keys()))
    sp_obsm = set(list(adata_sp.obsm.keys()))

    # store unique attributes for cleanup in split
    attr_dict = dict(
        sc_obs_only=list(sc_obs.difference(sp_obs)),
        sp_obs_only=list(sp_obs.difference(sc_obs)),
        sc_obsm_only=list(sc_obsm.difference(sp_obsm)),
        sp_obsm_only=list(sp_obsm.difference(sc_obsm)),
    )

    adata_merged.uns["attr_dict"] = attr_dict

    return adata_merged


def split_sc_and_sp(
    adata_merged: ad.AnnData,
    batch_key: str = "modality",
    batch_categories: SpScBC = dict(spatial="sp", single_cell="sc"),
) -> Tuple[ad.AnnData, ad.AnnData]:

    # split single cell and spatial data
    is_sp = adata_merged.obs[batch_key] == batch_categories["spatial"]
    adata_sp = adata_merged[is_sp, :]
    adata_sc = adata_merged[~is_sp, :]

    # clean objects
    attr_dict = adata_merged.uns["attr_dict"]
    for col in attr_dict["sc_obs_only"]:
        del adata_sp.obs[col]
    for col in attr_dict["sc_obsm_only"]:
        del adata_sp.obsm[col]
    for col in attr_dict["sp_obs_only"]:
        del adata_sc.obs[col]
    for col in attr_dict["sp_obsm_only"]:
        del adata_sc.obsm[col]

    return (adata_sc, adata_sp)



def obs_means(adata: ad.AnnData, cluster_key: str) -> ad.AnnData:
    """Return means over observation key."""

    labels = adata.obs[cluster_key].cat.categories
    means = np.empty((labels.shape[0], adata.shape[1]))
    for i, lab in enumerate(labels):
        means[i, :] = adata[adata.obs[cluster_key] == lab].X.mean(axis=0).flatten()
    adata_means = AnnData(means)
    adata_means.obs_names = labels
    adata_means.var_names = adata.var_names

    return adata_means


def normalize_coefficients(_prop: np.array) -> np.array:
    """Normalize coefficients to sum to 1."""
    prop = _prop.copy()
    prop[prop < 0] = 0
    prop = prop / prop.sum(axis=1, keepdims=1)
    return prop
