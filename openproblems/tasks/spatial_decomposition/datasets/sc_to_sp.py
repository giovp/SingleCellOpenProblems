from ....data.pancreas import load_pancreas
from ....tools.decorators import dataset
from .._utils import merge_sc_and_sp
from ._sc_to_sp_utils import generate_synthetic_dataset
from ._sc_to_sp_utils import get_pancreas_integer


@dataset("Pancreas (alpha=1)")
def pancreas_alpha_1(test=False, n_obs=1000):
    adata = load_pancreas(test=test)
    adata = get_pancreas_integer(adata)
    adata.obs["label"] = adata.obs["celltype"]

    adata_spatial = generate_synthetic_dataset(adata, n_obs=n_obs, alpha=1)
    merged_anndata = merge_sc_and_sp(adata, adata_spatial)
    return merged_anndata


@dataset("Pancreas (alpha=5)")
def pancreas_alpha_5(test=False, n_obs=1000):
    adata = load_pancreas(test=test)
    adata = get_pancreas_integer(adata)
    adata.obs["label"] = adata.obs["celltype"]

    adata_spatial = generate_synthetic_dataset(adata, n_obs=n_obs, alpha=5)
    merged_anndata = merge_sc_and_sp(adata, adata_spatial)
    return merged_anndata


@dataset("Pancreas (alpha=0.5)")
def pancreas_alpha_0_1(test=False, n_obs=1000):
    adata = load_pancreas(test=test)
    adata = get_pancreas_integer(adata)
    adata.obs["label"] = adata.obs["celltype"]

    adata_spatial = generate_synthetic_dataset(adata, n_obs=n_obs, alpha=0.5)
    merged_anndata = merge_sc_and_sp(adata, adata_spatial)
    return merged_anndata
