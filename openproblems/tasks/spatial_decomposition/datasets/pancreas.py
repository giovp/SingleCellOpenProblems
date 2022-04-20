from ....data.pancreas import load_pancreas
from ....tools.decorators import dataset
from ._pancreas_utils import get_pancreas_integer
from ._utils import generate_synthetic_dataset


@dataset("Pancreas (alpha=1)")
def pancreas_alpha_1(test=False, n_obs=1000):
    adata = load_pancreas(test=test)
    adata = get_pancreas_integer(adata)
    adata.obs["label"] = adata.obs["celltype"]

    adata_spatial = generate_synthetic_dataset(adata, n_obs=n_obs, alpha=1)
    return adata_spatial


@dataset("Pancreas (alpha=5)")
def pancreas_alpha_5(test=False, n_obs=1000):
    adata = load_pancreas(test=test)
    adata = get_pancreas_integer(adata
    adata.obs["label"]=adata.obs["celltype"]

    adata_spatial=generate_synthetic_dataset(adata, n_obs=n_obs, alpha=5)
    return adata_spatial


@ dataset("Pancreas (alpha=0.5)")
def pancreas_alpha_0_1(test=False, n_obs=1000):
    adata=load_pancreas(test=test)
    adata=get_pancreas_integer(adata
    adata.obs["label"]=adata.obs["celltype"]

    adata_spatial=generate_synthetic_dataset(adata, n_obs=n_obs, alpha=0.5)
    return adata_spatial
