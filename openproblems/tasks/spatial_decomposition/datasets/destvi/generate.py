from openproblems.data.utils import filter_genes_cells
from openproblems.tools.decorators import dataset


@dataset(
    "DestVI",
    data_url="https://github.com/romain-lopez/DestVI-reproducibility/"
    "blob/master/simulations/make_dataset.py",
    data_reference="https://doi.org/10.1038/s41587-022-01272-8",
    dataset_summary="Synthetic dataset from DestVI manuscript",
    image="openproblems-python-extras",
)
def destvi(test=False):
    from .utils import generate_synthetic_dataset

    merged_adata = generate_synthetic_dataset(test=test)
    filter_genes_cells(merged_adata)
    return merged_adata
