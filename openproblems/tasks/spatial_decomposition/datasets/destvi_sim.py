from ....tools.decorators import dataset
from ._destvi_utils import generate_synthetic_dataset_destvi


@dataset("DestVI simulation (cell)")
def destvi_sim(test=False):

    adata = generate_synthetic_dataset_destvi()
    return adata
