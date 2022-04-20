import numpy as np
import scanpy as sc
import anndata as ad


def get_pancreas_integer(adata:ad.AnnData):
    is_int = ["smartseq2"]
    is_int += ["inDrop{}".format(x) for x in range(1,5)]

    keep = np.zeros(len(adata)).astype(bool)

    for tech in is_int:
        idx = adata.obs.tech.values == tech
        keep = keep | idx

    adata = adata[keep,:]
    adata.X = adata.X
    return adata

