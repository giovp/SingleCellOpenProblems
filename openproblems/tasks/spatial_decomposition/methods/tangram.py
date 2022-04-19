from ....tools.decorators import method
from ....tools.utils import check_version


@method(
    method_name="Tangram",
    paper_name="Deep learning and alignment of spatially resolved single-cell transcriptomes with Tangram",  # noqa: E501
    paper_url="https://www.nature.com/articles/s41592-021-01264-7",
    paper_year=2021,
    code_url="https://github.com/broadinstitute/Tangram",
    code_version=check_version("tangram-sc"),
    image="openproblems-python-extras",
)
def tangram_simple(adata, test=False):
    # analysis based on: https://github.com/broadinstitute/Tangram/blob/master/tutorial_tangram_with_squidpy.ipynb # noqa: E501
    # using tangram from PyPi, not github version

    import pandas as pd
    import scanpy as sc
    import tangram as tg
    import torch as t

    ad_sc = adata.uns["sc_reference"].copy()

    sc.tl.rank_genes_groups(ad_sc, groupby="label", use_raw=False)

    markers_df = pd.DataFrame(ad_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
    markers = list(set(markers_df.melt().value.values))

    tg.pp_adatas(ad_sc, adata, genes=markers)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    ad_map = tg.map_cells_to_space(
        ad_sc,
        adata,
        device=device,
        num_epochs=1000,
    )

    tg.project_cell_annotations(ad_map, adata, annotation="label")

    adata.obsm["proportions_pred"] = adata.obsm["tangram_ct_pred"].copy()
    del adata.obsm["tangram_ct_pred"]

    return adata
