from ....tools.decorators import method
from ....tools.utils import check_version
from ._utils import normalize_coefficients
from ._utils import obs_means
from sklearn.svm import NuSVR

import numpy as np
import pandas as pd


@method(
    method_name="NuSVR",
    paper_name="Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods",  # noqa: E501
    paper_url="http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639",
    paper_year=1999,
    code_url="https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html",
    code_version=check_version("scikit-learn"),
)
def stereoscope_raw(adata):
    adata_sc = adata.uns["sc_reference"].copy()
    labels = adata_sc.obs["label"].cat.categories
    adata_means = obs_means(adata_sc, "label")

    X = adata_means.X.T
    y = adata.X.T.toarray()
    res = np.zeros((y.shape[1], X.shape[1]))  # (voxels,cells)
    for i in range(y.shape[1]):
        model = NuSVR(kernel="linear")
        model.fit(X, y[:, i])
        res[i] = model.coef_s

    res_prop = normalize_coefficients(res)

    adata.obsm["proportions_pred"] = pd.DataFrame(res_prop, columns=labels)

    return adata
