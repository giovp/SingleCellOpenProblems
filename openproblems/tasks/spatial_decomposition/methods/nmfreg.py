from ....tools.decorators import method
from ....tools.normalize import log_cpm
from ....tools.utils import check_version
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler

import collections
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats


@method(
    method_name="NMF-reg",
    paper_name="Slide-seq: A scalable technology for measuring genome-wide expression at high spatial resolution",
    paper_url="https://science.sciencemag.org/content/363/6434/1463",
    paper_year=2009,
    code_url="N/A",
    code_version=check_version("NMF-reg"),
)
def nmfreg(adata):
    n_types = adata.obsm["proportions_true"].size[1]
    adata_sc = adata.uns["sc_reference"].copy()
    factors = 30
    projectiob_type = "l2"

    # Learn from reference
    X = adata_sc.X.toarray()
    X_norm = X / X.sum(1)[:, np.newaxis]
    X_scaled = StandardScaler(with_mean=False).fit_transform(X_norm)

    model = NMF(n_components=factors, init="random", random_state=17)
    Ha = model.fit_transform(X_scaled)
    Wa = model.components_

    cluster_df = adata.obs[['label']].copy()
    cluster_df.loc[:,'factor'] = np.argmax(Ha,axis=1)
    factor_to_cluster_map = np.array(
        [
            np.histogram(
                cluster_df.loc[cluster_df.factor == k, "code"],
                bins=n_types,
                range=(0, n_types),
            )[0]
            for k in range(factors)
        ]
    ).T

    factor_to_best_celltype = np.argmax(factor_to_cluster_map, axis=0)
    celltype_to_best_factor = np.argmax(factor_to_cluster_map, axis=1)

    factor_to_best_celltype_matrix = np.zeros((factors, n_types))
    for i, j in enumerate(factor_to_best_celltype):
        factor_to_best_celltype_matrix[i, j] = 1

    Ha_norm = StandardScaler(with_mean=False).fit_transform(Ha)
    if projection_type == "l2":
        sc_deconv = np.dot(Ha_norm ** 2, factor_to_best_celltype_matrix)
    else:
        sc_deconv = np.dot(Ha_norm, factor_to_best_celltype_matrix)

    sc_deconv = sc_deconv / sc_deconv.sum(1)[:, np.newaxis]

    # Evaluation on reference TODO: either move or delete
    cluster_df.loc[:, "predicted_code"] = np.argmax(sc_deconv, axis=1)
    pos_neg_dict = {
        i: [
            sc_deconv[cluster_df.predicted_code == i, i],
            sc_deconv[cluster_df.predicted_code != i, i],
        ]
        for i in range(n_types)
    }

    thresh_certainty = [0] * n_types
    for c in range(n_types):
        thresh_certainty[c] = np.max(pos_neg_dict[c][1])
    # Evaluation ends here

    # Start run on actual spatial data
    X_sp = adata.X.toarray()
    X_sp_norm = X_sp / X_sp.sum(1)[:, np.newaxis]
    X_sp_scaled = StandardScaler(with_mean=False).fit_transform(X_sp_norm)

    bead_prop_soln = np.array(
        [
            scipy.optimize.nnls(Wa.T, X_sp_scaled[b, :])[0]
            for b in range(X_sp_scaled.shape[0])
        ]
    )
    bead_prop_soln = StandardScaler(with_mean=False).fit_transform(bead_prop_soln)
    bead_prop = np.dot(bead_prop_soln, factor_to_best_celltype_matrix)

    prop = bead_prop / bead_prop.sum(1)[:, np.newaxis]
    adata.obsm["proportions_pred"] = prop

    return adata
