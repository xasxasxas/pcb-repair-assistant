"""Explain one Random Forest prediction as per-feature contributions.

For each tree, follow the path the job takes from the root to its leaf. Every
split moves the tree's estimate from the parent node's value to the child's;
that change is credited to the feature the node splits on. Averaged over all
trees:

    prediction = starting point (mean root value) + sum of feature contributions

exactly (no sampling or approximation), so the parts always add up to the
number the app shows. One-hot columns are summed back into their original
feature (board type, fault category). Same idea as the "treeinterpreter"
method; it needs nothing beyond scikit-learn and ports directly to the
exported tree JSON on Android.
"""
import numpy as np

FEATURES = ["board_type", "fault_category", "age_years", "previous_repairs"]


def _node_values(tree, classifier):
    v = tree.value[:, 0, :]                      # (n_nodes, n_outputs or n_classes)
    if classifier:
        v = v / v.sum(1, keepdims=True)          # class fractions per node
        return v[:, 1]                           # probability of the positive class (BER)
    return v[:, 0]


def _source_feature(name):
    """'cat__board_type_RF-COMM' -> 'board_type'; 'num__age_years' -> 'age_years'."""
    body = name.split("__", 1)[1]
    return next(f for f in FEATURES if body == f or body.startswith(f + "_"))


def contributions(pipe, X):
    """pipe: Pipeline(prep, RandomForest*). X: one-row DataFrame.
    Returns (start, {feature: contribution}, prediction)."""
    prep, forest = pipe.named_steps["prep"], pipe.named_steps["model"]
    classifier = hasattr(forest, "predict_proba")
    row = prep.transform(X)
    row = row.toarray() if hasattr(row, "toarray") else np.asarray(row)
    owner = [_source_feature(n) for n in prep.get_feature_names_out()]
    contrib = dict.fromkeys(FEATURES, 0.0)
    start = 0.0
    for est in forest.estimators_:
        t = est.tree_
        val = _node_values(t, classifier)
        node = 0
        start += val[0]
        while t.children_left[node] != -1:               # until a leaf
            f = t.feature[node]
            child = (t.children_left[node] if row[0, f] <= t.threshold[node]
                     else t.children_right[node])
            contrib[owner[f]] += val[child] - val[node]
            node = child
    n = len(forest.estimators_)
    start /= n
    contrib = {k: v / n for k, v in contrib.items()}
    return start, contrib, start + sum(contrib.values())
