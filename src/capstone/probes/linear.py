"""The registered probe (pre-registration §3), and nothing that could tune it.

One setting for everything: every representation, layer, condition and run,
the control task and the speaker-ID probe included. There is deliberately no
argument for C, the penalty or the solver. A probe whose regularization is a
parameter is a probe whose regularization can be chosen after a result.

A fit that does not converge is reported, through the `converged` flag, and is
never refit with other settings (§3).
"""

from __future__ import annotations

import warnings


def _probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # The scaler sits inside the pipeline, so it is fit on this probe's
    # training sites only and applied to its test sites (§3).
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
                           class_weight="balanced"),
    )


def fit_predict(X_train, y_train, X_test):
    """Fit the §3 probe; return (test-set output, converged).

    Binary y (bool, True = /d/): the output is P(/d/) per test site.
    Multiclass y (the speaker-ID probe): the output is the predicted class.
    """
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning

    probe = _probe()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        probe.fit(X_train, y_train)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    classes = probe.classes_
    if len(classes) == 2:
        if not (classes[0] == False and classes[1] == True):  # noqa: E712
            raise ValueError(f"binary probe expects bool labels, got {classes}")
        return probe.predict_proba(X_test)[:, 1].astype(np.float64), converged
    return probe.predict(X_test), converged
