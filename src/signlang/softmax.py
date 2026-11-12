"""The softmax classifier (multinomial logistic regression) used in every stage."""

import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .metrics import accuracy, macro_f1
from .timing import Timer


def fit_softmax(X, y, C, max_iter=2000, random_state=42):
    """Fit an L2-penalised softmax classifier with lbfgs.

    Returns (model, info) where info has the fit time in seconds, the number of
    lbfgs iterations and whether it converged (no ConvergenceWarning).
    """
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter,
                               random_state=random_state)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        with Timer() as t:
            model.fit(X, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    info = {"seconds": t.seconds, "n_iter": int(model.n_iter_.max()),
            "converged": converged}
    return model, info


def sweep_C(X_train, y_train, X_val, y_val, Cs, **fit_kwargs):
    """Fit one model per C; DataFrame with validation accuracy, macro-F1,
    fit time, iterations and convergence for each C."""
    rows = []
    for C in Cs:
        model, info = fit_softmax(X_train, y_train, C, **fit_kwargs)
        pred = model.predict(X_val)
        rows.append({"C": C, "val_acc": accuracy(y_val, pred),
                     "val_f1": macro_f1(y_val, pred), **info})
    return pd.DataFrame(rows)
