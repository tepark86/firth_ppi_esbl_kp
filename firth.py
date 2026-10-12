"""
Firth penalized logistic regression, pure Python implementation
===============================================================
Written for, and used to produce every adjusted estimate in:

    Park TE, Lee YB, Lee J, Choi H, Lee CS. Outpatient proton pump inhibitor use
    and clinical outcomes in community-onset extended-spectrum beta-lactamase-producing
    Klebsiella pneumoniae bacteremia: a retrospective cohort study.

Run validate_firth.py to reproduce the two verification checks reported in the paper.

Firth (1993) penalized log-likelihood: L*(β) = L(β) + 0.5*log|I(β)|
- Modified score: U*_j = Σ (y_i - π_i + h_i*(0.5 - π_i)) * x_ij
- IRLS with modified pseudo-response
- Wald 95% CI based on observed information at penalized MLE
- Penalized likelihood ratio test for p-value (preferred for sparse data)

Reference: Firth D. Bias reduction of maximum likelihood estimates.
           Biometrika 1993;80(1):27-38.
           Heinze G, Schemper M. A solution to the problem of separation
           in logistic regression. Stat Med 2002;21:2409-19.
"""
import numpy as np
from scipy.stats import chi2, norm
from scipy.optimize import minimize


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _firth_fit(X, y, max_iter=200, tol=1e-7):
    """Fit Firth-penalized logistic regression. Returns beta, cov, loglik*."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    beta = np.zeros(k)
    loglik_pen = -np.inf

    for it in range(max_iter):
        eta = X @ beta
        pi = _sigmoid(eta)
        W = pi * (1.0 - pi)
        # Information matrix
        XtWX = X.T @ (W[:, None] * X)
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
        # Hat diagonal
        WX = (np.sqrt(W)[:, None]) * X
        H = WX @ XtWX_inv @ WX.T
        h = np.clip(np.diag(H), 0, 1)
        # Modified score
        U = X.T @ (y - pi + h * (0.5 - pi))
        # Newton step
        try:
            delta = XtWX_inv @ U
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(XtWX, U, rcond=None)[0]
        beta_new = beta + delta
        # Penalized log-likelihood
        eta_new = X @ beta_new
        pi_new = _sigmoid(eta_new)
        ll = np.sum(y * np.log(np.clip(pi_new, 1e-300, 1)) +
                    (1 - y) * np.log(np.clip(1 - pi_new, 1e-300, 1)))
        sign, logdet = np.linalg.slogdet(X.T @ ((pi_new * (1 - pi_new))[:, None] * X))
        ll_pen = ll + 0.5 * logdet
        if abs(ll_pen - loglik_pen) < tol:
            beta = beta_new
            loglik_pen = ll_pen
            break
        beta = beta_new
        loglik_pen = ll_pen

    # Final info matrix and covariance
    pi = _sigmoid(X @ beta)
    W = pi * (1.0 - pi)
    XtWX = X.T @ (W[:, None] * X)
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(XtWX)
    return beta, cov, loglik_pen


def _pen_loglik(beta, X, y):
    """Firth penalized log-likelihood L*(b) = L(b) + 0.5*log|I(b)|. X includes intercept."""
    eta = X @ beta
    pi = _sigmoid(eta)
    ll = np.sum(y * np.log(np.clip(pi, 1e-300, 1.0))
                + (1.0 - y) * np.log(np.clip(1.0 - pi, 1e-300, 1.0)))
    W = pi * (1.0 - pi)
    _, logdet = np.linalg.slogdet(X.T @ (W[:, None] * X))
    return ll + 0.5 * logdet


def _max_pen(X, y, fix=None, val=None):
    """Maximize the penalized log-likelihood, optionally holding coefficient `fix` at `val`."""
    k = X.shape[1] - (1 if fix is not None else 0)

    def neg(b):
        bb = np.insert(np.asarray(b, float), fix, val) if fix is not None else np.asarray(b, float)
        return -_pen_loglik(bb, X, y)

    r = minimize(neg, np.zeros(k), method="Nelder-Mead",
                 options=dict(maxiter=100000, maxfev=100000, xatol=1e-10, fatol=1e-10))
    r = minimize(neg, r.x, method="Powell",
                 options=dict(maxiter=100000, maxfev=100000, xtol=1e-12, ftol=1e-12))
    return -r.fun, r.x


def firth_profile_ci(X, y, idx, alpha=0.05, bound=30.0):
    """95% profile penalized-likelihood CI for the odds ratio of column idx.

    Wald intervals from the observed information are unreliable near separation, where a
    covariate is almost perfectly predictive; Heinze and Schemper recommend the profile
    penalized likelihood instead. Returns (lo, hi) on the odds-ratio scale; a bound of
    +/- inf indicates the interval is unbounded on that side.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xd = np.hstack([np.ones((X.shape[0], 1)), X])
    j = idx + 1
    lmax, bhat = _max_pen(Xd, y)
    target = lmax - chi2.ppf(1 - alpha, 1) / 2.0
    g = lambda v: _max_pen(Xd, y, fix=j, val=v)[0] - target

    def edge(direction):
        far = bhat[j] + direction * bound
        if g(far) > 0:
            return direction * np.inf
        a, b, fa = bhat[j], far, g(bhat[j])
        for _ in range(70):
            mid = (a + b) / 2.0
            fm = g(mid)
            if (fa > 0) == (fm > 0):
                a, fa = mid, fm
            else:
                b = mid
        return (a + b) / 2.0

    return float(np.exp(edge(-1))), float(np.exp(edge(+1)))


def firth_logit(X, y, var_names=None, ci="wald"):
    """
    Fit Firth logistic regression. X may include intercept column or not;
    if not, an intercept is prepended.

    Returns: dict per variable with OR, CI, p (Wald), and overall LRT.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # Prepend intercept
    X_int = np.hstack([np.ones((X.shape[0], 1)), X])
    names = ["(Intercept)"] + (
        list(var_names) if var_names is not None
        else [f"x{i+1}" for i in range(X.shape[1])]
    )
    beta, cov, ll_full = _firth_fit(X_int, y)
    se = np.sqrt(np.diag(cov))

    out = []
    for j, nm in enumerate(names):
        z = beta[j] / se[j] if se[j] > 0 else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        or_ = float(np.exp(beta[j]))
        if ci == "profile" and j > 0:
            lo, hi = firth_profile_ci(X, y, j - 1)
        else:
            lo = float(np.exp(beta[j] - 1.96 * se[j]))
            hi = float(np.exp(beta[j] + 1.96 * se[j]))
        out.append({
            "var": nm,
            "beta": float(beta[j]),
            "se": float(se[j]),
            "OR": or_,
            "lo": lo,
            "hi": hi,
            "p_wald": float(p),
        })
    # Penalized LRT for each non-intercept var (vs reduced model dropping that var)
    # For brief report we use Wald primarily; LRT optional
    return out, ll_full


def firth_lrt_pvalue(X_full, y, drop_idx):
    """Penalized likelihood ratio test dropping column drop_idx (0-based, in X_full WITHOUT intercept).
    Returns p-value comparing full vs reduced.

    The restricted model is obtained by maximizing the penalized log-likelihood
    subject to beta[drop_idx] = 0 while the Jeffreys penalty 0.5*log|I(beta)| is
    still evaluated on the FULL design matrix, as in Heinze and Schemper and in
    the logistf implementation. Refitting the reduced model on a smaller design
    matrix would change the penalty term, so the two penalized log-likelihoods
    would no longer be on a common scale and their difference would not follow
    a chi-square distribution. This also keeps the test consistent with
    firth_profile_ci, so a 95% profile interval excludes 1 exactly when p < 0.05.
    """
    X_full = np.asarray(X_full, dtype=float)
    y = np.asarray(y, dtype=float)
    if X_full.ndim == 1:
        X_full = X_full.reshape(-1, 1)
    X_int = np.hstack([np.ones((X_full.shape[0], 1)), X_full])
    j = drop_idx + 1  # column index in X_int
    ll_full, _ = _max_pen(X_int, y)
    ll_red, _ = _max_pen(X_int, y, fix=j, val=0.0)
    stat = 2 * (ll_full - ll_red)
    if stat < 0:
        stat = 0.0
    p = 1 - chi2.cdf(stat, df=1)
    return float(p), float(stat)


if __name__ == "__main__":
    # quick sanity check on a tiny dataset
    rng = np.random.default_rng(42)
    n = 50
    x = rng.normal(size=(n, 1))
    logit = -0.5 + 1.2 * x[:, 0]
    p = _sigmoid(logit)
    y = (rng.uniform(size=n) < p).astype(float)
    res, _ = firth_logit(x, y, var_names=["x1"])
    for r in res:
        print(r)
