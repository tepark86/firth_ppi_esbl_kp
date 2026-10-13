# -*- coding: utf-8 -*-
"""Validation of the Firth penalized logistic regression implementation in firth.py.

Firth regression produced every adjusted estimate in the manuscript, and the
implementation was written for this study rather than taken from an existing package,
so it is checked two independent ways.

Check 1, closed-form equivalence.
    For a saturated model with a single binary covariate, Firth penalization is exactly
    equivalent to adding one half to each cell of the corresponding 2 x 2 table, so the
    fitted odds ratio must equal (a+.5)(d+.5)/((b+.5)(c+.5)). This check runs on the
    published 2 x 2 table of outpatient PPI use against in-hospital mortality, which is
    an aggregate reported in Table 1 of the paper, and therefore runs without any
    patient-level data.

Check 2, independent optimizer.
    The same penalized log-likelihood, L*(b) = L(b) + 0.5*log|I(b)|, is maximized with a
    general-purpose derivative-free optimizer (Nelder-Mead refined by Powell) instead of
    the modified-score IRLS used in firth.py, and every coefficient must agree. Because
    this check needs patient-level covariates, which cannot be released, it runs here on
    the synthetic cohort in example_data.csv. It therefore demonstrates that the two
    estimators agree. On the real cohort, all odds ratios reported in the paper were
    identical after rounding. Point --data at the real file to repeat the check there.

Expected output on the shipped example data:
    check 1  theoretical 1.7111111  vs  firth.py 1.7111120   (difference 8.9e-07)
    check 2  worst relative difference across all coefficients below 1e-4
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from firth import firth_logit

TOL_IRLS = 1e-7        # convergence tolerance inside firth.py
TOL_REL = 1e-4         # acceptance threshold for the independent refit

# Outpatient PPI use against in-hospital mortality, n = 62 (Table 1 of the paper).
#            died  survived
# PPI user      4        13
# non-user      7        38
TABLE_2X2 = ((4, 13), (7, 38))


def check_closed_form():
    (a, b), (c, e) = TABLE_2X2
    n = a + b + c + e
    theoretical = ((a + .5) * (e + .5)) / ((b + .5) * (c + .5))

    # Rebuild the individual records implied by the 2 x 2 table. A saturated model with a
    # single binary covariate depends on the data only through these four cell counts.
    x = np.array([1] * (a + b) + [0] * (c + e), dtype=float)
    y = np.array([1] * a + [0] * b + [1] * c + [0] * e, dtype=float)

    res, _ = firth_logit(x[:, None], y, var_names=['home_ppi_use'])
    fitted = res[1]['OR']
    diff = abs(theoretical - fitted)

    print('check 1 - Firth on a saturated 2 x 2 model equals the add-one-half odds ratio')
    print(f'  table                  [[{a}, {b}], [{c}, {e}]]  (n = {n})')
    print(f'  theoretical (add 0.5)  {theoretical:.10f}')
    print(f'  firth.py               {fitted:.10f}')
    print(f'  difference             {diff:.2e}   (IRLS tolerance {TOL_IRLS:.0e})')
    ok = diff < 10 * TOL_IRLS
    print(f'  verdict                {"PASS" if ok else "FAIL"}')
    return ok


def neg_penalized_ll(beta, X, y):
    eta = X @ beta
    pi = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
    ll = np.sum(y * np.log(np.clip(pi, 1e-300, 1.0))
                + (1 - y) * np.log(np.clip(1 - pi, 1e-300, 1.0)))
    W = pi * (1.0 - pi)
    _, logdet = np.linalg.slogdet(X.T @ (W[:, None] * X))
    return -(ll + 0.5 * logdet)


def independent_fit(X, y):
    Xd = np.column_stack([np.ones(len(y)), X])
    opt = dict(maxiter=200000, maxfev=200000)
    r = minimize(neg_penalized_ll, np.zeros(Xd.shape[1]), args=(Xd, y),
                 method='Nelder-Mead', options={**opt, 'xatol': 1e-12, 'fatol': 1e-12})
    r = minimize(neg_penalized_ll, r.x, args=(Xd, y),
                 method='Powell', options={**opt, 'xtol': 1e-13, 'ftol': 1e-13})
    return np.exp(r.x[1:])


MODELS = [
    ('mortality ~ PPI', ['home_ppi_use'], 'in_hospital_mortality'),
    ('mortality ~ PPI + age', ['home_ppi_use', 'age'], 'in_hospital_mortality'),
    ('mortality ~ PPI + Pitt', ['home_ppi_use', 'pitt_score'], 'in_hospital_mortality'),
    ('mortality ~ PPI + immuno', ['home_ppi_use', 'immuno'], 'in_hospital_mortality'),
    ('mortality ~ PPI + malignancy', ['home_ppi_use', 'malig'], 'in_hospital_mortality'),
    ('mortality ~ PPI + approp', ['home_ppi_use', 'emp_ap'], 'in_hospital_mortality'),
    ('mortality ~ PPI + liver disease', ['home_ppi_use', 'cld'], 'in_hospital_mortality'),
    ('mortality ~ PPI + immuno + UTI',
     ['home_ppi_use', 'immuno', 'src_uti'], 'in_hospital_mortality'),
    ('failure ~ PPI + immuno + UTI', ['home_ppi_use', 'immuno', 'src_uti'], 'fail'),
]


def check_independent_optimizer(path):
    d = pd.read_csv(path)
    d['immuno'] = (d['immunosuppression'] == 1).astype(int)
    d['src_uti'] = (d['infection_source'] == 'UTI').astype(int)
    d['malig'] = (d['malignancy'] == 1).astype(int)
    d['emp_ap'] = (d['empiric_appropriate'] == 1).astype(int)
    d['cld'] = (d['chronic_liver_disease'] == 1).astype(int)
    d['fail'] = ((d['in_hospital_mortality'] == 1) | (d['clinical_cure_48h'] == 0)).astype(int)

    print(f'\ncheck 2 - independent maximization of the same penalized log-likelihood')
    print(f'  data: {path}  (n = {len(d)})')
    print(f'  {"model":32s} {"firth.py":>12s} {"independent":>12s} {"max rel diff":>13s}')
    worst = 0.0
    for label, cols, outcome in MODELS:
        sub = d[[outcome] + cols].dropna()
        yy = sub[outcome].values.astype(float)
        XX = sub[cols].values.astype(float)
        r, _ = firth_logit(XX, yy, var_names=cols)
        mine = np.array([r[i + 1]['OR'] for i in range(len(cols))])
        theirs = independent_fit(XX, yy)
        rel = float(np.max(np.abs(mine - theirs) / mine))
        worst = max(worst, rel)
        print(f'  {label:32s} {mine[0]:12.6f} {theirs[0]:12.6f} {rel:13.2e}')
    ok = worst < TOL_REL
    print(f'\n  worst relative difference across all coefficients: {worst:.2e}')
    print(f'  verdict                {"PASS" if ok else "FAIL"}')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default='example_data.csv',
                    help='patient-level file for check 2 (default: the synthetic example cohort)')
    args = ap.parse_args()

    ok1 = check_closed_form()
    ok2 = check_independent_optimizer(args.data)
    print('\nOVERALL: ' + ('PASS - implementation reproduces both references'
                           if (ok1 and ok2) else 'FAIL - investigate before reporting'))
    raise SystemExit(0 if (ok1 and ok2) else 1)


if __name__ == '__main__':
    main()
