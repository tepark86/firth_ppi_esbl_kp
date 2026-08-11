# -*- coding: utf-8 -*-
"""Generate the synthetic example cohort shipped as example_data.csv.

The patient-level data behind the paper cannot be released (see README). This script
builds a synthetic cohort with the same column names, coding, and approximate marginal
frequencies so that validate_firth.py runs end to end for anyone who clones the
repository. The records are drawn from a fixed seed and correspond to no real patient.

The synthetic cohort deliberately contains one stratum with no events, so that the
separation behaviour that motivated Firth penalization in the first place is exercised.
"""
import numpy as np
import pandas as pd

SEED = 20260814
N = 62
N_PPI = 17          # PPI users in the real cohort
N_DEATH = 11        # in-hospital deaths in the real cohort

SOURCES = ['biliary', 'UTI', 'intraabdominal', 'pneumonia', 'catheter', 'unknown']
SOURCE_P = [0.45, 0.23, 0.13, 0.08, 0.05, 0.06]


def main():
    rng = np.random.default_rng(SEED)

    home_ppi_use = np.zeros(N, dtype=int)
    home_ppi_use[rng.choice(N, N_PPI, replace=False)] = 1

    age = np.clip(rng.normal(71, 13, N).round(), 19, 98).astype(int)
    pitt_score = rng.choice([0, 1, 2, 3, 4, 6], N, p=[.34, .24, .16, .12, .09, .05])
    immunosuppression = rng.binomial(1, 0.32, N)
    malignancy = rng.binomial(1, 0.47, N)
    empiric_appropriate = rng.binomial(1, 0.34, N)
    infection_source = rng.choice(SOURCES, N, p=SOURCE_P)

    # Deaths depend on severity and immunosuppression, never on a urinary source, which
    # reproduces the empty stratum seen in the real cohort.
    risk = -3.0 + 0.45 * pitt_score + 1.2 * immunosuppression + 0.35 * malignancy
    risk[infection_source == 'UTI'] = -50.0
    order = np.argsort(-(risk + rng.gumbel(0, 1.0, N)))
    in_hospital_mortality = np.zeros(N, dtype=int)
    in_hospital_mortality[order[:N_DEATH]] = 1

    # Clinical cure at 48 h: never among those who died, otherwise severity dependent.
    cure_p = 1.0 / (1.0 + np.exp(-(2.0 - 0.4 * pitt_score)))
    clinical_cure_48h = rng.binomial(1, cure_p)
    clinical_cure_48h[in_hospital_mortality == 1] = 0

    d = pd.DataFrame({
        'patient_id': [f'SYN-{i + 1:04d}' for i in range(N)],
        'age': age,
        'home_ppi_use': home_ppi_use,
        'pitt_score': pitt_score,
        'immunosuppression': immunosuppression,
        'malignancy': malignancy,
        'empiric_appropriate': empiric_appropriate,
        'infection_source': infection_source,
        'in_hospital_mortality': in_hospital_mortality,
        'clinical_cure_48h': clinical_cure_48h,
    })
    d.to_csv('example_data.csv', index=False, encoding='utf-8')

    print(f'wrote example_data.csv  n={len(d)}  seed={SEED}')
    print(f'  PPI users            {int(d.home_ppi_use.sum())}')
    print(f'  deaths               {int(d.in_hospital_mortality.sum())}')
    print(f'  urinary source       {int((d.infection_source == "UTI").sum())} '
          f'(deaths among them: {int(d.loc[d.infection_source == "UTI", "in_hospital_mortality"].sum())})')


if __name__ == '__main__':
    main()
