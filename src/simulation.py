import numpy as np
import logging
from .config import SAMPLE_SIZES, NULL_DISTRIBUTIONS, N_SIMULATIONS, QUANTILES
from .stats.ks import ks_statistic
from .stats.ad import ad_statistic
from .stats.dw import dw_statistic
from scipy.stats import norm, uniform, expon, laplace

import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def log_with_timestamp(msg):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"[{now}] {msg}")

# Map distribution names to sampling and CDF functions
DISTRIBUTIONS = {
    'normal': {
        'rvs': lambda n: norm.rvs(size=n),
        'cdf': norm.cdf
    },
    'uniform': {
        'rvs': lambda n: uniform.rvs(size=n),
        'cdf': uniform.cdf
    },
    'exponential': {
        'rvs': lambda n: expon.rvs(size=n),
        'cdf': expon.cdf
    },
    'laplace': {
        'rvs': lambda n: laplace.rvs(size=n),
        'cdf': laplace.cdf
    }
}

def run_ks_simulation():
    results = {}
    for dist_name in NULL_DISTRIBUTIONS:
        dist = DISTRIBUTIONS[dist_name]
        log_with_timestamp(f"Running KS simulation for distribution: {dist_name}")
        for n in SAMPLE_SIZES:
            log_with_timestamp(f"  Sample size: {n}")
            ks_stats = []
            import time
            start_time = time.time()
            for i in range(N_SIMULATIONS):
                sample = dist['rvs'](n)
                ks = ks_statistic(sample, dist['cdf'])
                ks_stats.append(ks)
                if (i+1) % max(1, N_SIMULATIONS // 10) == 0:
                    log_with_timestamp(f"    Progress: {int(100 * (i+1) / N_SIMULATIONS)}% ({i+1}/{N_SIMULATIONS}) simulations done")
            duration = time.time() - start_time
            log_with_timestamp(f"    Time taken for n={n}, dist={dist_name}: {duration:.2f} seconds")
            ks_stats = np.array(ks_stats)
            quantiles = np.quantile(ks_stats, QUANTILES)
            log_with_timestamp(f"    Quantiles (n={n}, dist={dist_name}): {dict(zip(QUANTILES, quantiles))}")
            results[(dist_name, n)] = quantiles
    # Write results to CSV
    import csv
    import os
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ks_quantiles.csv')
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['distribution', 'sample_size'] + [f'q_{int(q*100):02d}' for q in QUANTILES]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (dist_name, n), quantiles in results.items():
            row = {
                'distribution': dist_name,
                'sample_size': n
            }
            row.update({f'q_{int(q*100):02d}': val for q, val in zip(QUANTILES, quantiles)})
            writer.writerow(row)
    logger.info(f"KS quantile results written to {output_path}")
    return results

def run_ad_simulation():
    results = {}
    for dist_name in NULL_DISTRIBUTIONS:
        dist = DISTRIBUTIONS[dist_name]
        log_with_timestamp(f"Running AD simulation for distribution: {dist_name}")
        for n in SAMPLE_SIZES:
            log_with_timestamp(f"  Sample size: {n}")
            ad_stats = []
            import time
            start_time = time.time()
            for i in range(N_SIMULATIONS):
                sample = dist['rvs'](n)
                ad = ad_statistic(sample, dist['cdf'])
                ad_stats.append(ad)
                if (i+1) % max(1, N_SIMULATIONS // 10) == 0:
                    percent = int(100 * (i+1) / N_SIMULATIONS)
                    log_with_timestamp(f"    Progress: {percent}% ({i+1}/{N_SIMULATIONS}) simulations done")
            duration = time.time() - start_time
            log_with_timestamp(f"    Time taken for n={n}, dist={dist_name}: {duration:.2f} seconds")
            ad_stats = np.array(ad_stats)
            quantiles = np.quantile(ad_stats, QUANTILES)
            log_with_timestamp(f"    Quantiles (n={n}, dist={dist_name}): {dict(zip(QUANTILES, quantiles))}")
            results[(dist_name, n)] = quantiles
    # Write results to CSV
    import csv
    import os
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ad_quantiles.csv')
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['distribution', 'sample_size'] + [f'q_{int(q*100):02d}' for q in QUANTILES]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (dist_name, n), quantiles in results.items():
            row = {
                'distribution': dist_name,
                'sample_size': n
            }
            row.update({f'q_{int(q*100):02d}': val for q, val in zip(QUANTILES, quantiles)})
            writer.writerow(row)
    logger.info(f"AD quantile results written to {output_path}")
    return results

def run_dw_simulation():
    from .config import DW_SAMPLE_SIZES, DW_PREDICTORS
    results = {}
    for n in DW_SAMPLE_SIZES:
        for k in DW_PREDICTORS:
            log_with_timestamp(f"Running DW simulation for n={n}, predictors={k}")
            dw_stats = []
            import time
            start_time = time.time()
            for i in range(N_SIMULATIONS):
                # Generate predictors (X) and response (y)
                X = np.random.normal(size=(n, k))
                X = np.column_stack([np.ones(n), X])  # add intercept
                beta = np.random.normal(size=(k+1,))
                y = X @ beta + np.random.normal(size=n)
                # Fit regression (least squares)
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ coef
                residuals = y - y_pred
                dw = dw_statistic(residuals)
                dw_stats.append(dw)
                if (i+1) % max(1, N_SIMULATIONS // 10) == 0:
                    percent = int(100 * (i+1) / N_SIMULATIONS)
                    log_with_timestamp(f"    Progress: {percent}% ({i+1}/{N_SIMULATIONS}) simulations done")
            duration = time.time() - start_time
            log_with_timestamp(f"    Time taken for n={n}, predictors={k}: {duration:.2f} seconds")
            dw_stats = np.array(dw_stats)
            quantiles = np.quantile(dw_stats, QUANTILES)
            log_with_timestamp(f"    Quantiles (n={n}, predictors={k}): {dict(zip(QUANTILES, quantiles))}")
            results[(n, k)] = quantiles
    # Write results to CSV
    import csv
    import os
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dw_quantiles.csv')
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['sample_size', 'predictors'] + [f'q_{int(q*100):02d}' for q in QUANTILES]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (n, k), quantiles in results.items():
            row = {
                'sample_size': n,
                'predictors': k
            }
            row.update({f'q_{int(q*100):02d}': val for q, val in zip(QUANTILES, quantiles)})
            writer.writerow(row)
    logger.info(f"DW quantile results written to {output_path}")
    return results

if __name__ == "__main__":
    # Uncomment the simulation you want to run
    run_ks_simulation()
    run_ad_simulation()
    run_dw_simulation()
    
