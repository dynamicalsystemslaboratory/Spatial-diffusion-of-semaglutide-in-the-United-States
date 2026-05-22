
import numpy as np, pandas as pd, statsmodels.api as sm
import libpysal
from libpysal.weights import KNN, Rook
import esda


import os
import seaborn as sns
import pandas as pd
import pysal as ps
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

import statsmodels.api as statsmodel
from scipy.stats import linregress

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.sparse.csgraph import connected_components
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# df_ozempic_2018_COUNTY = pd.read_csv("./Data/df_ozempic_2018_COUNTY.csv")
# df_ozempic_2019_COUNTY = pd.read_csv("./Data/df_ozempic_2019_COUNTY.csv")
# df_ozempic_2020_COUNTY = pd.read_csv("./Data/df_ozempic_2020_COUNTY.csv")
# df_ozempic_2021_COUNTY = pd.read_csv("./Data/df_ozempic_2021_COUNTY.csv")
# df_ozempic_2022_COUNTY = pd.read_csv("./Data/df_ozempic_2022_COUNTY.csv")
# df_ozempic_2023_COUNTY = pd.read_csv("./Data/df_ozempic_2023_COUNTY.csv")
# county_health = pd.read_csv("./Data/county_health.csv")

# countyshape = gpd.read_file("./Data/cb_2018_us_county_500k/cb_2018_us_county_500k.shp")
# countyshape['GEOID'] = countyshape.GEOID.astype(int)
# countyshape['FIPS'] = countyshape['GEOID']

# county_shape = gpd.read_file('./Data/cb_2018_us_county_500k/cb_2018_us_county_500k.shp')
# county_shape['FIPS'] = county_shape.GEOID.astype('int')
# # cbsa_shape = gpd.read_file('./DATA/tl_2020_us_cbsa/tl_2020_us_cbsa.shp')
# # cbsa_shape['cbsacode'] = cbsa_shape['GEOID'].astype('int')
# state_shape = gpd.read_file('./Data/tl_2020_us_state/tl_2020_us_state.shp')
# county_shape = county_shape.merge(state_shape[['STATEFP','STUSPS']], on='STATEFP').copy()


def scaling_iloc(x0,y0):
    # print(len(x0))
    samis = Get_SAMIs(x0,y0)
    # print("SAMIs = ",samis)
    # print("Var = ",np.round(np.var(samis),2))
    x = np.log(x0)
    Y = np.log(y0)
    X = statsmodel.add_constant(x)
    model = statsmodel.OLS(Y,X)
    fit = model.fit(cov_type='HC1')
    # print(fit.summary())
        
    intercept, slope = fit.params
    # print("intercept = ", np.round(intercept,2))
    # print("c = ", np.round(np.exp(intercept),3))

    x_0 =  np.sort(x0)[0]
    y_0 = np.exp(slope*np.log(x_0)+intercept)

    x_f = np.sort(x0)[-1]
    y_f = np.exp(slope*np.log(x_f)+intercept)
    y_null = np.exp(np.log(x_f)+intercept)

    beta = round(slope,4)
    R2 = str(round(fit.rsquared,2))
    # print(R2)
    # R2 = str(round(fit.pvalues[1],3))
    beta_lowerbound, beta_upper = fit.conf_int().iloc[1]
    beta_lowerbound = np.round(beta_lowerbound,3)
    beta_upper = np.round(beta_upper,3)

    x = np.log(x0)
    Y = np.log(y0/x0)
    X = statsmodel.add_constant(x)
    model = statsmodel.OLS(Y,X)
    fit2 = model.fit(cov_type='HC1')
    intercept2, slope2 = fit2.params
    print("intercept = ", np.round(intercept2,2))
    print("c = ", np.round(np.exp(intercept2),3))
    x_0 =  np.sort(x0)[0]
    y_0 = np.exp(slope2*np.log(x_0)+intercept2)
    x_f = np.sort(x0)[-1]
    y_f = np.exp(slope2*np.log(x_f)+intercept2)
    y_null = np.exp(np.log(x_f)+intercept2)
    print(r'$\beta = {}$'.format("{:.3f}".format(beta))+r'$ \, \in \,[{}$'.format(beta_lowerbound) + r'$,{}]$'.format(beta_upper)+r', $\mathit{R}^2 = $' +r'${}$'.format(R2),)
    return fit.conf_int().iloc[1,:],samis,beta,float(R2)

def Get_SAMIs(x0,y0):
    x = np.log(x0)
    y = np.log(y0)
    res = linregress(x,y)
    y_hat = res.intercept + res.slope*x
    SAMIs = y-y_hat
    return SAMIs

def all_upper(my_list): 
    return list(map(lambda x: x.upper(), my_list))

import numpy as np
import geopandas as gpd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt

def align_to_W(gdf, W, id_col, x_col):
    """
    Reorder gdf to match W.id_order and return:
      idx_map: dict id -> row index in reindexed gdf {area_id: index}
      x:       np.array aligned to W.id_order (SAMIs) 
      A:       csr_matrix adjacency (binary, symmetric)
      gdf2:    reindexed GeoDataFrame (same order as W.id_order)
    """
    order = list(W.id_order)
    gdf2  = gdf.set_index(id_col).reindex(order)
    if gdf2[x_col].isna().any():
        missing = gdf2[gdf2[x_col].isna()].index.tolist()
        raise ValueError(f"{len(missing)} IDs in W not found in gdf: {missing[:5]} ...")
    x = gdf2[x_col].to_numpy(np.float64) # residuals of interest
    A = W.sparse.tocsr()
    # ensure binary symmetric
    A.data[:] = 1
    A = ((A + A.T) > 0).astype(np.int8).tocsr()
    idx_map = {k:i for i,k in enumerate(order)} # 
    return idx_map, x, A, gdf2

def percolation_sweep(A, x, step=1, descending = True):
    """
    Rank-occupy percolation on an arbitrary graph.
    Returns dict with arrays over k (or p=k/N):
      'p', 'S1', 'Nc', 'labels_at_k' (optional heavy; here omitted for speed)
    """
    n = len(x)
    if descending is True:
        order = np.argsort(-x) 
    # order = np.argsort(-x)  # descending
    else:
        order = np.argsort(x)
    occupied = np.zeros(n, dtype=bool)

    ks = np.arange(step, n+1, step, dtype=int)
    S1 = np.zeros_like(ks, dtype=int)
    Nc = np.zeros_like(ks, dtype=int)

    for j, k in enumerate(ks):
        occupied[ order[k-step:k] ] = True  # add the new block
        mask = occupied
        if mask.sum() == 0:
            S1[j] = 0; Nc[j] = 0
            continue
        A_k = A[mask][:, mask] # adjacency matrix of occupied areas
        nc, labels = connected_components(A_k, directed=False, return_labels=True) # labels identify which component(cluster) the node belongs to
        # sizes
        counts = np.bincount(labels) # count of occurrences of each component  
        S1[j] = counts.max() # size of largest component
        Nc[j] = nc # number of connected components

    p = ks / n # occupation fraction or percolation probability
    return {"p": p, "S1": S1, "Nc": Nc, "order": order, "step": step}

def first_argmax(a):
    i = int(np.argmax(a))
    return i

def percolation_pc(sweep):
    i = first_argmax(sweep["Nc"]) # index where Nc is maximal (when number of components or fragmentation is maximal)
    return {"pc": float(sweep["p"][i]),  # ~approx percolation threshold
            "k_c": int(np.round(sweep["p"][i] * (1.0 / sweep["p"][-1]) * (len(sweep["p"])*sweep["step"]))), 
            "idx": i}

from shapely.ops import unary_union

def shuffled_sweep(A, x, step=1, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    x_shuf = x.copy()
    rng.shuffle(x_shuf)
    return percolation_sweep(A, x_shuf, step=step)


import numpy as np
import matplotlib.pyplot as plt

def null_envelope(A, x, step=1, n_runs=200, seed=0):
    """
    Shuffle x n_runs times, run percolation_sweep each time,
    and return percentiles across runs for S1 and Nc.
    """
    rng = np.random.default_rng(seed)
    # run once to fix the p-grid
    base = percolation_sweep(A, x, step=step)
    p = base["p"]
    m = len(p)

    S1_mat = np.empty((n_runs, m), dtype=float)
    Nc_mat = np.empty((n_runs, m), dtype=float)

    for r in range(n_runs):
        s = shuffled_sweep(A, x, step=step, rng=rng)
        # p is identical across runs as long as n and step don't change
        S1_mat[r, :] = s["S1"]
        Nc_mat[r, :] = s["Nc"]
    # print("chek")
    # percentiles and median
    S1_lo, S1_hi = np.percentile(S1_mat, [5, 95], axis=0)
    S1_med       = np.percentile(S1_mat, 50, axis=0)
    Nc_lo, Nc_hi = np.percentile(Nc_mat, [5, 95], axis=0)
    Nc_med       = np.percentile(Nc_mat, 50, axis=0)
    # print(S1_lo, S1_hi)
    return {
        "p": p,
        "S1_lo": S1_lo, "S1_hi": S1_hi, "S1_med": S1_med,
        "Nc_lo": Nc_lo, "Nc_hi": Nc_hi, "Nc_med": Nc_med
    }


def compute_pc(real, x):
    idx_pc = np.argmax(real["Nc"])
    p_c = real["p"][idx_pc]
    k_c = int(real["p"][idx_pc] * len(x))
    return idx_pc, p_c, k_c



