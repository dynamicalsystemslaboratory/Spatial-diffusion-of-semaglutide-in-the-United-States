import pandas as pd
import geopandas as gpd
import numpy as np

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import os
import seaborn as sns
import pandas as pd
import pysal as ps
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.sparse.csgraph import connected_components
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from percolation import scaling_iloc
from percolation import percolation_sweep, percolation_pc, align_to_W, compute_pc, null_envelope
import libpysal
from libpysal.weights import KNN, Rook
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from patsy import dmatrix
import libpysal
from libpysal.weights import KNN, Rook
import esda

def ozempic_health_year(df_ozempic_2022_COUNTY, year):
    county_ozempic_health_ffr = df_ozempic_2022_COUNTY.merge(county_health, left_on='COUNTY',right_on='FIPS')
    # county_ozempic_health_ffr = county_ozempic_health_ffr[(county_ozempic_health_ffr['Population_2020']!=0)&(county_ozempic_health_ffr['Suicide']!=0)]
    county_ozempic_health_ffr = countyshape.merge(county_ozempic_health_ffr, on='FIPS')
    county_ozempic_health_ffr = county_ozempic_health_ffr[~county_ozempic_health_ffr['STATEFP'].isin(['02','15'])].copy()
    relig_nm = "Conservative Protestant"
    county_ozempic_health_ffr = county_ozempic_health_ffr[county_ozempic_health_ffr[f'{relig_nm}']!=0]
    _,county_ozempic_health_ffr[f'SAMIs_Claims_{year}'], _,_ =  scaling_iloc(county_ozempic_health_ffr[f'Population_2022'],county_ozempic_health_ffr['Tot_Clms'])
    county_ozempic_health_ffr[f'Claims_PC_{year}'] = county_ozempic_health_ffr['Tot_Clms']/county_ozempic_health_ffr[f'Population_2022']
    return county_ozempic_health_ffr

def mean_abs_smd(Z, T, w=None):
    Z = np.asarray(Z, float); t = (T==1); c = ~t
    if w is None:
        m1, m0 = Z[t].mean(0), Z[c].mean(0)
        v1, v0 = Z[t].var(0, ddof=1), Z[c].var(0, ddof=1)
    else:
        w1, w0 = w[t], w[c]
        m1 = np.average(Z[t], 0, weights=w1); m0 = np.average(Z[c], 0, weights=w0)
        def wvar(x, w): mu = np.average(x, weights=w); return np.average((x-mu)**2, weights=w)
        v1 = np.array([wvar(Z[t][:,j], w1) for j in range(Z.shape[1])])
        v0 = np.array([wvar(Z[c][:,j], w0) for j in range(Z.shape[1])])
    sd = np.sqrt((v1+v0)/2.0)
    return float(np.mean(np.abs((m1-m0)/np.where(sd==0, 1.0, sd))))


# --- Data Reader ---

df_ozempic_2018_COUNTY = pd.read_csv("./Data/df_ozempic_2018_COUNTY.csv")
df_ozempic_2019_COUNTY = pd.read_csv("./Data/df_ozempic_2019_COUNTY.csv")
df_ozempic_2020_COUNTY = pd.read_csv("./Data/df_ozempic_2020_COUNTY.csv")
df_ozempic_2021_COUNTY = pd.read_csv("./Data/df_ozempic_2021_COUNTY.csv")
df_ozempic_2022_COUNTY = pd.read_csv("./Data/df_ozempic_2022_COUNTY.csv")
df_ozempic_2023_COUNTY = pd.read_csv("./Data/df_ozempic_2023_COUNTY.csv")

county_health = pd.read_csv("./Data/county_health.csv")

countyshape = gpd.read_file("./Data/cb_2018_us_county_500k/cb_2018_us_county_500k.shp")
countyshape['GEOID'] = countyshape.GEOID.astype(int)
countyshape['FIPS'] = countyshape['GEOID']

county_shape = gpd.read_file('./Data/cb_2018_us_county_500k/cb_2018_us_county_500k.shp')
county_shape['FIPS'] = county_shape.GEOID.astype('int')
# cbsa_shape = gpd.read_file('./DATA/tl_2020_us_cbsa/tl_2020_us_cbsa.shp')
# cbsa_shape['cbsacode'] = cbsa_shape['GEOID'].astype('int')
state_shape = gpd.read_file('./Data/tl_2020_us_state/tl_2020_us_state.shp')
county_shape = county_shape.merge(state_shape[['STATEFP','STUSPS']], on='STATEFP').copy()
# state_shape['STUSPS'] = [s[-2:] for s in state_shape['NAME']]
# state_shape['STUSPS'] = state_shape['STATEFP']
# state_shape.head()


# --- Configuration & Setup ---

Y_col = f"SAMIs_Depression"
X_cols = [
    # "SAMIs_MHP", 
          f"SAMIs_Unemployment", 
          "SAMIs_REPUBLICAN_2024", 
          f"SAMIs_Conservative Protestant",
          f"SAMIs_Obesity",
          f"SAMIs_Diabetes"
          ]

k_values = range(5, 26)
subgroups = ['HH', 'HL', 'LH', 'LL']
k_values = range(5, 25 + 1)


years = [2018,2019,2020,2021,2022,2023]
def classify_bivariate(row):
    # High Diabetes / High Obesity
    if row[f'SAMIs_Diabetes'] > db_median and row[f'SAMIs_Obesity'] > ob_median :
        return 'HH'
    
    # High Diabetes / Low Obesity (The Outliers)
    elif row[f'SAMIs_Diabetes'] > db_median and row[f'SAMIs_Obesity'] <= ob_median :
        return 'HL'
    
    # Low Diabetes / High Obesity
    elif row[f'SAMIs_Diabetes'] <= db_median and row[f'SAMIs_Obesity'] > ob_median :
        return 'LH'
    
    # Low Diabetes / Low Obesity
    else:
        return 'LL'

effect = 'ATT'
# fig, axes = plt.subplots(2, 3, figsize=(20, 10))
all_gdfs = []

cmap = sns.color_palette("Set2", n_colors=len(years))
palette = {j: cmap[j] for j in range(len(years))}
plt.figure(figsize=(14, 8))
plt.tick_params(bottom=True, left=True, direction='out')

n_years = len(years)
jitter_width = 0.6  # Total horizontal spread
offsets = np.linspace(-jitter_width/2, jitter_width/2, n_years)
for i, year in enumerate(years):
    # ax = axes[i]
    df_name = f"df_ozempic_{year}_COUNTY"
    
    # if df_name not in locals():
    #     ax.axis('off')
    #     continue
    
    # 1. Data Preparation & Merging
    df_current = locals()[df_name]
    gdf_year = ozempic_health_year(df_current, year)#.merge(county_health, left_on='COUNTY', right_on='FIPS').copy()
    STATE_FE = "STATEFP" if "STATEFP" in gdf_year.columns else None
    # Standardize variables (Use local copies to avoid SettingWithCopyWarning)
    df_selected = gdf_year.copy()
    
    # Helper for repetitive scaling logic
    cols_to_fix = [
        (f'Population_2022', 'REPUBLICAN_2024', 'SAMIs_REPUBLICAN_2024'),
        # ('Population_2022', 'MHP', 'SAMIs_MHP'),
        (f'Population_2022', f'Unemployment', f'SAMIs_Unemployment'),
        (f'Population_2022', f'Depression', f'SAMIs_Depression'),
        (f'Population_2022', f'Obesity', f'SAMIs_Obesity'),
        (f'Population_2022', f'Diabetes', f'SAMIs_Diabetes'),
        (f'Population_2022', f'Conservative Protestant', f'SAMIs_Conservative Protestant')
    ]
    
    for pop, raw, sami in cols_to_fix:
        # Fill zeros to avoid log/division errors if scaling_iloc requires it
        df_selected[raw] = df_selected[raw].replace(0, 1e-6)
        df_selected[raw] = df_selected[raw].replace(np.nan, 1e-6)
        _, df_selected[sami], _, _ = scaling_iloc(df_selected[pop], df_selected[raw],)

    # 2. Percolation Adjacency (Consistent Adjacency is better for time-series)
    # Using Rook as it's the standard for geographical 'physical' contact
    if year == 2018:
        W_county = KNN.from_dataframe(df_selected, k=5, ids=df_selected["COUNTY"].tolist())
    else:
        W_county = Rook.from_dataframe(df_selected, ids=df_selected["COUNTY"].tolist())
    idx_map, x, A, gdf_final = align_to_W(df_selected, W_county, id_col="COUNTY", x_col=f"SAMIs_Claims_{year}")

    # 3. Percolation Sweep
    real = percolation_sweep(A, x, step=1, descending=True)
    idx_pc, p_c, k_c = compute_pc(real, x) # Identify the threshold
    occ_mask = np.zeros(len(x), dtype=bool)
    occ_mask[real["order"][:k_c]] = True
    A_pc = A[occ_mask][:, occ_mask]
    n_comp, labels_pc = connected_components(A_pc, directed=False, return_labels=True)
    labels = np.full(len(x), -1)
    labels[occ_mask] = labels_pc
    gdf_final['cluster_pc'] = labels
    comp_all = gdf_final.loc[gdf_final["cluster_pc"] >= 0, "cluster_pc"]
    all_sizes = comp_all.value_counts().sort_values(ascending=False)
    
    
    # Get cluster IDs at the critical point
    # occ, labels, sizes = clusters_at_pc(A, x, real['order'], idx_pc, step=1)
    # all_sizes = pd.Series(sizes).sort_values(ascending=False)
    
    # 4. Sensitivity Analysis (Looping through k clusters)
    sensitivity_results = []

    ob_median = gdf_final[f'SAMIs_Obesity'].median()
    db_median = gdf_final[f'SAMIs_Diabetes'].median()
    
    # Prepare Propensity Score inputs once per year
    
    all_gdfs.append(gdf_final)
    for k in k_values:
        # Define Treatment: County is in the top-k largest clusters
        topK_ids = all_sizes.index[:k].to_numpy()

        gdf_final["T"] = gdf_final["cluster_pc"].isin(topK_ids).astype(int)
        
        T = gdf_final["T"].to_numpy(int)
        Y = pd.to_numeric(gdf_final[Y_col], errors="coerce").to_numpy(float)

        FE = pd.get_dummies(gdf_final[STATE_FE], drop_first=True, dtype=float) if STATE_FE else pd.DataFrame(index=gdf_final.index)
    
        # Geographic Splines (Controls for spatial auto-correlation)
        gdf_proj = gdf_final.to_crs(5070)
        cent = gdf_proj.geometry.centroid
        x_m = cent.x.to_numpy(float); y_m = cent.y.to_numpy(float)
        Xyspl = dmatrix("bs(x, df=4, include_intercept=False) + bs(y, df=4, include_intercept=False)",
                {"x": x_m, "y": y_m}, return_type="dataframe")
        
        # X_ps = pd.concat([gdf_final[X_cols], FE], axis=1)
        X_confounders = gdf_final[X_cols]
        X_ps = X_confounders.reset_index(drop=True)
        # Propensity Score Model
        preprocess = ColumnTransformer(
            [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), X_ps.columns.tolist())],
            remainder="drop"
        )
        ps_model = make_pipeline(preprocess, LogisticRegression(max_iter=6000, solver="lbfgs", class_weight="balanced"))
        ps_model.fit(X_ps, T)
        e = ps_model.predict_proba(X_ps)[:, 1].clip(1e-6, 1-1e-6)
        p_treat = T.mean()
        
        # ATT Weights (Stabilized)
        if effect == 'ATT':
            weights = T + (1 - T) * (p_treat / (1 - p_treat)) * (e / (1 - e))
        else:
            weights = T*(p_treat/e) + (1-T)*((1-p_treat)/(1-e)) 
        weights = weights / np.mean(weights)
        weights = np.clip(weights, None, np.percentile(weights, 99))
        
        gdf_final["weight"] = weights
        Z = ps_model.named_steps["columntransformer"].transform(X_ps)
        smd_post = mean_abs_smd(Z, T, weights)
        
        # WLS Outcome Model with Geographic Splines
        X_out_list =  [np.ones((len(T),1)),
                        T.reshape(-1,1), 
                    #    Xyspl.to_numpy(float)
                        # FE,
                       ]
        X_out = np.column_stack(X_out_list)
        mask = np.isfinite(Y) & np.isfinite(weights)
        
        res = sm.WLS(Y[mask], X_out[mask], weights=weights[mask]).fit()
        
        sensitivity_results.append({
            'K': k,
            f'{effect}': res.params[1],
            'SE': res.bse[1],
            'P-Value': res.pvalues[1],
            'Mean_SMD': smd_post
        })



    # 5. Output and Plotting (Refining the look)
    df_res = pd.DataFrame(sensitivity_results)
    current_color = palette[i] # One color per year
    # df_res.to_csv(f"./causal_sensitivity_ATT_{year}.csv", index=False)
    # plt.figure(figsize=(10, 8))
    # plt.tick_params(bottom=True, left=True, direction='out')
    df_res['CI_lower'] = df_res[effect] - 1.96 * df_res['SE']
    df_res['CI_upper'] = df_res[effect] + 1.96 * df_res['SE']

    plt.rcParams.update({
    'font.size' : 25,
    # "lines.linewidth": 2,
    # 'lines.markersize':10,
    "font.family":"arial",
    #"font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "mathtext.default": "it",    
    "mathtext.rm"  : "arial",
        })
    
    shifted_k = df_res['K'] + offsets[i]
    df_res['shifted_k'] = shifted_k
    plt.errorbar(df_res['shifted_k'], 
                 df_res[effect], yerr=1.96*df_res['SE'],
                fmt='o', 
                # color="#8eb4cb",
                color = current_color, 
                ecolor=current_color, 
                capsize=5, markersize=8,alpha=0.7,
                label=f'{year}')

    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel(r'Number of macroclusters ($k$)', )
    plt.ylabel(f'{effect} estimate ($\\tau$)',)
    plt.tick_params(length=10)
    plt.xticks(range(5, 26, 2))
    plt.ylim(-0.025, 0.15)
    
    # plt.ylim(-0.14, 0.02)
    # Adding significance markers for p < 0.05
    first_sig = True
    for i, row in df_res.iterrows():
        if row['P-Value'] < 0.05:
            if first_sig:
                plt.plot(row['shifted_k'], row[effect], '*', markersize=20, 
                         color = current_color,
                        #  label=f'Year {year}'
                        )
                first_sig = False
            else:
                plt.plot(row['shifted_k'], row[effect], '*', markersize=20, 
                         color = current_color,
                        #  color="#ff818c"
                         )
    # plt.title(f'Sensitivity of {Y_col} to Cluster Size (K)', fontsize=20)
    # plt.grid(axis='y', alpha=0.3)
leg = plt.legend(title="$\star$ denotes $p < 0.05$",
           ncols=3,
           loc='upper left', frameon=False)
leg._legend_box.align = "left"
plt.tight_layout()
# plt.savefig(f"causal_sensitivity_{effect}_DEP2024_ALL.png", dpi=300, bbox_inches="tight")
plt.close()
# plt.show()




# --- Configuration ---
Y_col = "SAMIs_Depression"
# years = [2018, 2019, 2020, 2021, 2022, 2023]
years = [2023]
k_values = range(5, 26)  # Full range as requested
subgroups = ['HH', 'HL', 'LH', 'LL']

effect = 'ATT'
for year in years:
    subgroup_results_all = []

    
    # 1. Prepare Data & Spatial Splines
    df_name = f"df_ozempic_{year}_COUNTY"
    
    # if df_name not in locals():
    #     ax.axis('off')
    #     continue
    
    # 1. Data Preparation & Merging
    df_current = locals()[df_name]
    gdf_year = ozempic_health_year(df_current, year)#.merge(county_health, left_on='COUNTY', right_on='FIPS').copy()
    STATE_FE = "STATEFP" if "STATEFP" in gdf_year.columns else None
    # Standardize variables (Use local copies to avoid SettingWithCopyWarning)
    df_selected = gdf_year.copy()
    
    # Helper for repetitive scaling logic
    cols_to_fix = [
        (f'Population_2022', 'REPUBLICAN_2024', 'SAMIs_REPUBLICAN_2024'),
        ('Population_2022', 'MHP', 'SAMIs_MHP'),
        (f'Population_2022', f'Unemployment', f'SAMIs_Unemployment'),
        (f'Population_2022', f'Depression', f'SAMIs_Depression'),
        (f'Population_2022', f'Obesity', f'SAMIs_Obesity'),
        (f'Population_2022', f'Diabetes', f'SAMIs_Diabetes'),
        (f'Population_2022', f'Conservative Protestant', f'SAMIs_Conservative_Protestant')
    ]
    
    for pop, raw, sami in cols_to_fix:
        # Fill zeros to avoid log/division errors if scaling_iloc requires it
        df_selected[raw] = df_selected[raw].replace(0, 1)
        df_selected[raw] = df_selected[raw].replace(np.nan, 1)
        # df_selected = df_selected[~df_selected[raw].isin([0, np.nan])]
        _, df_selected[sami], _, _ = scaling_iloc(df_selected[pop], df_selected[raw])

    # 2. Percolation Adjacency (Consistent Adjacency is better for time-series)
    # Using Rook as it's the standard for geographical 'physical' contact
    if year == 2018:
        W_county = KNN.from_dataframe(df_selected, k=5, ids=df_selected["COUNTY"].tolist())
    else:
        W_county = Rook.from_dataframe(df_selected, ids=df_selected["COUNTY"].tolist())
    idx_map, x, A, gdf_final = align_to_W(df_selected, W_county, id_col="COUNTY", x_col=f"SAMIs_Claims_{year}")

    # 3. Percolation Sweep
    real = percolation_sweep(A, x, step=1, descending=True)
    idx_pc, p_c, k_c = compute_pc(real, x) # Identify the threshold
    occ_mask = np.zeros(len(x), dtype=bool)
    occ_mask[real["order"][:k_c]] = True
    A_pc = A[occ_mask][:, occ_mask]
    n_comp, labels_pc = connected_components(A_pc, directed=False, return_labels=True)
    labels = np.full(len(x), -1)
    labels[occ_mask] = labels_pc
    gdf_final['cluster_pc'] = labels
    comp_all = gdf_final.loc[gdf_final["cluster_pc"] >= 0, "cluster_pc"]
    all_sizes = comp_all.value_counts().sort_values(ascending=False)
    
    # 2. Define Metabolic Subgroups
    ob_median = gdf_final['SAMIs_Obesity'].median()
    db_median = gdf_final['SAMIs_Diabetes'].median()
    
    def get_subgroup(row):
        db = row['SAMIs_Diabetes'] > db_median
        ob = row['SAMIs_Obesity'] > ob_median
        if db and ob: return 'HH'
        if db and not ob: return 'HL'
        if not db and ob: return 'LH'
        return 'LL'

    gdf_final['subgroup'] = gdf_final.apply(get_subgroup, axis=1)

    # 3. K-Sensitivity Loop
    comp_all = gdf_final.loc[gdf_final["cluster_pc"] >= 0, "cluster_pc"]
    all_sizes = comp_all.value_counts().sort_values(ascending=False)

    for k in k_values:
        # Define Treatment: Top-K clusters
        topK_ids = all_sizes.index[:k].to_numpy()
        gdf_final["T"] = gdf_final["cluster_pc"].isin(topK_ids).astype(int)
        
        T = gdf_final["T"].to_numpy(int)
        Y = pd.to_numeric(gdf_final[Y_col], errors="coerce").to_numpy(float)
        
        # 4. Create Interaction Regressors
        # This allows the model to calculate a separate ATT for each subgroup
        X_interactions = []
        for sg in ['HH', 'HL', 'LH', 'LL']:
            # Treatment interaction for this specific subgroup
            gdf_final[f'T_{sg}'] = ((gdf_final['T'] == 1) & (gdf_final['subgroup'] == sg)).astype(int)
            X_interactions.append(gdf_final[f'T_{sg}'])
        
        # Add baseline subgroup controls (to account for different starting depression levels)
        sg_dummies = pd.get_dummies(gdf_final['subgroup'], drop_first=True).astype(int)
        
        
        FE = pd.get_dummies(gdf_final[STATE_FE], drop_first=True, dtype=float) if STATE_FE else pd.DataFrame(index=gdf_final.index)
        FE = FE.reset_index(drop=True)
    
        # Geographic Splines (Controls for spatial auto-correlation)
        gdf_proj = gdf_final.to_crs(5070)
        cent = gdf_proj.geometry.centroid
        x_m = cent.x.to_numpy(float); y_m = cent.y.to_numpy(float)
        Xyspl = dmatrix("bs(x, df=4, include_intercept=False) + bs(y, df=4, include_intercept=False)",
                {"x": x_m, "y": y_m}, return_type="dataframe")
        
        X_cols = [
            # "SAMIs_MHP", 
          "SAMIs_Unemployment", 
          "SAMIs_REPUBLICAN_2024", 
          "SAMIs_Conservative_Protestant",
          f"SAMIs_Obesity",
          f"SAMIs_Diabetes"
          ]
        X_confounders = gdf_final[X_cols]
        # Combine everything: Interactions + Baseline Controls + Spatial Splines + Constant
        X_final = pd.concat([
            pd.concat(X_interactions, axis=1).reset_index(drop=True),
            sg_dummies.reset_index(drop=True),
            Xyspl.reset_index(drop=True),
            # X_confounders.reset_index(drop=True), 
            # FE,
        ], axis=1)
        X_final = sm.add_constant(X_final)
        
        X_ps = X_confounders.reset_index(drop=True)
        # Propensity Score Model
        preprocess = ColumnTransformer(
            [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), X_ps.columns.tolist())],
            remainder="drop"
        )
        ps_model = make_pipeline(preprocess, LogisticRegression(max_iter=6000, solver="lbfgs", class_weight="balanced"))
        ps_model.fit(X_ps, T)
        e = ps_model.predict_proba(X_ps)[:, 1].clip(1e-6, 1-1e-6)
        p_treat = T.mean()
        
        # ATT Weights (Stabilized)
        if effect == 'ATT':
            # weights = T + (1 - T) * (e / (1 - e))
            weights = T + (1 - T) * (p_treat / (1 - p_treat)) * (e / (1 - e))
        else:
            weights = T*(p_treat/e) + (1-T)*((1-p_treat)/(1-e)) 
        weights = weights / np.mean(weights)
        weights = np.clip(weights, None, np.percentile(weights, 99))
        
        gdf_final["weight"] = weights

        # 5. Run Weighted Least Squares (The Causal Model)
        try:
            mask =  ~np.isnan(Y) & ~np.isnan(weights)
            # # If the weights are too extreme, clip them to the 95th percentile for stability
            # res = sm.WLS(Y[mask], X_final[mask], weights=weights[mask]).fit()
            W_clipped = np.clip(weights, None, np.percentile(weights[mask], 95))
            res = sm.WLS(Y[mask], X_final[mask], weights=W_clipped).fit()
            # print(X_final.shape, Y.shape, gdf_final.shape)
            # Store results for each of the 4 subgroups
            for sg in ['HH', 'HL', 'LH', 'LL']:
                coef_name = f'T_{sg}'
                if coef_name in res.params:
                    subgroup_results_all.append({
                        'Year': year,
                        'K': k,
                        'Subgroup': sg,
                        'ATT': res.params[coef_name],
                        'SE': res.bse[coef_name],
                        'P-Value': res.pvalues[coef_name]
                    })
        except Exception as e:
            print(f"Error in Year {year}, K={k}: {e}")

    # --- 6. Visualization: The Multi-Year / Multi-K Plot ---
    df_plot = pd.DataFrame(subgroup_results_all)

    if not df_plot.empty:
        plt.figure(figsize=(14, 8))
        plt.tick_params(bottom=True, left=True, direction='out')
        plt.rcParams.update({
            'font.size' : 25,
            # "lines.linewidth": 2,
            # 'lines.markersize':10,
            "font.family":"arial",
            #"font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "mathtext.default": "it",    
            "mathtext.rm"  : "arial",
                })
        # Use a specific color palette for the 4 subgroups
        palette = {'HH': "#dd6868", 
                'HL': '#985bac',
                'LH': "#5b7dac",
                'LL': "#5bac69",}
        jitter_width = 0.3
        offsets = np.linspace(-jitter_width/2, jitter_width/2, len(subgroups))
        subgroup_offsets = dict(zip(subgroups, offsets))
        # 2. Add Error Bars (Confidence Intervals: 1.96 * SE)
        for subgroup in df_plot['Subgroup'].unique():
            current_color = palette[subgroup]
            sub_df = df_plot[df_plot['Subgroup'] == subgroup]
            sub_df['CI_lower'] = sub_df[effect] - 1.96 * sub_df['SE']
            sub_df['CI_upper'] = sub_df[effect] + 1.96 * sub_df['SE']

            current_offset = subgroup_offsets[subgroup]
            sub_df = df_plot[df_plot['Subgroup'] == subgroup].copy()
            
            # Apply the horizontal shift to the 'K' values
            shifted_k = sub_df['K'] + current_offset

            sub_df['shifted_k'] = shifted_k
            plt.errorbar(sub_df['shifted_k'], 
                        sub_df['ATT'], 
                        yerr=1.96*sub_df['SE'], 
                        fmt='o', 
                        color = current_color, 
                        ecolor=current_color, 
                        capsize=5, markersize=8,alpha=0.7,
                        label=f'{subgroup}')
            
            # 3. Add Significance Markers (Stars for p < 0.05)
            first_sig = True
            for i, row in sub_df.iterrows():
                if row['P-Value'] < 0.05:
                    if first_sig:
                        plt.plot(row['shifted_k'], row[effect], '*', markersize=20, 
                                color = current_color,
                                alpha=0.8,
                                #  label=f'Year {year}'
                                )
                        first_sig = False
                    else:
                        plt.plot(row['shifted_k'], row[effect], '*', markersize=20, 
                                color = current_color,alpha=0.8,)
                                #  color="#ff818c"
            # sig_points = sub_df[sub_df['P'] < 0.05]
            # plt.scatter(sig_points['K'], sig_points['ATT'], 
            #             marker='*', s=300, color=palette[subgroup], edgecolors='black', zorder=5)

        # 4. Final Formatting
        plt.axhline(0, color='black', linestyle='--', linewidth=2)
        # plt.title(f"Subgroup Causal Significance: Ozempic ATT on Depression\n($*$ denotes $p < 0.05$)", fontsize=18)
        plt.xlabel(r"Number of macroclusters ($k$)",fontsize=25)
        plt.ylabel(r"ATT estimate ($\tau$)",fontsize=25)
        plt.tick_params(length=10)
        plt.ylim(-0.1, 0.15)
        plt.xticks(range(5, 26, 2), fontsize=25)
        plt.yticks(fontsize=25)
        # Add a custom legend entry for the star
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='gray', lw=2),
                        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=15)]
        
        leg = plt.legend(title="$\star$ denotes $p < 0.05$", loc='upper left', 
                   ncol=4,
                   #bbox_to_anchor=(1, 1),
                   frameon=False)
        leg._legend_box.align = "left"
        
        plt.tight_layout()
        # plt.savefig(f"multilevel_sensitivity_DEP2024_{year}.png", dpi=300, bbox_inches="tight")
plt.show()