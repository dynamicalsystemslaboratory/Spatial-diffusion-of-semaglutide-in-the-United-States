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


# --- 1. Configuration & Global Style ---
years = [2018,2019,2020,2021,2022,2023]
top_num = 10
cmap = sns.color_palette("Reds_r", n_colors=top_num)
palette = {j+1: cmap[j] for j in range(top_num)}
palette[0] = "#86d7ffff"   # Blue for out-of-cluster
null_color = "#c6c4c4"    

# Setup Figure
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()

subcap_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

plt.rcParams.update({
    'font.size' : 40, # Adjusted for 2x3 grid readability
    "lines.linewidth": 2,
    "font.family":"arial",
    "mathtext.fontset": "cm",
    "mathtext.default": "rm",
    "mathtext.rm"  : "arial",
})
# --- 2. Temporal Loop ---
all_gdfs = []
for i, year in enumerate(years):
    ax = axes[i]
    
    df_name = f"df_ozempic_{year}_COUNTY"
    if df_name not in locals():
        ax.text(0.5, 0.5, f"Data for {year} Missing", ha='center')
        ax.axis('off')
        continue
    
    # Process Data
    df_current = locals()[df_name]
    # gdf_year = ozempic_health_year(df_current).copy()
    gdf_year = ozempic_health_year(df_current, year)#.merge(county_health, left_on='COUNTY',right_on='FIPS').copy()
    # Percolation Analysis
    if year == 2018:
        W_county = KNN.from_dataframe(gdf_year, k=5, ids=gdf_year["COUNTY"])
    else:
        W_county = Rook.from_dataframe(gdf_year, ids=gdf_year["COUNTY"].tolist())
    idx_map, x, A, gdf2 = align_to_W(gdf_year, W_county, id_col="COUNTY", x_col=f"SAMIs_Claims_{year}")

    real = percolation_sweep(A, x, step=1, descending = True)
    pc   = percolation_pc(real)
    idx_pc, p_c, k_c = compute_pc(real, x)
    # Run Null Envelope for this specific year (n_runs=50 for speed in temporal plot)
    env = null_envelope(A, x, step=1, n_runs=1, seed=year)
    
    # Cluster Extraction
    occ_mask = np.zeros(len(x), dtype=bool)
    occ_mask[real["order"][:k_c]] = True
    A_pc = A[occ_mask][:, occ_mask]
    n_comp, labels_pc = connected_components(A_pc, directed=False, return_labels=True)
    
    sizes = np.bincount(labels_pc)
    top_cluster_ids = np.argsort(sizes)[-top_num:][::-1]
    rank_map = {lab: j+1 for j, lab in enumerate(top_cluster_ids)}
    
    temp_labels = np.full(len(x), -1)
    temp_labels[occ_mask] = labels_pc
    gdf2['cluster_pc'] = temp_labels
    gdf2["top_rank"] = pd.Series(temp_labels).map(rank_map).fillna(0).astype(int).values
    gdf2["color"] = gdf2["top_rank"].map(palette)
                 # --- 3. Panel Plotting ---
    contig = gdf2[~gdf2['STATEFP'].isin(['AK', 'HI', 'PR', 'VI', 'GU', 'MP', 'AS'])] # 'AK', 'HI', 'PR', 'VI', 'GU', 'MP', 'AS'
    contig.plot(ax=ax, color=contig["color"], edgecolor="white", linewidth=0.1)
    
    # Overlay Boundaries
    contig_states = state_shape[~state_shape["STUSPS"].isin(['AK', 'HI', 'PR', 'VI', 'GU', 'MP', 'AS'])]
    contig_states.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.6)
    contig_states.plot(ax=ax, color=null_color, linewidth=0.3, alpha=0.2)
    ax.set_title(f"{year}",fontsize=25,)
    ax.set_axis_off()
    all_gdfs.append(gdf2)
    print('Results: ','largest size: ',np.max(sizes), 'num_com: ',n_comp)

# --- 5. Global Legend ---
handles = [Line2D([0], [0], marker='s', lw=0, color=palette[r], markersize=15) for r in range(1, top_num+1)]
handles.append(Line2D([0], [0], marker='s', lw=0, color=palette[0], markersize=15))
handles.append(Line2D([0], [0], marker='s', lw=0, color=null_color, alpha=0.5, markersize=15))
labels = [f"Rank #{r}" for r in range(1, top_num + 1)] + ["Small Clusters", "Null"]

fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.05), 
           fontsize=25,
           frameon=False, handletextpad=0.5, markerscale=1.2,)

plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3, hspace=0.0, bottom=0.1)
# plt.savefig("ozempic_temporal_percolation_descending_all_2024.png", dpi=300, bbox_inches='tight')
plt.show()


