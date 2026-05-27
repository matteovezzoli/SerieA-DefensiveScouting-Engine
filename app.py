import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import euclidean, pdist

# Page configuration  
st.set_page_config(page_title="Serie A Scouting Engine", layout="wide", page_icon="⚽")

# Data Loading 
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed_defenders_data.csv')
    df_pca = pd.read_csv('data/pca_coordinates.csv')
    with open('data/radar_metrics.json', 'r') as f:
        radar_features = json.load(f)
    return df, df_pca, radar_features

df, df_pca, radar_features = load_data()

# Clean names for dropdown menus
df['name'] = df['name'].str.strip()
player_list = sorted(df['name'].unique())

# Pre-compute min/max of pairwise distances in PCA space.
# Used by the Similarity% formula in the Clone Finder tab.
@st.cache_data
def get_distance_bounds(df_pca):
    coords = df_pca[['PC1', 'PC2', 'PC3', 'PC4']].values
    pairwise = pdist(coords, metric='euclidean')
    return float(pairwise.min()), float(pairwise.max())

DIST_MIN, DIST_MAX = get_distance_bounds(df_pca)

# Cluster naming is loaded dynamically from cluster_name_map.json
# so the notebook can re-assign cluster integer IDs without breaking the app.
try:
    with open('data/cluster_name_map.json', 'r') as f:
        _name_map_raw = json.load(f)
    CLUSTER_NAMES = {int(k): v for k, v in _name_map_raw.items()}
except FileNotFoundError:
    # Hard-coded fallback if the JSON is missing.
    CLUSTER_NAMES = {0: "Aggressor", 1: "Aerial Stopper", 2: "Ground Specialist", 3: "Coverage Defender"}

# Stable colour palette keyed by profile name (not by cluster integer)
PROFILE_COLORS = {
    'Aggressor':          'dodgerblue',
    'Aerial Stopper':     'crimson',
    'Ground Specialist':  'mediumseagreen',
    'Coverage Defender':  'darkorange',
}
# Build cluster_colors dict for downstream code (cluster integer -> hex/name color)
cluster_colors = {c: PROFILE_COLORS[CLUSTER_NAMES[c]] for c in CLUSTER_NAMES}

# Short labels used in chart legends. Profile names only, no internal cluster IDs.
CLUSTER_SHORT = {c: CLUSTER_NAMES[c] for c in CLUSTER_NAMES}

# Sidebar
with st.sidebar:
    st.markdown("### ⚽ About the Project")
    st.markdown(
        "This dashboard turns Serie A defensive data into **four tactical profiles**, "
        "so you can find player clones, compare profiles, and read team defensive identities at a glance."
    )

    st.markdown("**Under the hood:**")
    st.markdown("""
    * **Data Source:** Opta / Stats Perform, accessed via [TheAnalyst](https://theanalyst.com/competition/serie-a/stats).
    * **Dataset Scope:** Serie A Centre-Backs, end of 2025/26 season, with >1000 minutes played (77 players).
    * **Possession-Adjusted (PAdj) metrics:** raw stats are scaled as if every team had 50% possession, so a CB from Inter and one from Lecce can be compared fairly.
    * **Dimensionality reduction:** PCA compresses the 9 defensive metrics into a 4-dim tactical space.
    * **Clustering:** K-Means with centroids initialized from a Ward dendrogram cut at K=4.
    * **Similarity engine:** min-max normalized Euclidean distance in PCA space.
    """)

    st.markdown("📓 **See the full analysis:** [Notebook](https://github.com/matteovezzoli/SerieA-DefensiveScouting-Engine/blob/main/Clustering_defenders_SerieA.ipynb) · [HTML preview](https://github.com/matteovezzoli/SerieA-DefensiveScouting-Engine/blob/main/Clustering_defenders_SerieA.html)")

    st.write("---")

    st.markdown("### 👨‍💼 Author")
    st.markdown("**Matteo Vezzoli**")
    st.markdown("*Data Scientist*")

    st.markdown("""
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/matteo-vezzoli83)
    [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/matteovezzoli/SerieA-DefensiveScouting-Engine)
    """)
# MAIN HEADER 
st.title("⚽ Serie A Defensive Scouting Engine")
st.markdown("Explore the tactical DNA of defenders and discover the ideal profiles for your tactical system.")

# TABS
# Order: Clone Finder (1-to-1) -> Player DNA (1-to-many) -> Metric Explorer (all-on-2-axes panorama)
# -> Market Explorer (segment deep-dive) -> Team DNA (aggregated)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Clone Finder",
    "🧬 Player DNA",
    "📊 Metric Explorer",
    "🌍 Market Explorer",
    "🛡️ Team DNA"
])

# ==========================================
# TAB 1: CLONE FINDER
# ==========================================

with tab1:
    st.header("Find the Ideal Tactical Replacement")
    
    col1, col2 = st.columns([1.4, 2])
    
    with col1:
        target_player = st.selectbox("Select Target Player:", player_list,
                                     index=player_list.index("Alessandro Bastoni") if "Alessandro Bastoni" in player_list else 0)

        # Quick context card under the target selectbox: profile + team + minutes
        target_row = df[df['name'] == target_player].iloc[0]
        target_profile = CLUSTER_NAMES[int(target_row['Cluster'])]
        target_team = target_row['team']
        target_mins = int(target_row['mins'])
        st.caption(f"**{target_profile}** · {target_team} · {target_mins}'")

        top_n = st.slider("How many clones?", 3, 10, 5)

        target_idx = df[df['name'] == target_player].index[0]
        target_coords = df_pca.loc[target_idx, ['PC1', 'PC2', 'PC3', 'PC4']].values
        
        distances = []
        for idx, row in df_pca.iterrows():
            if idx == target_idx: continue
            dist = euclidean(target_coords, row[['PC1', 'PC2', 'PC3', 'PC4']].values)
            distances.append((idx, dist))
        
        distances.sort(key=lambda x: x[1])
        top_indices = [x[0] for x in distances[:top_n]]
        
        # Tactical Mapping & Similarity Score
        # Similarity% uses min-max normalization on the full dataset's pairwise distance range.
        # 100% = identical to the most similar pair in Serie A; 0% = as far apart as the most different pair.
        clones = df.loc[top_indices].copy()
        clones['Cluster'] = clones['Cluster'].map(CLUSTER_NAMES)
        clones['Similarity %'] = [
            round((1 - (x[1] - DIST_MIN) / (DIST_MAX - DIST_MIN)) * 100, 1)
            for x in distances[:top_n]
        ]

        # Essential columns selection
        display_cols = {'name': 'Player', 'team': 'Team', 'Cluster': 'Profile', 'Similarity %': 'Similarity %'}
        clones_final = clones[list(display_cols.keys())].rename(columns=display_cols)
        
        st.subheader("Top Tactical Matches")
        st.dataframe(
            clones_final.style.background_gradient(cmap='RdYlGn', subset=['Similarity %'])
                              .format(precision=1, subset=['Similarity %']),
            hide_index=True,
            column_config={
                'Player': st.column_config.TextColumn(width='medium'),
                'Team': st.column_config.TextColumn(width='small'),
                'Profile': st.column_config.TextColumn(width='medium'),
                'Similarity %': st.column_config.NumberColumn(format='%.1f%%', width='small'),
            }
        )

        
        with st.expander("💡 How is Similarity % calculated?"):
            st.markdown("""
            The **Similarity %** is computed in three steps:

            1. **Dimensionality reduction:** the 9 defensive metrics are compressed into a 4-dimensional tactical space via **Principal Component Analysis (PCA)**.
            2. **Spatial distance:** the **Euclidean distance** between the target player and every other defender is measured in that space.
            3. **Normalization:** the raw distance is rescaled to the 0-100% range using min-max normalization on the dataset's pairwise distances:

            $$Similarity\\% = \\left(1 - \\frac{d - d_{min}}{d_{max} - d_{min}}\\right) \\times 100$$

            This means **100% = the two most similar defenders in the league**, **0% = the two most different**, and everything else is in proportion. The score is meant for ranking comparisons within Serie A, not as an absolute match index across leagues.

            ---
            **Why PCA and not Euclidean distance directly on the 9 z-scored metrics?**
            Several defensive metrics in this dataset are strongly correlated — most notably Tackles ↔ Ground Duels Total (r = 0.80) and Clearances ↔ Blocks (r = 0.57). Computing distance directly on the 9 features would double-count correlated information, biasing the result towards whichever tactical dimension happens to be represented by multiple metrics. PCA decorrelates the features first, so each independent tactical axis contributes exactly once to the distance.
            """)

    with col2:
        st.subheader("Interactive Radar Comparison")
        selected_clone = st.selectbox("Select a clone to compare:", options=clones['name'].tolist(), index=0)
        selected_sim = clones[clones['name'] == selected_clone]['Similarity %'].values[0]
        
        labels = [f.replace('_padj_z', '').replace('_z', '').title() for f in radar_features]
        fig_radar = go.Figure()

        # Compute dynamic range so outliers (e.g. Acerbi's clearances z-score) are not clipped.
        target_vals = df[df['name'] == target_player][radar_features].values[0]
        clone_vals = df[df['name'] == selected_clone][radar_features].values[0]
        radial_min = min(target_vals.min(), clone_vals.min(), -3) - 0.3
        radial_max = max(target_vals.max(), clone_vals.max(), 3) + 0.3

        # Target Trace & Selected Clone Trace
        for p, color in zip([target_player, selected_clone], ['crimson', 'dodgerblue']):
            val = df[df['name'] == p][radar_features].values[0].tolist()
            val += val[:1]
            fig_radar.add_trace(go.Scatterpolar(r=val, theta=labels + [labels[0]], fill='toself', name=p, line=dict(color=color)))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[radial_min, radial_max])),
            title=dict(text=f"{target_player} vs {selected_clone} ({selected_sim}%)", font=dict(size=18)),
            height=600
        )
        st.plotly_chart(fig_radar, width='stretch')
        st.caption(
            "Values are **z-scores**: 0 = Serie A average for centre-backs, +1 = one standard deviation above, "
            "-1 = below. Anything above +1.5 is top-tier; below -1.5 is bottom-tier."
        )

# ==========================================
# TAB 2: PLAYER DNA
# ==========================================

with tab2:
    st.header("Individual Profile Analysis")
    selected_player = st.selectbox("Select a player to view their DNA:", player_list, key='player_dna')

    player_data = df[df['name'] == selected_player].iloc[0]
    player_cluster = int(player_data['Cluster'])

    # Quick context line under the selectbox: team, minutes, appearances
    st.caption(f"**{player_data['team']}** · {int(player_data['mins'])} minutes · {int(player_data['apps'])} appearances")
    
    # Tactical descriptions, keyed by profile name (stable across notebook reruns).
    # Numbers refer to centroid deviations from the league average (PAdj metrics).
    cluster_definitions = {
        "Aggressor": {
            "name": "Aggressor / Ball-Winner",
            "desc": "League leaders in defensive volume: top tier in tackles (+27%), interceptions (+27%), possessions won (+38%) and ground duels engaged (+30%). Typical centre-back of dominant, high-line teams that defend forward and recover the ball in midfield areas (Inter, Roma, Napoli, Juventus, Atalanta)."
        },
        "Aerial Stopper": {
            "name": "Aerial Stopper",
            "desc": "League leaders in clearances (+37%) and aerial win rate (~65%, +7pp vs league avg), with very high aerial volume (+26%) and above-average blocks (+21%) but reduced ground engagement (-12%). Classic stay-at-home centre-back of low/mid-block teams (Hellas Verona, Como, Lecce): holds the line, dominates the box, lets the ball come to them."
        },
        "Ground Specialist": {
            "name": "Ground Specialist",
            "desc": "High ground engagement (Ground Duels engaged +10%) but a clear aerial weakness: very low aerial volume (-29%) and the lowest aerial win rate of the four profiles (~49%, -9pp vs league avg). The label captures *where* these defenders engage (on the ground, not in the air), reflecting a profile shaped by aerial limitations rather than ground dominance. Fits as a wide centre-back in a back three or as a technical defender in possession-based sides that minimize aerial situations. Most represented at: Milan, Cagliari, Fiorentina."
        },
        "Coverage Defender": {
            "name": "Coverage Defender",
            "desc": "Below average across most defensive volume metrics: tackles -19%, possessions won -24%, ground duels engaged -20%, aerial duels engaged -12%. Common in deep-block teams that funnel play away from the centre-backs (Cremonese, Parma, Pisa). Often also a fullback redeployed as a wide centre-back. Low engagement is a tactical signature, not necessarily a weakness."
        }
    }

    player_profile_name = CLUSTER_NAMES[player_cluster]

    # Profile Summary
    st.metric(label="Primary Tactical Profile", value=cluster_definitions[player_profile_name]['name'])
    st.info(f"**Tactic Focus:** {cluster_definitions[player_profile_name]['desc']}")
        
    # DNA Breakdown Chart
    st.write("---")
    perc_cols = ['Cluster_0_%', 'Cluster_1_%', 'Cluster_2_%', 'Cluster_3_%']
    vals = player_data[perc_cols].values
    labels = [CLUSTER_SHORT[i] for i in range(4)]

    # Tactical identity classifier: how "pure" is this player's profile?
    max_pct = float(player_data['Max_Percentage'])
    if max_pct > 50:
        identity_label, identity_color = "Tactical Specialist", "🟢"
        identity_desc = "a clear, single-profile player"
    elif max_pct >= 35:
        identity_label, identity_color = "Tactical Hybrid", "🟡"
        identity_desc = "blends two or more profiles"
    else:
        identity_label, identity_color = "Tactical Allrounder", "🟠"
        identity_desc = "spread across multiple profiles, no dominant trait"
    st.markdown(
        f"{identity_color} **Tactical Identity: {identity_label}** "
        f"(top match = {max_pct:.1f}% on {player_profile_name}). *{identity_desc}*"
    )
    
    # Create Bar Chart with embedded percentage labels
    fig_bar = go.Figure(go.Bar(
        x=vals, 
        y=labels, 
        orientation='h',
        text=[f"{v:.1f}%" for v in vals], # Format values to 1 decimal place (e.g., 35.2%)
        textposition='auto',              # Smart positioning: inside if bar is long, outside if short
        marker_color=[cluster_colors[0], cluster_colors[1], cluster_colors[2], cluster_colors[3]]
    ))
    
    # Apply font styling for maximum readability on the bars
    fig_bar.update_traces(textfont=dict(size=14, color='black'))
    
    # Clean layout configuration
    fig_bar.update_layout(
        title=dict(text=f"<b>Tactical Membership % for {selected_player}</b>", font=dict(size=16)), 
        xaxis_title="Percentage Match", 
        xaxis_range=[0, 105], 
        height=350,
        margin=dict(l=20, r=40, t=50, b=20), 
        template="plotly_white"              
    )
    
    st.plotly_chart(fig_bar, width='stretch')

    # Most similar defenders (mini-table). Re-uses the same PCA + min-max similarity
    # logic as the Clone Finder tab, but condensed to 3 rows for quick context.
    st.markdown("#### 🔍 Most similar defenders in Serie A")
    similar_idx = df[df['name'] == selected_player].index[0]
    similar_coords = df_pca.loc[similar_idx, ['PC1', 'PC2', 'PC3', 'PC4']].values
    similar_dists = []
    for idx, row in df_pca.iterrows():
        if idx == similar_idx:
            continue
        d = euclidean(similar_coords, row[['PC1', 'PC2', 'PC3', 'PC4']].values)
        similar_dists.append((idx, d))
    similar_dists.sort(key=lambda x: x[1])
    top3 = similar_dists[:3]
    top3_df = df.loc[[i for i, _ in top3], ['name', 'team']].copy()
    top3_df['Profile'] = df.loc[[i for i, _ in top3], 'Cluster'].map(CLUSTER_NAMES).values
    top3_df['Similarity %'] = [
        round((1 - (d - DIST_MIN) / (DIST_MAX - DIST_MIN)) * 100, 1)
        for _, d in top3
    ]
    top3_df = top3_df.rename(columns={'name': 'Player', 'team': 'Team'}).reset_index(drop=True)
    st.dataframe(
        top3_df.style.background_gradient(cmap='RdYlGn', subset=['Similarity %'])
                     .format(precision=1, subset=['Similarity %']),
        width='stretch',
        hide_index=True
    )

    # Methodology Expander
    with st.expander("📖 View Full Tactical Profile Definitions & Methodology"):
        st.markdown("""
        ### Cluster Interpretation: Defensive Profiles
        *Percentages below indicate deviation from the league average (computed on PAdj metrics).*

        **🔵 Aggressor / Ball-Winner**
        * **Data Signature:** Tackles +27%, Interceptions +27%, Possessions Won +38%, Ground Duels engaged +30%, Aerial Duels engaged +22%. The highest defensive volume profile in the league.
        * **Tactical Profile:** Centre-backs of dominant, high-line teams. They step up rather than drop, win the ball back in midfield areas, and engage aggressively across the pitch. Top sources: Inter, Roma, Napoli, Juventus, Atalanta.

        **🔴 Aerial Stopper**
        * **Data Signature:** Clearances +37%, Aerial Duels engaged +26%, Aerial win rate ~65% (+7pp vs league avg), Blocks +21%. Ground engagement is lower than the league average (Tackles -3%, Ground Duels -12%).
        * **Tactical Profile:** Classic stay-at-home centre-back. Holds the defensive line, absorbs pressure, dominates the box aerially. Top sources: Hellas Verona, Como, Lecce.

        **🟢 Ground Specialist**
        * **Data Signature:** High Ground Duels volume (+10%) and decent Tackles (+3%), with a clear aerial weakness: Aerial Duels engaged -29%, the lowest Aerial win rate of the four profiles (~49%, -9pp vs league avg), and low Clearances (-28%).
        * **Tactical Profile:** Defenders whose engagement happens on the ground rather than in the air. Their tactical value lies in *being technically available* on the ground in systems built to neutralise aerial situations — high lines, possession football, back-three setups with wide CBs. The profile is defined by what these defenders avoid (aerial duels), not by what they dominate. Top sources: Milan, Cagliari, Fiorentina.

        **🟠 Coverage Defender**
        * **Data Signature:** Below league average across most volume metrics: Tackles -19%, Pos Won -24%, Ground Duels engaged -20%, Aerial Duels engaged -12%. Blocks slightly above average (+14%).
        * **Tactical Profile:** Two overlapping sub-profiles emerge here: fullbacks redeployed as wide centre-backs, and defenders in deep blocks where the team funnels play away from them. Low engagement is a tactical signature, not necessarily a quality judgement. Top sources: Cremonese, Parma, Pisa.
        """)

# ==========================================
# TAB 4: MARKET EXPLORER
# ==========================================

with tab4:
    st.header("Find Top Players by Tactical Profile")
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        # Name -> integer ID lookup; the UI shows only the profile name.
        _name_to_id = {name: cid for cid, name in CLUSTER_NAMES.items()}
        profile_options = list(_name_to_id.keys())
        sel_cluster_name = st.selectbox("Filter by Profile:", profile_options)
        c_idx = _name_to_id[sel_cluster_name]
    with col_m2:
        sort_metric = st.radio("Rank by:", ["Tactical DNA %", "Minutes Played"])
    with col_m3:
        top_k = st.number_input("Show top:", 5, 20, 10)

    # Summary statement: how big is this profile across the league?
    total_in_profile = (df['Cluster'] == c_idx).sum()
    teams_in_profile = df[df['Cluster'] == c_idx]['team'].nunique()
    # Cap "top K" if K > how many are actually in the cluster
    shown = min(top_k, total_in_profile)
    profile_label = sel_cluster_name + "s" if total_in_profile != 1 else sel_cluster_name
    teams_label = "team" if teams_in_profile == 1 else "teams"
    st.markdown(
        f"Showing top **{shown}** *{sel_cluster_name}* defenders in Serie A, "
        f"ranked by {sort_metric.lower()}. "
        f"The league has **{total_in_profile}** {profile_label} in total across **{teams_in_profile}** {teams_label}."
    )

    # Filter and Sort Logic
    discovery_df = df[df['Cluster'] == c_idx].copy()
    sort_col = f'Cluster_{c_idx}_%' if sort_metric == "Tactical DNA %" else 'mins'
    discovery_df = discovery_df.sort_values(by=sort_col, ascending=False).head(top_k)

    # Build the final table with per-90 PAdj stats so the scout sees the raw signal
    # behind the profile, not just the DNA % aggregate.
    padj_cols = {
        'tackles_padj': 'Tackles',
        'ints_padj': 'Ints',
        'pos won_padj': 'Pos Won',
        'blocks_padj': 'Blocks',
        'clearances_padj': 'Clear.',
        'Ground Duels -total_padj': 'GD Vol',
        'Ground Duels %': 'GD %',
        'Aerial Duels -total_padj': 'AD Vol',
        'Aerial Duels %': 'AD %',
    }
    cols_to_show = ['name', 'team', 'mins', f'Cluster_{c_idx}_%'] + list(padj_cols.keys())
    rename_map = {'name': 'Player', 'team': 'Team', 'mins': 'Mins',
                  f'Cluster_{c_idx}_%': 'DNA %'}
    rename_map.update(padj_cols)

    final_discovery = discovery_df[cols_to_show].rename(columns=rename_map)

    # Format: 1 decimal for floats, integer for Mins, gradient on DNA % for visual ranking
    numeric_cols = ['DNA %'] + list(padj_cols.values())
    format_map = {col: "{:.1f}" for col in numeric_cols}
    format_map['Mins'] = "{:.0f}"
    st.dataframe(
        final_discovery.style
            .background_gradient(cmap='YlGn', subset=['DNA %'])
            .format(format_map),
        width='stretch',
        hide_index=True,
    )
    st.caption(
        "Per-90 metrics are **possession-adjusted (PAdj)** so values are directly comparable across teams. "
        "GD = Ground Duels, AD = Aerial Duels. DNA % indicates how strongly the player belongs to the selected profile."
    )
# ==========================================
# TAB 3: METRIC EXPLORER
# ==========================================

with tab3:
    st.header("Interactive Metric Comparison")
    st.markdown("Analyze players across core defensive metrics. The scatter plot dynamically highlights tactical extremes.")
    
    with st.expander("📖 Methodology: What is PAdj & Core Metrics"):
        st.markdown("""
        ### Why PAdj (Possession-Adjusted)?
        Defensive actions only happen when the **opponent** has the ball. A centre-back of a high-possession team is less exposed to defensive situations than one of a low-block side, simply because the ball spends less time near his own goal. Raw per-90 stats therefore penalize defenders of dominant teams: they get fewer chances to act, not because they're worse, but because they're defending less often.

        **PAdj rescales every defensive count as if the player's team had exactly 50% possession**, so the metric reflects defensive **rate per unit of out-of-possession time**, not per match clock. The standard Opta formula is:

        $$\\text{PAdj stat} = \\text{Raw stat per 90} \\times \\frac{50}{\\text{Opponent possession \\%}}$$

        After this adjustment, defenders from high- and low-possession teams are directly comparable on the same scale. Win-rate metrics (Ground/Aerial Duels %) are already ratios, so they don't need PAdj adjustment.

        ### The 9 Core Metrics Explained
        * **Tackles (PAdj):** Volume of ground challenges made to dispossess an opponent.
        * **Interceptions (PAdj):** Reading the game to cut out passing lanes and intercept the ball.
        * **Possession Won (PAdj):** Ball recoveries credited to the player, regardless of how possession was regained (tackle, interception, opponent error, loose ball).
        * **Blocks (PAdj):** Standing in the way to physically block passes or shots.
        * **Clearances (PAdj):** Clearing the ball out of the defensive danger zone.
        * **Ground Duels - Total (PAdj):** Total volume of ground battles engaged.
        * **Ground Duels %:** Win rate in ground duels (Efficiency).
        * **Aerial Duels - Total (PAdj):** Total volume of aerial battles engaged.
        * **Aerial Duels %:** Win rate in aerial duels (Efficiency).
        """)
        
    st.write("---") 
    
    # Map clean labels to technical column names
    metric_mapping = {
        "Tackles": "tackles_padj",
        "Interceptions": "ints_padj",
        "Possession Won": "pos won_padj",
        "Blocks": "blocks_padj",
        "Clearances": "clearances_padj",
        "Ground Duels Total": "Ground Duels -total_padj",
        "Ground Duels %": "Ground Duels %",
        "Aerial Duels Total": "Aerial Duels -total_padj",
        "Aerial Duels %": "Aerial Duels %"
    }
    
    available_labels = [label for label, col in metric_mapping.items() if col in df.columns]
    
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        x_label = st.selectbox("Select X-Axis Metric:", options=available_labels, index=0)
        x_axis = metric_mapping[x_label]
    with col_sc2:
        y_label = st.selectbox("Select Y-Axis Metric:", options=available_labels, index=1 if len(available_labels)>1 else 0)
        y_axis = metric_mapping[y_label]
        
    df_plot = df.copy()
    
    mean_x = df_plot[x_axis].mean()
    mean_y = df_plot[y_axis].mean()
    std_x = df_plot[x_axis].std()
    std_y = df_plot[y_axis].std()

    # Calculate normalized Euclidean distance (Z-Score approach) 
    df_plot['dist_from_mean_norm'] = (
        ((df_plot[x_axis] - mean_x) / std_x)**2 + 
        ((df_plot[y_axis] - mean_y) / std_y)**2
    )**0.5

    # Extract the names of the Top 3 players in the Top-Right quadrant
    top_right_3 = df_plot[(df_plot[x_axis] > mean_x) & (df_plot[y_axis] > mean_y)]\
                  .nlargest(3, 'dist_from_mean_norm')['name'].tolist()

    # Extract the names of the Top 3 players in the Bottom-Left quadrant
    bottom_left_3 = df_plot[(df_plot[x_axis] < mean_x) & (df_plot[y_axis] < mean_y)]\
                    .nlargest(3, 'dist_from_mean_norm')['name'].tolist()

    # Combined list of players to highlight
    dynamic_highlights = top_right_3 + bottom_left_3
    # -----------------------------------------------------------------
    
    cluster_names_map = {i: CLUSTER_NAMES[i] for i in range(4)}
    df_plot['Tactical Profile'] = df_plot['Cluster'].map(cluster_names_map)
    color_map = {cluster_names_map[i]: cluster_colors[i] for i in range(4)}
    
    fig_scatter = go.Figure()

    # Create the chart iterating over clusters
    for cluster_name, color in color_map.items():
        df_sub = df_plot[df_plot['Tactical Profile'] == cluster_name]
        
        if not df_sub.empty:
            texts = []
            text_colors = []
            text_sizes = []
            marker_sizes = []
            
            # Apply styling player by player
            for _, row in df_sub.iterrows():
                name = row['name']
                if name in dynamic_highlights:
                    texts.append(f"<b>{name}</b>") 
                    text_colors.append(color)    
                    text_sizes.append(13)          
                    marker_sizes.append(14)        
                else:
                    texts.append(name)             
                    text_colors.append("dimgray") 
                    text_sizes.append(9)           
                    marker_sizes.append(10)        
            
            # Add the trace for this specific Cluster
            fig_scatter.add_trace(go.Scatter(
                x=df_sub[x_axis],
                y=df_sub[y_axis],
                mode='markers+text',
                name=cluster_name,
                text=texts,
                textposition='top center',
                textfont=dict(color=text_colors, size=text_sizes),
                marker=dict(color=color, size=marker_sizes, line=dict(width=1, color='white'), opacity=0.85),
                hoverinfo='text',
                hovertext=df_sub['name'] + " (" + df_sub['team'] + ")"
            ))
    
    # Add average reference lines
    fig_scatter.add_hline(y=mean_y, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.add_vline(x=mean_x, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Pearson correlation between the two selected metrics (same value as in the
    # notebook's correlation matrix). Used both in the title and as a textual callout.
    pearson_r = df[x_axis].corr(df[y_axis])
    if abs(pearson_r) < 0.2:
        corr_strength = "very weak"
    elif abs(pearson_r) < 0.4:
        corr_strength = "weak"
    elif abs(pearson_r) < 0.6:
        corr_strength = "moderate"
    elif abs(pearson_r) < 0.8:
        corr_strength = "strong"
    else:
        corr_strength = "very strong"
    corr_direction = "positive" if pearson_r >= 0 else "negative"

    # Clean layout configuration
    fig_scatter.update_layout(
        title=dict(
            text=(
                f"<b>{y_label} vs {x_label}</b>"
                f"<br><span style='font-size:13px'>Pearson r = {pearson_r:+.2f} "
                f"({corr_strength} {corr_direction} correlation)</span>"
            ),
            font=dict(size=20)
        ),
        height=800,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(
            title="Tactical Profiles", orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.02,
            bgcolor="rgba(0, 0, 0, 0)",
            bordercolor="rgba(128, 128, 128, 0.4)",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=90, b=40)
    )

    st.plotly_chart(fig_scatter, width='stretch')
    st.caption(
        "All per-90 volume metrics shown here are **possession-adjusted (PAdj)** — rescaled as if every team had 50% possession, so defenders from high- and low-possession teams are directly comparable. "
        "Win-rate metrics (Ground Duels %, Aerial Duels %) are already ratios and not adjusted. See the methodology expander above for details."
    )

    # Top / Bottom performers callout
    col_top, col_bot = st.columns(2)
    with col_top:
        if top_right_3:
            st.markdown(
                f"**🟢 Top performers** (high {x_label} **and** high {y_label}): "
                + ", ".join(f"**{p}**" for p in top_right_3)
            )
        else:
            st.markdown(f"**🟢 Top performers**: no player above the mean on both axes.")
    with col_bot:
        if bottom_left_3:
            st.markdown(
                f"**🔴 Bottom performers** (low {x_label} **and** low {y_label}): "
                + ", ".join(f"**{p}**" for p in bottom_left_3)
            )
        else:
            st.markdown(f"**🔴 Bottom performers**: no player below the mean on both axes.")
    st.caption(
        "Highlighted players are the most extreme cases in each diagonal quadrant, "
        "measured by normalized Euclidean distance from the league means on the two selected metrics."
    )
# ==========================================
# TAB 5: TEAM DNA
# ==========================================
with tab5:
    st.header("Team Tactical Identity")
    st.markdown("Analyze how different teams build their defensive lines based on minutes played by each tactical profile.")
    
    
    cluster_names_map = CLUSTER_NAMES

    # GLOBAL LEAGUE OVERVIEW
    with st.expander("🌍 View League-Wide Comparison (All Teams)", expanded=False):
        # Aggregate minutes per (team, profile name) so the ordering is profile-name-driven
        # rather than dependent on the arbitrary integer cluster ID.
        team_profile_mins = df.groupby(['team', 'Cluster_Name'])['mins'].sum().unstack(fill_value=0)
        team_dna_pct = team_profile_mins.div(team_profile_mins.sum(axis=1), axis=0) * 100

        # Canonical profile order, left to right inside the stacked bar.
        profile_order = ['Aggressor', 'Aerial Stopper', 'Ground Specialist', 'Coverage Defender']
        profile_order_present = [p for p in profile_order if p in team_dna_pct.columns]
        team_dna_pct = team_dna_pct.reindex(columns=profile_order_present, fill_value=0).round(1)

        # Sort teams by Aggressor share descending, so the most proactive teams sit at the top.
        # Plotly draws horizontal bars bottom-up, so the actual sort is ascending here.
        sort_key = 'Aggressor' if 'Aggressor' in team_dna_pct.columns else profile_order_present[0]
        team_dna_pct = team_dna_pct.sort_values(by=sort_key, ascending=True)

        fig_team = go.Figure()
        for profile in profile_order_present:
            fig_team.add_trace(go.Bar(
                y=team_dna_pct.index,
                x=team_dna_pct[profile],
                name=profile,
                orientation='h',
                marker=dict(color=PROFILE_COLORS[profile], line=dict(color='white', width=1))
            ))

        fig_team.update_layout(
            barmode='stack', title="Serie A Teams Defensive DNA",
            xaxis_title="Percentage of Total Defensive Minutes (%)",
            yaxis_title="", height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_team, width='stretch')

    st.write("---")

    # SPECIFIC TEAM DEEP DIVE 
    st.subheader("🔍 Specific Team Deep Dive")
    
    # Teams
    team_list = sorted(df['team'].unique())
    selected_team = st.selectbox("Select a Team to analyze their defensive roster:", team_list)

    col_dna, col_roster = st.columns([1, 1.2])

    
    team_df = df[df['team'] == selected_team].copy()
    team_cluster_dist = team_df.groupby('Cluster')['mins'].sum()
    team_total_mins = team_cluster_dist.sum()
    
    with col_dna:
        # Donut Chart 
        labels = [cluster_names_map[i] for i in team_cluster_dist.index]
        values = team_cluster_dist.values
        colors = [cluster_colors[i] for i in team_cluster_dist.index]

        fig_donut = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=.45, 
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='label+percent'
        )])
        fig_donut.update_layout(
            title=dict(text=f"{selected_team} Tactical Breakdown", font=dict(size=18)),
            showlegend=False, 
            height=400,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_donut, width='stretch')

        
        dominant_cluster = team_cluster_dist.idxmax()
        dominant_pct = (team_cluster_dist.max() / team_total_mins) * 100
        st.info(f"💡 **Tactical Insight:** The defensive identity of {selected_team} is primarily driven by **{cluster_names_map[dominant_cluster]}** defenders, accounting for **{dominant_pct:.1f}%** of their total defensive minutes.")

    with col_roster:
        st.write(f"**{selected_team} Defensive Roster**")
        
        # Table of the team's players
        team_df['Profile'] = team_df['Cluster'].map(cluster_names_map)
        display_df = team_df[['name', 'Profile', 'mins']].sort_values(by='mins', ascending=False)
        display_df = display_df.rename(columns={'name': 'Player', 'mins': 'Minutes Played'})
        
        
        st.dataframe(
            display_df.style.background_gradient(cmap='Blues', subset=['Minutes Played']),
            width='stretch',
            hide_index=True 
        )

