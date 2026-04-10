import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import euclidean

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Serie A Scouting Engine", layout="wide", page_icon="⚽")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('processed_defenders_data.csv')
    df_pca = pd.read_csv('pca_coordinates.csv')
    with open('radar_metrics.json', 'r') as f:
        radar_features = json.load(f)
    return df, df_pca, radar_features

df, df_pca, radar_features = load_data()

# Clean names for dropdown menus
df['name'] = df['name'].str.strip()
player_list = sorted(df['name'].unique())

# Official cluster colors
cluster_colors = {
    0: 'dodgerblue',     # Proactive
    1: 'crimson',        # Positional
    2: 'mediumseagreen', # Balanced
    3: 'darkorange'      # Recovery
}

# --- SIDEBAR (ABOUT THE PROJECT) ---
with st.sidebar:
    st.markdown("### 👨‍💻 About the Project")
    st.markdown("""
    This application is the front-end interface of a comprehensive **Data Science pipeline** designed for football scouting.
    
    **Under the hood:**
    * **Data Source:** Raw defensive metrics powered by **Opta Analyst / Stats Perform**.
    * **Dataset Scope:** Serie A Center-Backs (Current Season) with >800 minutes played.
    * **Data Standardization:** Possession-Adjusted (PAdj) metrics.
    * **Dimensionality Reduction:** Principal Component Analysis (PCA).
    * **Machine Learning:** K-Means Clustering for tactical profiling.
    * **Similarity Engine:** Euclidean Distance scoring.
    """)
    st.write("---")
    
    # --- SEZIONE AUTORE (LA TUA FIRMA) ---
    st.markdown("### 👨‍💼 Author")
    st.markdown("**Matteo Vezzoli**")
    st.markdown("*Data Scientist | Sports Analytics*") 
    
    # Contatti
    st.markdown("""
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/matteo-vezzoli83)
    [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/matteovezzoli/)
    """)
# --- MAIN HEADER ---
st.title("⚽ Serie A Defensive Scouting Engine")
st.markdown("Explore the tactical DNA of defenders and discover the ideal profiles for your tactical system.")

# --- TABS (FLAT NAVIGATION) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Clone Finder", 
    "🧬 Player DNA", 
    "🌍 Market Explorer", 
    "📊 Metric Explorer", 
    "🛡️ Team DNA"
])

# ==========================================
# TAB 1: CLONE FINDER
# ==========================================
with tab1:
    st.header("Find the Ideal Tactical Replacement")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        target_player = st.selectbox("Select Target Player:", player_list, 
                                     index=player_list.index("Alessandro Bastoni") if "Alessandro Bastoni" in player_list else 0)
        top_n = st.slider("How many clones?", 3, 10, 5)
        
        # Search Engine Logic
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
        cluster_map = {0: "Proactive", 1: "Box Protector", 2: "Tactical Ground", 3: "Recovery"}
        clones = df.loc[top_indices].copy()
        clones['Cluster'] = clones['Cluster'].map(cluster_map)
        clones['Similarity %'] = [round(np.exp(-0.15 * x[1]) * 100, 1) for x in distances[:top_n]]

        # Essential columns selection
        display_cols = {'name': 'Player', 'team': 'Team', 'Cluster': 'Profile', 'Similarity %': 'Similarity %'}
        clones_final = clones[list(display_cols.keys())].rename(columns=display_cols)
        
        st.subheader("Top Tactical Matches")
        st.dataframe(
            clones_final.style.background_gradient(cmap='RdYlGn', subset=['Similarity %'])
                              .format(precision=1, subset=['Similarity %']),
            width='stretch'
        )

        # --- NEW SECTION: SIMILARITY EXPLANATION ---
        with st.expander("💡 How is Similarity % calculated?"):
            st.markdown("""
            The **Similarity %** is driven by an advanced mathematical model, not just a basic comparison of raw numbers:
            
            1. **Dimensionality Reduction:** We use **Principal Component Analysis (PCA)** to compress defensive metrics into a 4-dimensional "tactical space", capturing the true underlying DNA of a player.
            2. **Spatial Distance:** We calculate the **Euclidean Distance** between the target player and all other defenders within this 4D space.
            3. **Similarity Score:** We apply an exponential decay formula ($Similarity=e^{-0.15 \\cdot distance}\\cdot 100$) to convert this abstract mathematical distance into an intuitive 0-100% match score. 
            """)

    with col2:
        st.subheader("Interactive Radar Comparison")
        selected_clone = st.selectbox("Select a clone to compare:", options=clones['name'].tolist(), index=0)
        selected_sim = clones[clones['name'] == selected_clone]['Similarity %'].values[0]
        
        labels = [f.replace('_padj_z', '').replace('_z', '').title() for f in radar_features]
        fig_radar = go.Figure()
        
        # Target Trace & Selected Clone Trace
        for p, color in zip([target_player, selected_clone], ['crimson', 'dodgerblue']):
            val = df[df['name'] == p][radar_features].values[0].tolist()
            val += val[:1]
            fig_radar.add_trace(go.Scatterpolar(r=val, theta=labels + [labels[0]], fill='toself', name=p, line=dict(color=color)))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
            title=dict(text=f"{target_player} vs {selected_clone} ({selected_sim}%)", font=dict(size=18)),
            height=600
        )
        st.plotly_chart(fig_radar, width='stretch')
# ==========================================
# TAB 2: PLAYER DNA
# ==========================================
with tab2:
    st.header("Individual Profile Analysis")
    selected_player = st.selectbox("Select a player to view their DNA:", player_list, key='player_dna')
    
    player_data = df[df['name'] == selected_player].iloc[0]
    player_cluster = int(player_data['Cluster'])
    
    # Dictionary for tactical descriptions
    cluster_definitions = {
        0: {"name": "Proactive / Ground Defender", "desc": "Plays a proactive style. Steps up to engage opponents and win the ball back early rather than dropping deep. High volume in tackles and interceptions."},
        1: {"name": "Positional / Box Protector", "desc": "Traditional center-back. Excels at absorbing pressure, holding the defensive line, clearing the ball, and dominating aerial duels."},
        2: {"name": "Tactical / Ground-Focused", "desc": "Comfortable engaging opponents on the ground, often profiling as a wide center-back. Highly effective in ground duels but shows weakness in physical aerial battles."},
        3: {"name": "Passive / Recovery Defender", "desc": "Relies on pace, positioning, and covering open space rather than direct physical duels. Low volume of actions, but focuses on spatial control."}
    }
    
    # Profile Summary
    st.metric(label="Primary Tactical Profile", value=f"C{player_cluster}: {cluster_definitions[player_cluster]['name']}")
    st.info(f"**Tactic Focus:** {cluster_definitions[player_cluster]['desc']}")
        
    # DNA Breakdown Chart
    st.write("---")
    perc_cols = ['Cluster_0_%', 'Cluster_1_%', 'Cluster_2_%', 'Cluster_3_%']
    vals = player_data[perc_cols].values
    labels = ['C0: Proactive', 'C1: Positional', 'C2: Tactical', 'C3: Recovery']
    
    fig_bar = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker_color=[cluster_colors[0], cluster_colors[1], cluster_colors[2], cluster_colors[3]]
    ))
    fig_bar.update_layout(
        title=f"Tactical Membership % (DNA Breakdown) for {selected_player}", 
        xaxis_title="Percentage Match", 
        xaxis_range=[0, 100],
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_bar, width='stretch')
    
    # Methodology Expander
    with st.expander("📖 View Full Tactical Profile Definitions & Methodology"):
        st.markdown("""
        ### Cluster Interpretation: Defensive Profiles
                    
        **🔵 Cluster 0: High-Volume Ground Defenders**
        * **Data Overview:** Well above the league average in Tackles (+33%), Interceptions (+32%), and Ground Duels (+35%). Lower numbers in blocks and clearances.
        * **Tactical Profile:** A proactive style. Instead of dropping deep, these defenders step up to engage opponents and win the ball back early in the defensive or middle third.

        **🔴 Cluster 1: Penalty Box Defenders**
        * **Data Overview:** Leads the league in Clearances (+25%) and Blocks (+19%) with a very high aerial win rate (approx. 64%). Significantly lower tackling volume (-24%).
        * **Tactical Profile:** Traditional, positional center-backs. They excel at absorbing pressure, holding the defensive line, and clearing the ball from the penalty area.

        **🟢 Cluster 2: Ground-Focused Defenders**
        * **Data Overview:** Solid numbers in ground actions (Tackles +15%, Blocks +26%), but show a clear weakness in the air, with a significant drop in aerial volume (-10%) and win rate (-15.5%).
        * **Tactical Profile:** Comfortable engaging opponents on the ground, often profiling as wide center-backs in a three-man defense. They prioritize agility and ground duels over physical aerial battles.

        **🟠 Cluster 3: Low-Volume Recovery Defenders**
        * **Data Overview:** Sits below the league average in almost all volume metrics (Blocks -32%, Aerial Duels -32%, Clearances -24%).
        * **Tactical Profile:** Low volume does not mean poor defending. These players rely on pace, positioning, and the ability to cover open space or track runners rather than engaging in direct physical duels.
        """)

# ==========================================
# TAB 3: MARKET EXPLORER
# ==========================================
with tab3:
    st.header("Find Top Players by Tactical Profile")
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        sel_cluster_name = st.selectbox("Filter by Profile:", ["Proactive (C0)", "Box Protector (C1)", "Tactical Ground (C2)", "Recovery (C3)"])
        c_idx = int(sel_cluster_name[-2])
    with col_m2:
        sort_metric = st.radio("Rank by:", ["Tactical DNA %", "Minutes Played"])
    with col_m3:
        top_k = st.number_input("Show top:", 5, 20, 10)

    # Filter and Sort Logic
    discovery_df = df[df['Cluster'] == c_idx].copy()
    sort_col = f'Cluster_{c_idx}_%' if sort_metric == "Tactical DNA %" else 'mins'
    discovery_df = discovery_df.sort_values(by=sort_col, ascending=False).head(top_k)
    
    discovery_df['Cluster Name'] = sel_cluster_name.split(" (")[0]
    
    final_discovery = discovery_df[['name', 'team', 'mins', f'Cluster_{c_idx}_%']].rename(columns={
        'name': 'Player', 'team': 'Team', 'mins': 'Mins Played', f'Cluster_{c_idx}_%': 'DNA Match %'
    })
    
    # Display table with green background gradient and fixed decimals
    st.dataframe(
        final_discovery.style.background_gradient(cmap='YlGn', subset=['DNA Match %'])
                             .format(precision=1, subset=['DNA Match %']),
        width='stretch'
    )
# ==========================================
# TAB 4: METRIC EXPLORER
# ==========================================
with tab4:
    st.header("Interactive Metric Comparison")
    st.markdown("Analyze players across core defensive metrics. The scatter plot highlights different tactical profiles.")
    
    # --- NEW SECTION: GLOSSARY AND PAdj EXPLANATION ---
    with st.expander("📖 Methodology: What is PAdj & Core Metrics"):
        st.markdown("""
        ### Why PAdj (Possession-Adjusted)?
        Raw defensive stats can be highly misleading. A defender playing for a dominant team (e.g., 65% possession) will naturally attempt fewer tackles than a defender in a low-block team (e.g., 35% possession), simply because they have the ball more often.
        
        **PAdj (Possession-Adjusted)** metrics mathematically adjust these raw numbers as if every team had exactly 50% possession. This completely levels the playing field, allowing us to fairly compare a defender from Inter with a defender from Lecce based on their true defensive output.

        ### The 9 Core Metrics Explained
        * **Tackles (PAdj):** Volume of ground challenges made to dispossess an opponent.
        * **Interceptions (PAdj):** Reading the game to cut out passing lanes and intercept the ball.
        * **Possession Won (PAdj):** Times a player successfully won back possession for their team.
        * **Blocks (PAdj):** Standing in the way to physically block passes or shots.
        * **Clearances (PAdj):** Clearing the ball out of the defensive danger zone.
        * **Ground Duels - Total (PAdj):** Total volume of ground battles engaged.
        * **Ground Duels %:** Win rate in ground duels (Efficiency).
        * **Aerial Duels - Total (PAdj):** Total volume of aerial battles engaged.
        * **Aerial Duels %:** Win rate in aerial duels (Efficiency).
        """)
        
    st.write("---") # An elegant separator line
    
    # The 9 exact metrics requested
    plot_features = [
        'tackles_padj', 
        'ints_padj', 
        'pos won_padj', 
        'blocks_padj', 
        'clearances_padj', 
        'Ground Duels -total_padj', 
        'Ground Duels %', 
        'Aerial Duels -total_padj', 
        'Aerial Duels %'
    ]
    
    # Safety check: keep only columns that actually exist in the DataFrame
    plot_features = [col for col in plot_features if col in df.columns]
    
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        x_axis = st.selectbox("Select X-Axis Metric:", options=plot_features, index=0)
    with col_sc2:
        y_axis = st.selectbox("Select Y-Axis Metric:", options=plot_features, index=1 if len(plot_features)>1 else 0)
        
    # Create a copy of the DataFrame to avoid modifying the original
    df_plot = df.copy()
    
    # Map cluster names for a clear legend
    cluster_names_map = {
        0: '0: Proactive',
        1: '1: Positional',
        2: '2: Tactical',
        3: '3: Recovery'
    }
    df_plot['Tactical Profile'] = df_plot['Cluster'].map(cluster_names_map)
    
    # Exact color mapping (Blue for 0, Red for 1, Green for 2, Orange for 3)
    color_map = {
        '0: Proactive': 'dodgerblue',
        '1: Positional': 'crimson',
        '2: Tactical': 'mediumseagreen',
        '3: Recovery': 'darkorange'
    }
    
    # Create Interactive Scatter Plot
    fig_scatter = px.scatter(
        df_plot, 
        x=x_axis, 
        y=y_axis,
        text='name',                  # This makes the name appear next to the marker
        hover_name='name',      
        hover_data=['team', 'Tactical Profile'],    
        color='Tactical Profile',     
        color_discrete_map=color_map  # Apply your colors
    )
    
    # Marker and text aesthetics
    fig_scatter.update_traces(
        textposition='top center',               # Position text above the marker
        textfont=dict(size=10, color='dimgray'), # Legible but unobtrusive text
        marker=dict(size=11, opacity=0.85, line=dict(width=1, color='white')) # Clear markers with white borders
    )
    
    # Add dashed lines for the mean
    fig_scatter.add_hline(y=df_plot[y_axis].mean(), line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.add_vline(x=df_plot[x_axis].mean(), line_dash="dash", line_color="gray", opacity=0.5)
    
    # Layout formatting and legend positioning on the LEFT
    fig_scatter.update_layout(
        title=dict(text=f"{y_axis.replace('_padj', '').replace('_', ' ').title()} vs {x_axis.replace('_padj', '').replace('_', ' ').title()}", font=dict(size=20)),
        height=800, # Taller to give the names some breathing room
        xaxis_title=x_axis.replace('_padj', '').replace('_', ' ').title(),
        yaxis_title=y_axis.replace('_padj', '').replace('_', ' ').title(),
        legend=dict(
            title="Tactical Profiles",
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02, # Legend top-left INSIDE the chart
            bgcolor="rgba(255, 255, 255, 0.85)", # Semi-transparent white background
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    st.plotly_chart(fig_scatter, width='stretch')
# ==========================================
# TAB 5: TEAM DNA
# ==========================================
with tab5:
    st.header("Team Tactical Identity")
    st.markdown("Analyze how different teams build their defensive lines based on minutes played by each tactical profile.")
    
    # Mappatura dei nomi dei cluster per coerenza
    cluster_names_map = {
        0: 'Proactive',
        1: 'Box Protector',
        2: 'Tactical Ground',
        3: 'Recovery'
    }

    # --- SECTION 1: GLOBAL LEAGUE OVERVIEW ---
    # Inseriamo il grafico globale in un expander per mantenere pulita l'interfaccia
    with st.expander("🌍 View League-Wide Comparison (All Teams)", expanded=False):
        team_cluster_mins = df.groupby(['team', 'Cluster'])['mins'].sum().unstack(fill_value=0)
        team_dna_pct = team_cluster_mins.div(team_cluster_mins.sum(axis=1), axis=0) * 100
        team_dna_pct = team_dna_pct.sort_values(by=0, ascending=True).round(1)
        
        fig_team = go.Figure()
        
        for i in range(4):
            if i in team_dna_pct.columns:
                fig_team.add_trace(go.Bar(
                    y=team_dna_pct.index, 
                    x=team_dna_pct[i], 
                    name=f'C{i}: {cluster_names_map[i]}',
                    orientation='h', 
                    marker=dict(color=cluster_colors[i], line=dict(color='white', width=1))
                ))
                
        fig_team.update_layout(
            barmode='stack', title="Serie A Teams Defensive DNA",
            xaxis_title="Percentage of Total Defensive Minutes (%)",
            yaxis_title="", height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_team, width='stretch')

    st.write("---")

    # --- SECTION 2: SPECIFIC TEAM DEEP DIVE ---
    st.subheader("🔍 Specific Team Deep Dive")
    
    # Selezione della squadra
    team_list = sorted(df['team'].unique())
    selected_team = st.selectbox("Select a Team to analyze their defensive roster:", team_list)

    col_dna, col_roster = st.columns([1, 1.2])

    # Filtriamo i dati solo per la squadra selezionata
    team_df = df[df['team'] == selected_team].copy()
    team_cluster_dist = team_df.groupby('Cluster')['mins'].sum()
    team_total_mins = team_cluster_dist.sum()
    
    with col_dna:
        # Donut Chart del DNA della singola squadra
        labels = [cluster_names_map[i] for i in team_cluster_dist.index]
        values = team_cluster_dist.values
        colors = [cluster_colors[i] for i in team_cluster_dist.index]

        fig_donut = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=.45, # Dimensione del buco centrale (Donut)
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='label+percent'
        )])
        fig_donut.update_layout(
            title=dict(text=f"{selected_team} Tactical Breakdown", font=dict(size=18)),
            showlegend=False, # Nascondiamo la legenda perché i nomi sono già sul grafico
            height=400,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_donut, width='stretch')

        # Insight Dinamico (Generazione automatica del testo)
        dominant_cluster = team_cluster_dist.idxmax()
        dominant_pct = (team_cluster_dist.max() / team_total_mins) * 100
        st.info(f"💡 **Tactical Insight:** The defensive identity of {selected_team} is primarily driven by **{cluster_names_map[dominant_cluster]}** defenders, accounting for **{dominant_pct:.1f}%** of their total defensive minutes.")

    with col_roster:
        st.write(f"**{selected_team} Defensive Roster**")
        
        # Prepariamo la tabella dei giocatori per quella squadra
        team_df['Profile'] = team_df['Cluster'].map(cluster_names_map)
        display_df = team_df[['name', 'Profile', 'mins']].sort_values(by='mins', ascending=False)
        display_df = display_df.rename(columns={'name': 'Player', 'mins': 'Minutes Played'})
        
        # Mostriamo la tabella con un gradiente sui minuti giocati (chi gioca di più è più scuro)
        st.dataframe(
            display_df.style.background_gradient(cmap='Blues', subset=['Minutes Played']),
            width='stretch',
            hide_index=True # Rimuove la colonna dei numeri di riga a sinistra per un look più pulito
        )