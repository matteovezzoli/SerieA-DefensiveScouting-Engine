# ⚽ Serie A Defensive Scouting Engine (25/26)
**A Data-Driven Pipeline and Web App for Tactical Profiling of Center-Backs.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---
🚀 **[Live App: Explore the Scouting Engine](https://seriea-defensive-engine.streamlit.app/)**
---

## 📌 Project Overview
This project presents an analytical study of Serie A center-backs for the 2025/2026 season. The goal is to move beyond standard volume stats by grouping defenders based on their actual playing style.

The project is structured in two connected parts: an interactive **Streamlit Web Application** serving as the frontend tool for scouting, powered by a rigorous **Jupyter Notebook** that handles the complete exploratory and machine learning pipeline.

---

## 🎯 The Four Tactical Profiles

The clustering pipeline identifies four data-driven profiles among Serie A center-backs. The labels are derived directly from the centroid signatures, not assigned a priori. Each name describes what the data actually shows.

All percentages below are **deviations of the cluster centroid from the league average** on PAdj per-90 metrics (so a high-possession Inter CB and a low-possession Lecce CB are on the same scale). For win-rate metrics (Aerial Duels %, Ground Duels %) the value in parentheses is the absolute centroid, with the deviation in percentage-points (pp).

| Profile | Color | Data Signature (vs. league avg.) | Football Interpretation |
|---|---|---|---|
| **Aggressor / Ball-Winner** | 🔵 | Possessions Won **+38%**, Ground Duels engaged **+30%**, Interceptions **+27%**, Tackles **+27%**, Aerial Duels engaged **+22%**. Win-rate is roughly league-average (AD% 60.5, +3pp). Blocks **-18%** and Clearances **-10%** — they win the ball before it reaches the box. | Centre-backs of dominant, high-line teams. They step up rather than drop and recover possession in midfield areas. The most stable archetype across stability tests. Most represented at: **Inter, Roma, Napoli, Juventus, Atalanta**. |
| **Aerial Stopper** | 🔴 | Clearances **+37%**, Aerial Duels engaged **+26%**, Blocks **+21%**, Aerial win rate **65%** (+7pp, the highest of the four profiles). Ground engagement is below the league average (Ground Duels engaged **-12%**, Tackles **-3%**). | Classic stay-at-home centre-back. Holds the defensive line, absorbs crosses, dominates the box. Most represented at: **Hellas Verona, Como, Lecce**. |
| **Ground Specialist** | 🟢 | Ground Duels engaged **+10%**, but the defining trait is a clear aerial weakness: Aerial Duels engaged **-29%**, Aerial win rate **49%** (-9pp, the worst of the four), Clearances **-28%**. Tackles roughly league-average (+3%). | Defenders who can compete on the ground but are physically out-matched in the air. Often fit as wide centre-backs in a back three or as technical defenders in possession-based sides where the team prevents aerial duels from happening. Most represented at: **Milan, Cagliari, Fiorentina**. |
| **Coverage Defender** | 🟠 | Below average across most volume metrics: Possessions Won **-24%**, Ground Duels engaged **-20%**, Interceptions **-19%**, Tackles **-19%**, Aerial Duels engaged **-12%**. Slight uplift on box actions (Blocks **+14%**, Clearances **+4%**). Win-rates near league average. | Low-engagement defenders, two overlapping sub-profiles: fullbacks redeployed wide in a back three, and centre-backs in deep blocks where the team funnels play away from them. Low volume is a tactical signature, not a quality judgement. Most represented at: **Cremonese, Parma, Pisa**. |

---

## 🖥️ The Frontend: Streamlit Dashboard
The `app.py` Streamlit dashboard makes the model usable by anyone, no Python required. It reads the processed datasets and PCA coordinates and exposes five tabs, ordered from one-to-one comparison up to league-level aggregation:

* **🔍 Clone Finder:** pick a target defender and get the top *N* tactical replacements ranked by **Similarity %** (min-max normalized Euclidean distance in the 4-D PCA space, where 100% = the two most similar defenders in the dataset and 0% = the two most different). Side-by-side interactive radar comparison on z-score scale.
* **🧬 Player DNA:** view a defender's **soft-cluster membership** across the four profiles, plus a *Tactical Identity* classifier (Specialist / Hybrid / Allrounder) computed from the top-membership percentage. Also surfaces the three most similar defenders in the league.
* **📊 Metric Explorer:** cross-analyze any two of the 9 PAdj defensive metrics on a scatter plot colored by tactical profile. The chart auto-highlights the top-3 and bottom-3 performers (most extreme cases in the top-right and bottom-left quadrants by normalized Euclidean distance from the league means) and reports the **Pearson correlation** between the two metrics in real time.
* **🌍 Market Explorer:** filter the dataset by tactical profile, rank by Tactical DNA % or minutes played, and view a per-90 PAdj table (tackles, interceptions, possessions won, blocks, clearances, ground/aerial duels volume & win-rate). Useful as a shortlist generator.
* **🛡️ Team DNA:** league-wide stacked bar (all 20 teams, profile share weighted by minutes) plus a per-team deep-dive donut chart and roster table. Shows how each team builds its defensive line.

---

## 🧠 Under the Hood: The Data Science Pipeline
Full reproducible analysis in [`Clustering_defenders_SerieA.ipynb`](Clustering_defenders_SerieA.ipynb). The pipeline runs end-to-end in seven steps:

1. **Possession Adjustment (PAdj).** Volume metrics rescaled as if every team had 50% possession (`x_padj = x · 50 / (100 - team_poss)`), so defenders from high- and low-possession teams sit on the same scale.
2. **Z-score standardization.** Mandatory for scale-sensitive methods downstream (PCA, K-Means). Applied to all 9 features (volume PAdj + win-rate percentages).
3. **EDA.** Correlation matrix, four tactical scatter maps (Ground vs Tackles, Tackles vs Interceptions, Clearances vs Blocks, Ground vs Aerial volume), post-standardization outlier check.
4. **PCA, 9 → 4 components.** Cumulative variance retained, biplot to inspect which raw metrics drive each component.
5. **Clustering.**
   * Ward's hierarchical clustering on the PCA space → dendrogram cut at t=10 produces 4 macro-groups separated by a large vertical gap (a visual stability signal).
   * **K-Means K=4** initialised from the hierarchical centroids (not random, not k-means++), so the assignment is deterministic and football-interpretable.
   * Validation via **Silhouette** and **Davies-Bouldin** scores at K=3, 4, 5. K=4 chosen as the trade-off between separation and tactical granularity.
6. **Cluster profiling & auto-naming.** Names assigned data-driven from centroid z-score signatures (no hand-labelling), robust to cluster-ID permutations across runs. Profiling reports raw PAdj averages vs the league for each cluster and a comparison radar.
7. **Player- and team-level outputs.**
   * **Soft clustering (membership %)**: each player gets a 4-vector of proximities to the centroids; the maximum determines whether they are a *Specialist*, *Hybrid* or *Allrounder*.
   * **Local Outlier Factor (LOF)** flags defenders who sit far from every cluster — genuine tactical outliers.
   * **Team Tactical DNA** aggregates individual cluster assignments back up to the team level weighted by minutes played, surfacing the team's defensive identity.

---

## 📊 Data Source
Raw defensive event data and metrics are powered by **Opta / Stats Perform** (accessed via TheAnalyst).

* **Competition:** Italian Serie A (2025/2026 Season).
* **Timestamp:** Data updated as of **end of season, May 2026**.
* **Dataset Scope:** Exclusively focused on Centre-Backs with **>1000 minutes played** (77 players). This strict filter ensures statistical reliability, preventing players with tiny sample sizes from heavily skewing the PAdj metrics and clustering algorithms.

---

## 📓 Notebook Preview

The full analysis pipeline lives in [`Clustering_defenders_SerieA.ipynb`](Clustering_defenders_SerieA.ipynb), an end-to-end notebook with executed outputs and inline interpretation of every chart. For a quick read without cloning the repo, an **HTML render** is also published: [`Clustering_defenders_SerieA.html`](Clustering_defenders_SerieA.html).

---

## 🔁 Stability Check: March vs End-of-Season

The same pipeline was run twice — on a mid-season snapshot (March 2026, ≥800', 68 players) and at end-of-season (May 2026, ≥1000', 77 players) — to check whether the cluster assignments hold up as more data arrives. An earlier snapshot is preserved in [`archive_march_2026/`](archive_march_2026/).

On the **67 players in common**, **63% kept the same cluster**. Two findings worth noting:

- **Aggressors are 100% stable** (17/17): Bastoni, Mancini, Akanji, Bisseck, Hien and others all stay Aggressors. This is the most robust archetype.
- **The Aerial Stopper ↔ Coverage Defender boundary is the most fluid** (~17 transitions). Football-meaningful: a defender who clears a lot in a deep block looks more "Aerial" with 800' of data and more "Coverage" with 2500', as volume per 90 settles.

63% is not stellar but is honest for K=4 on 77 points. The core archetypes are real; boundary cases need more data to settle, which is the *expected* behaviour, not a bug.

---

## ⚠️ Limitations & Future Work

**Limitations** of this iteration:

* **Sample size.** 77 players across 4 clusters (~19 each). Clusters are interpretable but boundary cases can shift with marginally different input data (see the [stability check](#-stability-check-march-vs-end-of-season) above).
* **Defensive scope only.** Build-up and on-ball metrics (progressive passes, carries, pass completion under pressure) are deliberately out of scope. A modern centre-back like Bastoni is not fully described by defensive volume alone.
* **Volume vs. quality.** PAdj corrects for possession but cannot fully separate *"low engagement because the team funnels play away"* from *"low engagement because the defender is not involved enough"*. The Coverage Defender label is intentionally neutral for this reason.
* **No temporal dynamics.** Stats are season-cumulative — a defender who changed role mid-season is averaged across both roles.
* **Similarity % is a ranking score, not an absolute one.** It is min-max normalized on the *Serie A* pairwise distance range, so values are meaningful within this dataset but not directly transferable across leagues.

**Future work** — natural next iterations:

1. **Multi-league expansion** to top-5 leagues, growing the sample and stabilising boundaries.
2. **Add a build-up dimension** alongside the defensive one, producing a 2-axis profile instead of a single label.
3. **Per-match data** instead of season cumulative, to detect role changes and form trends.

---

## 🚀 How to Run Locally
To run this application on your local machine:

```bash
# Clone the repository
git clone https://github.com/matteovezzoli/SerieA-DefensiveScouting-Engine.git
cd SerieA-DefensiveScouting-Engine

# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 👨‍💼 Author
**Matteo Vezzoli**

*Data Scientist | Sports Analytics*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/matteo-vezzoli83)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/matteovezzoli)
