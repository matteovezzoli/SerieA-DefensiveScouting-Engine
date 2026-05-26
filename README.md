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

| Profile | Color | Data Signature (vs. league avg.) | Football Interpretation |
|---|---|---|---|
| **Aggressor / Ball-Winner** | 🔵 | Tackles +27%, Interceptions +27%, Possessions Won +38%, Ground Duels engaged +30%, Aerial Duels engaged +22% | Centre-backs of dominant, high-line teams. Step up rather than drop, recover the ball forward. Typical of Roma, Inter, Napoli, Atalanta. |
| **Aerial Stopper** | 🔴 | Clearances +37%, Aerial Duels engaged +26%, Aerial win rate ~65% (+13%), Blocks +21% | Classic stay-at-home centre-back. Holds the line, absorbs pressure, dominates the box. Common in low/mid-block sides (Verona, Lecce, Como). |
| **Ground Specialist** | 🟢 | Ground Duels +10%, Tackles +3%, but Aerial Duels engaged -29% and Aerial win rate ~49% (-15%) | Strong on the ground but physically out-matched in the air. Fits as a wide centre-back in a back three or as a technical defender in possession-based sides. |
| **Coverage Defender** | 🟠 | Tackles -19%, Possessions Won -24%, Ground Duels -20%, Aerial Duels -12%, Blocks +14% | Two overlapping sub-profiles: fullbacks redeployed wide in a back three, and defenders in deep blocks where play is funneled away. Low engagement is a tactical signature, not necessarily a quality judgement. |

---

## 🖥️ The Frontend: Streamlit Dashboard
The `app.py` Streamlit dashboard is designed to make complex mathematical models accessible for football scouting. It reads the processed datasets and PCA coordinates to provide a dynamic, click-and-play interface divided into 5 sections:

* **🔍 Clone Finder:** Select a target player and use the Euclidean Distance similarity engine to find their top tactical replacements. Includes interactive radar chart comparisons.
* **🧬 Player DNA:** Explore a specific player's tactical membership (Soft Clustering percentages) to see their core playstyle.
* **🌍 Market Explorer:** Filter the dataset by tactical profile and minutes played to identify specific defender types.
* **📊 Metric Explorer:** Cross-analyze the 9 standardized PAdj metrics through dynamic scatter plots colored by tactical profile.
* **🛡️ Team DNA:** View how Serie A teams build their defensive lines, aggregated by the minutes played by each tactical profile (visualized via Donut and Stacked Bar charts).

---

## 🧠 Under the Hood: The Data Science Pipeline
The core engine driving the app is the `Clustering_defenders_SerieA.ipynb` notebook. The analysis strictly follows these methodological steps to ensure statistical validity:

* **Handling Team Bias (PAdj):** Applying Possession-Adjusted formulas to volume metrics to ensure fair comparisons between players in dominant teams vs. low-block teams.
* **Feature Scaling:** Z-Score Standardization to bring percentages and volume metrics onto the same mathematical scale.
* **Exploratory Data Analysis (EDA):**
  * Post-standardization Outlier Detection.
  * Correlation Matrix Heatmap to analyze mathematical relationships between defensive actions.
  * Scatterplot visualizations to map physical and tactical tendencies.
* **Dimensionality Reduction (PCA):** Compressing the 9 core defensive variables into a 4-dimensional tactical space.
* **Clustering & Profiling:**
  * **Hierarchical Clustering:** Using dendrograms to define the optimal number of groups (K=4).
  * **K-Means Clustering:** Applied on the PCA-reduced coordinates.
  * **Cluster Profiling & Comparison:** Translating mathematical clusters into the four football-meaningful profiles described above.
* **Advanced Player Segmentation:**
  * **Soft Clustering (Membership %):** Calculating a player's proximity to all centroids to define their Tactical Identity.
  * **Tactical Extremes:** Identifying pure Specialists vs. versatile Hybrids.
  * **Local Outlier Factor (LOF):** Detecting tactical anomalies and isolated player profiles.
* **Scouting Framework:**
  * **Player Similarity:** Using Euclidean Distance to find tactical clones.
  * **Team Tactical DNA:** Profiling the defensive systems of Serie A teams by weighting the clusters by minutes played.

---

## 📊 Data Source
Raw defensive event data and metrics are powered by **Opta / Stats Perform** (accessed via TheAnalyst).

* **Competition:** Italian Serie A (2025/2026 Season).
* **Timestamp:** Data updated as of **end of season, May 2026**.
* **Dataset Scope:** Exclusively focused on Centre-Backs with **>1000 minutes played** (77 players). This strict filter ensures statistical reliability, preventing players with tiny sample sizes from heavily skewing the PAdj metrics and clustering algorithms.

---

## 📓 Notebook Preview

The full analysis pipeline lives in [`Clustering_defenders_SerieA.ipynb`](Clustering_defenders_SerieA.ipynb), an end-to-end notebook with executed outputs and inline interpretation of every chart.

For a quick read without cloning the repo or running Jupyter, an **HTML render** is also published: [`Clustering_defenders_SerieA.html`](Clustering_defenders_SerieA.html).

An earlier mid-season snapshot of the same pipeline (March 2026 data, 68 players) is preserved in [`archive_march_2026/`](archive_march_2026/) for the stability comparison below.

---

## 🔁 Stability Check: March vs End-of-Season

A standard concern with K-Means on a small dataset is **stability**: would the same pipeline produce similar clusters if re-run with more data? To answer this, the analysis was performed twice:

| Run | Dataset | Mins threshold | N players |
|---|---|---|---|
| **March snapshot** | Mid-season (March 2026) | ≥800' | 68 |
| **End of season** | Full-season (May 2026) | ≥1000' | 77 |

**Result on the 67 players in common:**

| Outcome | Count | % |
|---|---|---|
| **Same cluster** in both runs | 42 | **62.7%** |
| Moved to a different cluster | 25 | 37.3% |

**Transition matrix** (rows = March profile, columns = End-of-season profile):

| March \ End | Aggressor | Aerial Stopper | Ground Specialist | Coverage Defender |
|---|---:|---:|---:|---:|
| **Aggressor** | **17** | 0 | 0 | 0 |
| **Aerial Stopper** | 1 | **9** | 0 | 10 |
| **Ground Specialist** | 0 | 1 | **2** | 5 |
| **Coverage Defender** | 1 | 0 | 7 | **14** |

### Reading the result

- **Aggressors are the most stable profile** (17/17 = 100%). Players like Mancini, Bastoni, Akanji, Bisseck, Hien stayed Aggressors with more data, confirming this is a robust archetype.
- **The Aerial Stopper ↔ Coverage Defender boundary is the most fluid** (10 + 7 = 17 transitions between these two). This is football-meaningful: a defender who clears the ball often in a deep block may look like an "Aerial Stopper" with 800' of data and a "Coverage Defender" with 2500', because the volume per 90 normalizes differently as games accumulate.
- **One notable promotion:** Bremer moved from *Aerial Stopper* (March) to *Aggressor* (end of season), consistent with his role evolution at Juventus through the year.

**Why this matters.** 62.7% stability is not stellar but it is **honest**: with 77 players across 4 clusters, the boundaries should shift somewhat as more data arrives. The core archetypes (especially Aggressor) are real and robust; the boundary cases need more data to settle. This is precisely the kind of caveat a portfolio project should expose, not hide.

---

## ⚠️ Limitations & Future Work

Transparency on what this engine does **not** capture is as important as what it does. Known limitations:

* **Sample size.** With 77 players across 4 clusters (~19 per cluster on average), the clustering boundaries are stable but not bulletproof (see the [stability check](#-stability-check--march-vs-end-of-season) above for an empirical measurement). A single player at the edge of two profiles could be re-assigned with marginally different input data.
* **Defensive scope only.** The engine intentionally ignores **build-up and on-ball metrics** (progressive passes, carries, pass completion under pressure). A modern centre-back like Bastoni is not fully captured by his defensive volume alone. This is a deliberate scope choice for this iteration, not an oversight.
* **Volume vs. quality.** PAdj normalises for possession, but cannot fully distinguish *"low engagement because the team funnels play away"* from *"low engagement because the defender is not involved enough"*. This is why **C3 is labelled neutrally as "Coverage Defender"** rather than something aspirational like "Recovery Specialist".
* **No temporal dynamics.** Stats are season-cumulative. A defender who changed role mid-season is averaged across both roles.
* **Similarity score calibration.** The Similarity % formula ($e^{-0.15 \cdot d}$) is designed for ranking, not for absolute comparison. The coefficient is chosen so that the median pairwise distance maps to a recognisable mid-range value; use the ordering, not the absolute number.

**Planned next iterations:**
1. **Multi-league expansion** (top 5 European leagues) to grow sample size 5-10x and stabilise cluster boundaries.
2. **Build-up dimension**: add a parallel set of on-ball metrics and produce a **2-axis profile** (defensive × possession) instead of a single defensive label.
3. **Per-match data** instead of season cumulative, to detect role changes and form trends throughout the season.
4. **Prospect predictor**: train a classifier on the labelled dataset to predict which tactical profile a Serie B or Primavera defender would fit if promoted to Serie A.

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
