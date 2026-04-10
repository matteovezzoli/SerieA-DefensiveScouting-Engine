# ⚽ Serie A Defensive Scouting Engine (25/26)
**A Data-Driven Pipeline and Web App for Tactical Profiling of Center-Backs.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## 📌 Project Overview
This project presents an analytical study of Serie A center-backs for the 2025/2026 season. The goal is to move beyond standard volume stats by grouping defenders based on their actual playing style. 

The project is structured in two connected parts: an interactive **Streamlit Web Application** serving as the frontend tool for scouting, powered by a rigorous **Jupyter Notebook** that handles the complete exploratory and machine learning pipeline.

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
* **Exploratory Data Analysis (EDA):** * Post-standardization Outlier Detection.
  * Correlation Matrix Heatmap to analyze mathematical relationships between defensive actions.
  * Scatterplot visualizations to map physical and tactical tendencies.
* **Dimensionality Reduction (PCA):** Compressing the 9 core defensive variables into a 4-dimensional tactical space.
* **Clustering & Profiling:**
  * **Hierarchical Clustering:** Using dendrograms to define the optimal number of groups (K=4).
  * **K-Means Clustering:** Applied on the previously calculated centroids.
  * **Cluster Profiling & Comparison:** Translating mathematical clusters into clear tactical definitions.
* **Advanced Player Segmentation:**
  * **Soft Clustering (Membership %):** Calculating a player's proximity to all centroids to define their Tactical Identity.
  * **Tactical Extremes:** Identifying pure Specialists vs. versatile Hybrids.
  * **Local Outlier Factor (LOF):** Detecting tactical anomalies and isolated player profiles.
* **Scouting Framework:**
  * **Player Similarity:** Using Euclidean Distance to find tactical clones.
  * **Team Tactical DNA:** Profiling the defensive systems of Serie A teams by weighting the clusters by minutes played.

---

## 📊 Data Source
Raw defensive event data and metrics are powered by **Opta / Stats Perform**. 

* **Competition:** Italian Serie A (2025/2026 Season).
* **Timestamp:** Data updated as of **March 2026**.
* **Dataset Scope:** Exclusively focused on Center-Backs with **>800 minutes played**. This strict filter ensures statistical reliability, preventing players with tiny sample sizes from heavily skewing the PAdj metrics and clustering algorithms.

## 🚀 How to Run Locally
To run this application on your local machine:

```bash
# Clone the repository
git clone [https://github.com/matteovezzoli/SerieA-Defensive-Scouting-Engine.git](https://github.com/matteovezzoli/SerieA-Defensive-Scouting-Engine.git)

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
