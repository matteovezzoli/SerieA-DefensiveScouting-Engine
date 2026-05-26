# Archive: March 2026 Snapshot

This folder contains the first iteration of the Serie A Defensive Scouting Engine, run on a **mid-season snapshot** of the 2025/2026 season (data as of March 2026, 68 centre-backs with ≥800 minutes played).

It is kept for reference and for the stability comparison documented in the main [README](../README.md#-stability-check-march-vs-end-of-season).

## Contents

| File | Description |
|---|---|
| `Clustering_defenders_SerieA.ipynb` | Original Jupyter notebook (mid-season pipeline). |
| `processed_defenders_data.csv` | March player table with cluster assignments and soft membership %. |
| `pca_coordinates.csv` | 4-dimensional PCA projection used by the similarity engine. |
| `radar_metrics.json` | List of z-score columns plotted in the radar charts. |

## Differences from the current iteration

| Aspect | March snapshot | Current (end of season) |
|---|---|---|
| Minutes threshold | ≥800 | ≥1000 |
| Players | 68 | 77 |
| Team possession values | Mid-season averages | Full-season averages |
| Cluster validation | Silhouette only | Silhouette + Davies-Bouldin (K=3,4,5) |
| Cluster naming | Hand-coded mapping | Auto-assigned from centroid signatures |
| Reproducibility | Partial | `random_state=42` throughout |

The cluster profiles (Aggressor / Aerial Stopper / Ground Specialist / Coverage Defender) are consistent across the two runs, but 25 of the 67 players in common changed cluster between March and May. The most stable profile is **Aggressor** (17/17 retained). The fluid boundary is **Aerial Stopper ↔ Coverage Defender**.
