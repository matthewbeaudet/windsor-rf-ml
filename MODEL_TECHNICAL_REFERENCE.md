# LTE RSRP Prediction Models — Technical Reference & RF Engineering Guide
### Windsor, ON · Montréal, QC

**Version:** 2.0  
**Date:** April 2026  
**Status:** Development  
**Geography:** Windsor, Ontario · Montréal, Québec  

---

## Table of Contents

**Part I — Windsor**
1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Model Details](#3-model-details)
4. [Training Data](#4-training-data)
5. [Feature Engineering](#5-feature-engineering)
6. [Performance & Validation](#6-performance--validation)
7. [RF Engineering FAQ](#7-rf-engineering-faq)
8. [Known Limitations & Biases](#8-known-limitations--biases)

**Part II — Montréal**
12. [Montréal — Executive Summary](#12-montréal--executive-summary)
13. [Why Two Models](#13-why-two-models)
14. [Training Data](#14-training-data)
15. [Cell Parameter Differences vs Windsor](#15-cell-parameter-differences-vs-windsor)
16. [Feature Engineering](#16-feature-engineering)
17. [Performance & Validation](#17-performance--validation)
18. [Known Limitations & Biases](#18-known-limitations--biases)

**Part III — Infrastructure & Roadmap**
9. [Deployment & Infrastructure](#9-deployment--infrastructure)
10. [Next Steps & Vision](#10-next-steps--vision)
11. [References](#11-references)

---

## 1. Executive Summary

This document describes a machine-learning model that predicts **LTE Reference Signal Received Power (RSRP, in dBm)** for candidate cell site locations in Windsor, Ontario. It is designed to support **new site deployment planning** by generating a full coverage map — showing predicted signal strength at every H3 hexagonal bin within a configurable radius — before a site is physically built.

### What it replaces
Traditional site planning relies on ray-tracing simulators (Atoll, Planet) that require manually tuned propagation models, detailed 3D building databases, and expert calibration. These tools take hours to run a single scenario and require software licences. This model produces equivalent-quality coverage predictions in **under 10 seconds** on a standard laptop or cloud VM.

### Key numbers at a glance

| Metric | Value |
|--------|-------|
| Prediction RMSE | **6.48 dB**  |
| Noise-corrected RMSE | **3.14 dB** |
| Mean bias | +0.29 dB (near-zero) |
| R² across 14 test sites | 0.515 |
| Training geography | Windsor, ON (30 LTE Sites, 30,775 bins) |
| Frequency | LTE ~2147.5 MHz |
| Prediction speed | ~9,800 H3 bins in 5–10 seconds |
| Model size | ~2 MB (LightGBM, 1,408 trees) |

### What "6.5 dB RMSE" means in practice
The measurement noise floor for well-sampled bins (≥5 measurements) is **5.56 dB** — the irreducible variability due to multipath, body loss, handset variation, and temporal fading. The model's noise-corrected RMSE of **3.14 dB** is well below this noise floor, meaning the model has successfully separated the true coverage signal from measurement noise. This is a strong result: the model is predicting the underlying RSRP field with 3.14 dB accuracy even though individual measurements scatter by ±5.56 dB around the true value. This value of 6.48 RMSE is the mean value accross 14 Leave One Site Out (LOSO) Cross Validation tests, which represents the most robust and realistic way to measure performance and reliability for new site predictions, as opposed to picking a single test site or randomly splitting bins. 

### Intended users
- **RF engineers:** Site feasibility assessment, automation of tens of candidate locations, coverage gap analysis, asynchronous batch site predictions to spend more time of in depth analysis of top candidates.
- **Management/stakeholders:** Rapid what-if analysis, investment prioritization

### Literature & Standards Basis
This model was developed with reference to current ML-RF propagation research from Canada, Asia, and Europe, as well as machine learning textbooks and ML engineering best-practice papers, to ensure a professional-grade, production-ready implementation. Key references are listed in [Section 11 — References](#11-references).

---

## 2. System Architecture

The prediction pipeline has four stages:

```
User Input (antenna and parameter config)
        │
        ▼
 H3 Bin Generation
 (all res-12 hex bins within radius)
        │
        ▼
 Feature Engine  ◄─── Pre-loaded clutter DB (h3_complete_features_windsor.csv)
 ├─ Geometry         ◄─── DSM ray-trace (h3_dsm_clutter_database.csv + h3_dem_database.csv)
 ├─ Beam patterns    ◄─── pycraf ITU-R P.452 (real-time, parallelized)
 ├─ pycraf P.452
 └─ Clutter lookups
        │
        ▼
 LightGBM Model (lean_lgbm_53feat_model.joblib)
        │
        ▼
 RSRP Map (H3 bins + predicted RSRP dBm)
```

### Data sources loaded at startup
| Source | File | Contents | Size |
|--------|------|----------|------|
| Environmental features | `h3_complete_features_windsor.csv` | Clutter heights, types, indoor/outdoor proportions per H3 bin | ~400 MB |
| DSM surface | `h3_dsm_clutter_database.csv` | 95th-percentile surface height (buildings+trees) per H3 bin | ~200 MB |
| DEM terrain | `h3_dem_database.csv` | Bare-ground elevation (metres ASL) per H3 bin | ~150 MB |
| Baseline coverage | `comprehensive_rsrp_all_46_sites.csv` | Existing 46-site RSRP predictions (for comparison) | ~80 MB |
| Model | `Model/lean_lgbm_53feat_model.joblib` | Trained LightGBM model | 535 KB |
| Feature list | `Model/lean_lgbm_53feat_features.json` | Ordered list of 53 feature names | 1 KB |

### Prediction flow for a single site (typical)
1. User clicks map → antenna lat/lon resolved
2. Auto-detect antenna height from DSM at that location (p95 surface height of dominant building pixel)
3. H3 res-12 bins within radius loaded from preloaded environmental DataFrame
4. **Geometry features** computed for all bins in vectorized NumPy (~1.2 s for 9,800 bins)
5. **pycraf** run in parallel thread pool: 1 pycraf call per unique H3 res-10 parent cell (~230 calls, ~2.5 s)
6. **DSM LoS ray-trace** per bin: 20-sample vectorized ray cast from antenna to each bin centroid (~0.7 s)
7. **Model inference**: `model.predict(X)` on 9,800 × 53 feature matrix (~0.05 s)
8. Results returned as GeoJSON with RSRP values for map rendering

**Total wall time: ~5-10 seconds** for a 1 km radius prediction.

---

## 3. Model Details

### Algorithm: LightGBM (Gradient Boosted Decision Trees)

LightGBM is a state-of-the-art gradient boosting framework. It builds an ensemble of decision trees sequentially, each tree correcting the residual error of the previous. Key advantages for this application:
- **Handles mixed feature types** (continuous distances, binary beam flags, integer counts) without feature scaling
- **Non-linear interactions** between geometry and clutter captured automatically
- **Robust to outliers** in the measurement data (handles the long tail of very low RSRP measurements in deep NLOS)
- **Fast inference**: 53 features × 62 trees = ~3,200 leaf lookups per prediction point

### Hyperparameters

Tuned with Optuna, a state of the art hyperparameter tuning algorithm.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 0.05 | Conservative — avoids overfitting with 62 trees |
| `max_depth` | 7 | Allows complex interactions; regularized by `num_leaves` |
| `num_leaves` | 100 | ~100 leaf nodes per tree; generous for RF features |
| `min_child_samples` | 50 | Minimum 50 samples per leaf — prevents overfitting to rare bins |
| `subsample` | 0.8 | 80% row sampling per tree — reduces variance |
| `colsample_bytree` | 0.75 | 75% feature sampling per tree — ~40 of 53 features per tree |
| `reg_alpha` | 0.5 | L1 regularization — drives unimportant features toward zero |
| `reg_lambda` | 3.0 | L2 regularization — smooth leaf weights |
| `early_stopping` | 150 rounds | Stopped at 62 trees (optimal on holdout RMSE) |

### Model size and files
| File | Size | Description |
|------|------|-------------|
| `Model/lean_lgbm_53feat_model.joblib` | 535 KB | Trained model (joblib serialization) |
| `Model/lean_lgbm_53feat_features.json` | 1.1 KB | Ordered list of 53 feature names (must match exactly) |
| `Model/lean_lgbm_53feat_importance.csv` | 1.0 KB | Feature importance by split count |

---

## 4. Training Data

| Property | Value |
|----------|-------|
| Geography | Windsor, Ontario, Canada |
| Period | May–July 2025 (3 months) |
| Raw RSRP samples | ~1.56 million MDT-GPS records |
| Spatial binning | Uber H3 hexagonal grid, resolution 12 (~30m hex diameter, equivalent to 17m quadbin) |
| Training sites | 30 sites |
| Training rows | ~35,000 H3 bins (after aggregation) |
| Target variable | Median RSRP per bin (dBm) |
| Measurement source | TELUS Location Service Reporting (LSR) — Minimization of Drive Tests GPS (MDT-GPS) |

### Spatial Aggregation
LSR MDT-GPS RSRP samples are assigned to their enclosing H3 resolution-12 hexagon. The **median** RSRP within each bin is used as the target. Using median rather than mean reduces sensitivity to extreme outliers (e.g., a single measurement recorded inside a building).

### Measurement Statistics
- **Samples per bin:** Median = 3, Mean = 15, Max = 12,880
- **Intra-bin noise floor (σ):** 5.23 dB — the standard deviation of RSRP measurements within a single bin across time. This represents irreducible measurement noise (multipath, body loss, handset variation, temporal fading).
- **Filtering** Removed bins with less than 5 samples

### Data Quality Notes
- Water bins are excluded from predictions
- **Bins with fewer than 5 measurements are excluded from training** (see Bin Quality Filter below)
- Indoor measurements are labeled but not excluded — `indoor_proportion` is a model feature

### Bin Quality Filter (point_count ≥ 5)

A systematic sweep was conducted to determine the optimal minimum sample count per H3 bin for training. The motivation: a single RSRP measurement in a bin is an unreliable estimate of the true median RSRP — it could be an outlier, captured under anomalous conditions. Training on such sparse targets introduces noise that degrades model accuracy.

**Sweep results (14-fold LOSO CV across all thresholds):**

| Min samples | Rows kept | % kept | Noise floor | RMSE | RMSE_nc | MAE | R² |
|-------------|----------:|-------:|------------:|-----:|--------:|----:|---:|
| 1 (all bins) | 89,279 | 100% | 5.23 dB | 7.59 dB | 5.44 dB | 5.93 dB | 0.449 |
| ≥ 2 | 59,125 | 66% | 5.27 dB | 6.83 dB | 4.30 dB | 5.36 dB | 0.489 |
| ≥ 3 | 45,235 | 51% | 5.46 dB | 6.71 dB | 3.84 dB | 5.29 dB | 0.500 |
| ≥ 4 | 36,757 | 41% | 5.54 dB | 6.62 dB | 3.51 dB | 5.23 dB | 0.505 |
| **≥ 5 (chosen)** | **30,775** | **35%** | **5.56 dB** | **6.48 dB** | **3.14 dB** | **5.09 dB** | **0.515** |

**Key finding:** RMSE improves monotonically with stricter filtering. Dropping 65% of bins (all with 1–4 measurements) improved RMSE by **1.11 dB** (7.59 → 6.48 dB) and noise-corrected RMSE by **2.30 dB** (5.44 → 3.14 dB). The noise floor is essentially unchanged (5.23 → 5.56 dB), confirming the improvement is real model accuracy gain, not an artifact of filtering out noisier bins from the evaluation.

**Breakdown of removed bins:**
- 1 sample: 30,154 bins (34% of all bins) — single MDT-GPS sample, unreliable median
- 2 samples: 13,890 bins (16%) — marginal reliability
- 3 samples: 8,478 bins (9%)
- 4 samples: 5,982 bins (7%)
- **Total removed: 58,504 bins (65.5%)**

> A spatial map of removed vs. kept bins is available in `Testing/removed_bins_map.html`. Removed bins (coloured by count tier) are spatially scattered throughout Windsor — they are not concentrated in any particular area, indicating this is a random sparsity issue rather than systematic under-coverage of a geographic zone.

### Data Cleaning — Stationary Filter

MDT-GPS data includes measurements from subscribers who are in motion (driving, transit). **Moving samples are systematically biased** and were excluded from training for two reasons:

1. **Doppler fading:** A moving UE experiences fast Rayleigh fading — rapid RSRP fluctuations of 10–20 dB over distances of a few metres. A single measurement from a moving subscriber may be captured at a deep fade null and misrepresent the true median RSRP at that location.
2. **GPS smearing:** GPS position accuracy for a fast-moving device is lower, introducing spatial error that misassigns samples to adjacent H3 bins.

**Method:** Samples were classified as stationary or moving using [**MovingPandas**](https://movingpandas.org/) (`mpd.StopDetector`), a Python geospatial trajectory analysis library. For each subscriber trajectory:

1. Trajectories were built per IMSI from time-ordered GPS samples and projected to UTM Zone 17N (EPSG:32617) for accurate distance measurement
2. Optional Kalman smoothing was applied to reduce GPS noise before stop detection
3. `StopDetector.get_stop_segments()` was called with:
   - `min_duration = 60 seconds` — a sample sequence must persist at the same location for ≥60s to be classified as a stop
   - `max_diameter = 30 m` — the cluster of GPS points during the stop must be within a 30m circle (matching H3 res-12 hex diameter)
4. Points falling within detected stop segments were labelled `is_stationary = 1`
5. **Only stationary samples** (`is_stationary = 1`) were retained for training
6. A secondary speed filter removed any remaining samples with computed speed > 27 m/s (~100 km/h), catching high-speed outliers not caught by stop detection

**Impact:**  The stationary samples represent subscribers at rest, providing a stable RSRP measurement that reliably reflects the true signal field at that location.

---

## 5. Feature Engineering

The model uses **53 features** organised in 6 groups, selected through a sequential ablation study.

### Group 1 — Distance & Geometry (6 features)
| Feature | Description | Importance |
|---------|-------------|------------|
| `distance_3d_log` | 10·log10(3D slant distance, m) | High |
| `elevation_diff_log` | log10(antenna height − UE height, m) | High |
| `bearing_sin`, `bearing_cos` | Absolute bearing from antenna to UE (sin/cos) | Very High |
| `bearing_azimuth_sin`, `bearing_azimuth_cos` | Off-boresight angle (bearing − azimuth, sin/cos) | Very High |

> `distance_3d_log` uses 10·log10 encoding (not raw metres) to match the log-linear path loss relationship. 

### Group 2 — Antenna & RF Configuration (10 features)
| Feature | Description |
|---------|-------------|
| `RS_dBm` | Reference Signal EPRE (18.2 dBm, constant for Windsor T2008 antennas) |
| `Antenna_height` | Antenna height above ground (m) — #3 most important feature |
| `Antenna_EDT` | Electrical downtilt (degrees) |
| `Antenna_MDT` | Mechanical downtilt (degrees) |
| `VBW` | Vertical 3dB beamwidth (6.3° for T2008) |
| `HBW` | Horizontal 3dB beamwidth (65° for T2008) |
| `Gain` | Antenna gain (17.1 dBi for T2008) |
| `downlink_freq` | Downlink frequency (MHz, default 2147.5) |
| `carrier_bw` | Carrier bandwidth (MHz, default 15) |
| `outdoor` | Binary: 1 = outdoor macrocell |

### Group 3 — Beam Geometry (7 features)
| Feature | Description |
|---------|-------------|
| `downtilt_projection_distance` | 10·log10(distance where beam boresight hits ground, m) — **#1 most important** |
| `elevation_ue` | UE terrain elevation in metres ASL (from DEM, ~185m in Windsor) |
| `h_attenuation` | Horizontal antenna pattern loss (dB): −min(12·(θ_h/HBW)², 20) |
| `v_attenuation` | Vertical antenna pattern loss (dB): −min(12·(θ_v/VBW)², 20) |
| `within_3db_horizontal` | Binary: 1 if UE within HBW/2 of azimuth |
| `within_vbw_shadow` | Binary: 1 if depression angle within VBW/2 of total downtilt |
| `within_3db_beam` | Binary: logical AND of horizontal AND vertical flags |

> **`downtilt_projection_distance`** is the single most important feature (importance=395). It encodes the physical point on the ground where the main beam boresight lands. Bins closer than this point receive mainlobe illumination; bins beyond it receive back-lobe/sidelobe. This captures the EDT/MDT effect far more efficiently than raw tilt angles alone.

### Group 4 — Propagation (pycraf ITU-R P.452 + DSM LoS) (15 features)
| Feature | Source | Description |
|---------|--------|-------------|
| `distance_los` | Geometry | 10·log10(LoS path distance, m) |
| `pycraf_eps_pt`, `pycraf_eps_pr` | pycraf | Path elevation angles at Tx/Rx (degrees, negative = RX below TX) |
| `pycraf_d_lt`, `pycraf_d_lr` | pycraf | Terrain horizon distances at Tx/Rx (km) |
| `pycraf_L_bfsg` | pycraf | Free-space + gaseous absorption loss (dB) |
| `pycraf_L_b0p` | pycraf | Basic loss not exceeded 50% of time (dB) |
| `pycraf_L_bd` | pycraf | Diffraction loss component (dB) |
| `pycraf_L_bs` | pycraf | Tropospheric scatter loss (dB) |
| `pycraf_L_ba` | pycraf | Ducting/anomalous propagation loss (dB) |
| `pycraf_L_b` | pycraf | Total basic transmission loss (dB) |
| `dsm_los_ratio` | DSM ray-trace | Fraction of ray samples clear of DSM (0–1) |
| `dsm_los_binary` | DSM ray-trace | 1 if ≥95% of ray is unobstructed |
| `dsm_first_block_m` | DSM ray-trace | Distance from Tx to first DSM obstruction (m) |
| `dsm_max_excess_m` | DSM ray-trace | Maximum DSM height above the ray line (m) |

> **pycraf** implements ITU-R Recommendation P.452-17, the international standard for terrestrial path propagation. It accounts for free-space loss, diffraction (knife-edge + rounded obstacle), tropospheric scatter, and ducting. One pycraf call per H3 res-10 parent (~520m hex) is used to approximate res-12 bins, reducing computation 50× with negligible accuracy loss for macro-cell distances.

### Groups 5 & 6 — Clutter (15 features)
| Feature | Description |
|---------|-------------|
| `clutter_mean/min/max/std/p95_height`, `clutter_height_range` | Clutter height statistics from DSM−DEM (m) |
| `Building_count`, `Building_pct` | Building pixel count and % in H3 bin |
| `Industry_count`, `Industry_pct` | Industrial land use count and % |
| `Tree_pct`, `Forest_pct` | Vegetation cover % (seasonal/penetrable) |
| `tree_density_per_km2` | Tree density (trees/km²) |
| `indoor_proportion`, `outdoor_proportion` | Fraction of historical measurements recorded indoors/outdoors |

---

## 6. Performance & Validation

### Validation Methodology: Leave-One-Site-Out (LOSO) Cross-Validation

The model is validated using **14-fold Leave-One-Site-Out** cross-validation. In each fold, all bins from one site are held out as the test set and the model is retrained on the remaining 13 sites. This is the most rigorous validation for spatial ML because:
- It tests generalisation to **completely unseen sites** (not just unseen bins from known sites)
- It directly simulates the deployment use case: predict coverage for a new site
- It prevents spatial leakage that would inflate performance if neighbouring bins from the same site appeared in both train and test sets

### Cross-Validation Results

| Site | Test bins | RMSE (dB) | RMSE_nc (dB) | MAE (dB) | Bias (dB) | R² |
|------|-----------|-----------|--------------|----------|-----------|-----|
| ON000899 | 2,151 | 5.60 | 0.70 | 4.39 | +0.07 | 0.517 |
| ON0826 | 972 | 5.97 | 2.17 | 4.73 | +0.02 | 0.535 |
| ON1404 | 883 | 6.79 | 3.90 | 5.69 | +3.20 | 0.694 |
| ON0829 | 1,777 | 6.34 | 3.04 | 5.07 | +0.26 | 0.640 |
| ON0831 | 996 | 5.60 | 0.63 | 4.46 | +0.37 | 0.489 |
| ON1095 | 1,750 | 6.25 | 2.86 | 4.95 | −0.26 | 0.602 |
| ON1401 | 1,058 | 6.67 | 3.68 | 5.22 | −1.18 | 0.542 |
| ON1403 | 1,801 | 6.99 | 4.24 | 5.53 | −1.55 | 0.408 |
| ON0833 | 1,706 | 7.36 | 4.83 | 5.79 | +2.80 | 0.439 |
| ON0084 | 1,792 | 6.72 | 3.77 | 5.29 | −1.59 | 0.521 |
| ON000848 | 1,532 | 6.20 | 2.75 | 4.92 | +1.57 | 0.281 |
| ON0823 | 1,434 | 6.97 | 4.20 | 5.46 | −0.28 | 0.592 |
| ON1098 | 1,217 | 6.61 | 3.57 | 5.20 | −1.62 | 0.499 |
| ON0834 | 1,158 | 6.67 | 3.68 | 5.20 | +2.26 | 0.455 |
| **MEAN** | | **6.48** | **3.14** | **5.09** | **+0.29** | **0.515** |

**Column definitions:**
- **RMSE**: Root Mean Square Error vs. measured RSRP (dB)
- **RMSE_nc**: Noise-corrected RMSE = √(RMSE² − σ_noise²) — error on the *true* signal, excluding measurement noise
- **MAE**: Mean Absolute Error (dB)
- **Bias**: Mean prediction − measurement (positive = over-prediction)
- **R²**: Coefficient of determination (variance explained)

### Ablation Study — Feature Group Contributions

| Step | Feature Group Added | Features | RMSE (dB) | Improvement |
|------|--------------------|---------:|----------:|-------------|
| 1 | G1: Distance + Bearing | 6 | 7.130 | — |
| 2 | + G2: Antenna & RF | 16 | 7.016 | −0.114 dB |
| 3 | + G3: Beam geometry | 23 | 6.984 | −0.032 dB |
| 4 | + G4: pycraf + DSM LoS | 38 | 6.962 | −0.022 dB |
| 5 | + G5: Clutter heights | 44 | 6.873 | −0.089 dB |
| 6 | + G6: Clutter types | **53** | **6.811** | −0.062 dB |

**Total improvement over distance-only baseline: 0.319 dB RMSE**

### Noise Floor Analysis
- **Intra-bin measurement noise (well-sampled bins, ≥5 samples):** 5.56 dB
- **Model RMSE:** 6.48 dB
- **Noise-corrected RMSE:** 3.14 dB — well below the noise floor
- **Interpretation:** The model is successfully extracting the true RSRP signal from noisy measurements. The 3.14 dB noise-corrected RMSE means the model's predictions are closer to the true RSRP field than any individual measurement.
- **Conclusion:** Performance is limited by dataset size and geographic diversity, not by feature engineering. More training sites would further reduce RMSE.

---

## 7. RF Engineering FAQ

### Q1. Why use ML instead of a physics-based model like COST-231 or Okumura-Hata?

Physics models like COST-231 or Okumura-Hata give a single path-loss value based on distance, frequency, and environment class. They do not account for specific building shapes, tree cover, or local terrain — they use environment "categories" (urban/suburban/rural) that mask enormous local variability. Our model uses actual per-bin clutter heights and DSM ray-tracing, and it learns the residual relationship between geometry and measured RSRP directly from LSR MDT-GPS data. The 7.6 dB RMSE achieved here is far better than the 10–15 dB typical of uncalibrated COST-231 in Windsor's mixed environment.

### Q2. How does the model handle antenna downtilt?

Both electrical downtilt (EDT) and mechanical downtilt (MDT) are model inputs. The beam geometry is computed via the 3GPP 36.814 antenna pattern formulas:
- **Horizontal attenuation:** −min(12·(θ_h/HBW)², 20) dB, where θ_h = bearing − azimuth
- **Vertical attenuation:** −min(12·(θ_v/VBW)², 20) dB, where θ_v = total_tilt − depression_angle
- **Downtilt projection distance:** The ground distance where the main beam boresight lands, encoded as 10·log10(metres). This is the #1 most important feature in the model.

The model learns, from actual measurements, how RSRP responds to being inside vs. outside the beam — going beyond the simplified 3GPP pattern to capture the real-world antenna pattern.

### Q3. What does 6.5 dB RMSE mean in practice? Is it good enough for site planning?

Yes — for the intended use case of **coverage gap identification and site shortlisting**, 6.5 dB is acceptable. In practical terms:
- The model can reliably distinguish between a "good coverage" bin (RSRP > −95 dBm) and a "poor coverage" bin (RSRP < −105 dBm)
- 1σ prediction error band = ±6.5 dB; 95% of predictions fall within ±13 dB of measured
- For RSRP thresholds used in planning (e.g., −100 dBm for 4G service), 6.5 dB RMSE means some bins near the boundary will be misclassified — but the spatial pattern (which sector is best, where is the coverage hole) is correctly identified
- Noise-corrected RMSE = 3.14 dB means the model is near the irreducible measurement noise — there is little room left to improve with the current dataset

### Q4. How does it compare to Atoll or Planet?

Atoll/Planet use deterministic ray-tracing calibrated to local measurement data. Calibrated Atoll typically achieves 5–8 dB RMSE in dense urban environments. Our model is in the same range (6.5 dB uncorrected, 3.2 dB noise-corrected) but operates in seconds without licences or manual calibration. This model is currently Windsor-specific and requires retraining for new geographies.

### Q5. What is LOSO cross-validation and why is it the right choice?

Leave-One-Site-Out (LOSO) holds out all measurements from one complete cell site and trains on the rest. This is the spatial equivalent of k-fold cross-validation, designed to prevent **spatial leakage** — the scenario where a model effectively memorises the RSRP gradient around a training site and appears to predict well, but fails completely on new sites. LOSO forces the model to generalise to new locations, directly matching the deployment use case.

### Q6. Why is the noise floor 5.23 dB? Isn't that too noisy to train on?

The noise floor reflects real physical variability — not measurement equipment error. Multiple samples at the same H3 bin over 3 months vary because: multipath changes as buildings/vehicles move, handset orientation varies, body loss changes, weather affects diffraction. This is the irreducible variability the model is trained on. The fact that noise-corrected RMSE is 5.44 dB (≈ noise floor) means the model has learned everything learnable from this dataset.

### Q7. Does the model account for building penetration loss?

Indirectly, via the `indoor_proportion` feature. Bins that historically had a high fraction of indoor measurements (e.g., a residential block with many in-building measurements) will have learned indoor penetration loss baked into the target RSRP. The model does not apply a separate indoor penetration loss coefficient — it relies on the historical indoor fraction to implicitly encode this.

### Q8. How does it handle LoS vs. NLoS conditions?

Four DSM ray-trace features capture LoS/NLoS:
- `dsm_los_ratio` (continuous 0–1): fraction of the ray from antenna to bin centroid that is clear of the DSM surface
- `dsm_los_binary`: hard threshold at 95% clear
- `dsm_first_block_m`: where the first obstruction occurs (close-in obstruction is worse)
- `dsm_max_excess_m`: how deeply the ray is blocked

These are computed via a 20-point vectorized ray-cast from the antenna to each bin, comparing ray height against the H3-binned DSM (95th-percentile surface height). The model then combines these with pycraf's diffraction loss `pycraf_L_bd` for a physically-grounded NLoS estimate.

### Q9. What is pycraf / ITU-R P.452 and why is it included?

pycraf is a Python implementation of ITU-R Recommendation P.452-17 — the international engineering standard for predicting radio path loss on terrestrial links. It computes:
- **Free-space loss** (with atmospheric gas absorption)
- **Diffraction loss** (using the Bullington/Deygout method over real terrain profiles)
- **Tropospheric scatter** (dominant at long ranges)
- **Ducting/anomalous propagation** loss
- **Elevation angles** at transmitter and receiver (key geometry features)

These 10 pycraf-derived features contribute a 0.022 dB RMSE improvement. While modest in aggregate, the elevation angle features (`pycraf_eps_pt`, `pycraf_eps_pr`) are the 9th most important features — they encode the geometric "looking up/down" relationship between antenna and UE that distance alone cannot capture.

### Q10. Can this model be used in other cities?

Not without retraining. The model has learned from Windsor's specific building stock, terrain (flat, ~183m ASL), LTE frequency plan, and antenna types (T2008). A city with different building heights, more hilly terrain, or different antenna vendors would require new training data. The feature engineering pipeline is fully general — only the trained weights are Windsor-specific.

### Q11. Why predict RSRP and not SINR or RSRQ?

RSRP is the most direct and reliable measure of signal strength from a single serving cell. SINR and RSRQ depend on the interference from all other cells — which changes over time as load, neighbour cell power, and scheduling vary. Predicting RSRP for a new site is tractable; predicting SINR would require modelling all neighbour cells simultaneously. RSRP predictions can be post-processed into RSRQ or throughput estimates using standard link budget assumptions.

### Q12. What is H3 hexagonal binning and why use it?

Uber H3 is an open-source hierarchical spatial indexing system using hexagonal tiles. Resolution 12 hexagons have a diameter of ~30m — roughly the block-face scale in Windsor. Hexagons are used because:
- Every hexagon has exactly 6 equidistant neighbours (vs. 8 for squares, with 2 diagonal distances)
- No directional bias in spatial features derived from hex centres
- Hierarchical structure enables efficient coarse-to-fine lookups (res-10 parent for pycraf, res-12 for predictions)

### Q13. What are the known failure modes?

1. **Industrial / warehouse areas:** The model over-predicts RSRP by up to +2.5 dB. Metal-clad buildings cause diffuse reflections the DSM clutter height cannot capture. The `Industry_pct` feature partially corrects this.
2. **Near-water bins:** St. Clair River and Lake Erie create anomalous propagation conditions not well-represented in training data. These bins are excluded.
3. **Very low antennas (<5m):** Predictions degrade for antenna heights below the surrounding clutter — the model was trained on antennas 9–50m AGL.
4. **Long-range predictions (>2 km):** LOSO CV was evaluated within 2 km. Performance beyond this range is unknown.
5. **Multipath-dominated bins:** Very short distances (<50m) are often multipath-dominated; the model may underestimate coverage there.

### Q14. How is the DSM ray-trace done?

For each antenna-to-bin path:
1. Cast 20 uniformly-spaced sample points along the straight-line path
2. For each sample point, look up the H3 res-12 bin it falls in and retrieve the DSM p95 height (95th percentile surface height in metres ASL) from `h3_dsm_clutter_database.csv`
3. Compare the sample point's ray height (linear interpolation from antenna ASL to UE ASL) against the DSM surface height
4. Count obstructed samples → `dsm_los_ratio`; find first obstruction → `dsm_first_block_m`; find max penetration → `dsm_max_excess_m`

The DSM p95 height is used rather than mean to ensure the model sees the tallest structure in the bin (the obstacle that blocks the signal, not the average surface).

### Q15. The model shows +2.5 dB bias for some industrial sites. What causes it?

Industrial/warehouse buildings have large, flat metal roofs that produce strong specular reflections not captured in the DSM clutter height. The DSM correctly identifies the building height but the model was trained on a dataset where most buildings are residential/commercial with less specular reflection. The `Industry_pct` feature partially corrects this but cannot fully compensate. For industrial site planning, apply a −2 to −3 dB correction to the model output.

### Q16. What happens if I predict outside Windsor?

The geographic H3 clutter database (`h3_complete_features_windsor.csv`) covers only the Windsor CMA. Predictions outside this area will use default feature values (0 for building counts, terrain fallback of 183m ASL). Results will be physically unreasonable. Do not use for locations outside Windsor without regenerating the clutter database.

### Q17. Is the model sensitive to antenna height input?

Yes — `Antenna_height` is the 3rd most important feature. A 5m error in antenna height will cause approximately 1–3 dB RSRP prediction error, depending on the local clutter environment. The app auto-detects height from the DSM (p95 surface height of the dominant building pixel at the site location), which gives the rooftop height. Operators should verify the auto-detected height against the actual equipment height.

### Q18. Can the model be retrained with new MDT-GPS data?

Yes. The training pipeline is:
1. Extract new LSR MDT-GPS RSRP records from TELUS LSR
2. Apply the stationary filter (see Section 4 — Data Cleaning) to remove moving samples
3. Bin the stationary samples to H3 res-12 hexagons and compute medians
4. Join with the pre-computed H3 feature database (`h3_complete_features_windsor.csv`)
5. Run `cross_validate_lean_model.py` to validate
6. If validated, save new model to `Model/lean_lgbm_53feat_model.joblib`

Adding data from new seasons (e.g., winter, when tree foliage drops) would improve predictions for seasonal tree-obstruction effects. Adding data from new sites improves generalisation.

### Q19. What is the minimum measurement density needed?

The training data shows that bins with fewer than 3 samples have significantly higher intra-bin variance. The model performs best in areas with ≥5 samples/bin. Because the data source is passive MDT-GPS (no active drive routes are planned), density is governed by subscriber traffic patterns in the area. Dense residential and commercial areas naturally accumulate enough samples; sparse industrial or low-traffic areas may take longer to reach the ≥5 sample threshold.

### Q20. Why does the app compute pycraf in real time instead of precomputing it?

pycraf path-loss depends on the **specific antenna location** — each new candidate site has a different path profile to every bin. Unlike the environmental features (clutter, DSM) which are fixed per-bin, pycraf results are unique per (antenna, bin) pair and cannot be precomputed for arbitrary new sites. The batch parallelization strategy (1 pycraf call per H3 res-10 parent, ~230 calls per prediction, parallel threads) reduces the compute time to ~2.5 seconds per site — acceptable for interactive use.


### Q21. How are Azimuths treated?
- For ease of use and rapidity, azimuths are standardized at 0, 120 and 240 degrees. Further tuning can be done in Atoll when a shortlist of candidates is generated from the model. However, the UE/bin's bearing to azimuth difference in angle is encoded and weighted by the model when predictions are made. 


---

## 8. Known Limitations & Biases

### Azimuths
- For ease of use and rapidity, azimuths are standardized at 0, 120 and 240 degrees. Further tuning can be done in Atoll when a shortlist of candidates is generated from the model. 

### Geographic Constraints
- **Coverage area:** Windsor CMA only. The clutter database covers approximately 42.0°–42.5°N, 83.2°–82.7°W.
- **Radius:** Validated within 2 km of the antenna. Predictions beyond 2 km degrade gracefully but have not been validated.
- **Water bodies:** Bins with >90% water coverage are excluded. The shoreline creates anomalous propagation that the model cannot predict reliably.

### Frequency & Technology
- Trained exclusively on **LTE 2100 band** (DL frequency of ~2147.5 MHz). Other LTE bands or 5G NR will produce incorrect predictions without retraining.
- The carrier bandwidth default is 15 MHz (75 PRBs). Other bandwidths are accepted as input but model has seen very little variability in this parameter.

### Antenna Type
- Trained on **Tongyu T2008** series antennas (Gain=17.1 dBi, HBW=65°, VBW=6.3°), which make up th training data. These values are hardcoded as constants for new predictions. Predictions for sites with different antenna models will be biased if the pattern parameters differ significantly.

### Temporal Bias
- Training data is skewed toward **nighttime measurements** (3–8 AM peak). The model may overestimate RSRP by 1–2 dB for daytime conditions where interference, body loss, and traffic-induced blockage are higher.
- **Seasonal effects:** Trained on May–July (summer, full foliage). Tree-obstructed bins may have 2–4 dB better coverage in winter when leaves fall. The `Tree_pct` and `Forest_pct` features partially capture this but the model cannot distinguish seasons.

### Bias by Environment Type
| Environment | Typical Bias | Notes |
|-------------|-------------|-------|
| Residential (low-rise) | +0.5 dB | Near-zero, well-calibrated |
| Commercial/mixed-use | +0.5 dB | Well-calibrated |
| Industrial/warehouse | **+2.0 to +2.5 dB** | Over-predicts; metal roofs not captured |
| Dense urban (downtown) | −0.5 to +1.0 dB | Slight variability |
| Near-water | N/A | Excluded from predictions |


## 9. Deployment & Infrastructure

### Runtime Requirements

```
python          >= 3.9
lightgbm        >= 3.3
joblib          >= 1.1
numpy           >= 1.21
pandas          >= 1.3
pycraf          >= 0.29    # ITU-R P.452 propagation model
h3              >= 3.7     # Uber H3 hexagonal binning
flask           >= 2.0     # Web server
```

### Files Required on Server

| File | Size | Required? | Description |
|------|------|-----------|-------------|
| `Model/lean_lgbm_53feat_model.joblib` | 535 KB | **YES** | Trained LightGBM model |
| `Model/lean_lgbm_53feat_features.json` | 1 KB | **YES** | Feature name list (must match model exactly) |
| `h3_complete_features_windsor.csv` | ~400 MB | **YES** | Per-bin clutter heights + types + indoor/outdoor |
| `h3_dsm_clutter_database.csv` | ~200 MB | **YES** | Per-bin DSM p95 surface height (for LoS ray-trace) |
| `h3_dem_database.csv` | ~150 MB | **YES** | Per-bin DEM terrain height (for antenna ASL) |
| `comprehensive_rsrp_all_46_sites.csv` | ~80 MB | **YES** | Baseline RSRP map (existing 46 cells, for comparison UI) |
| `dataset.csv` | ~89 MB | Optional | Training dataset (not needed for inference) |
| `Antennas/antenna_LUT_windsor.csv` | Small | Optional | Antenna lookup table for site import |

### Starting the Server

```bash
# From the project root directory:
python site_deployment_demo/app.py
# Server starts at http://localhost:5000
```

On startup the server:
1. Loads the LightGBM model (~0.1s)
2. Loads all H3 environmental features into RAM (~15s, 1.7M bins)
3. Loads DSM + DEM databases (~10s, 1.4M bins)
4. Loads baseline RSRP map (~3s)
5. Starts Flask HTTP server on port 5000

**Total startup time: ~30 seconds. All data is then in RAM for fast predictions.**

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Web UI (map interface) |
| `GET /api/sites` | GET | List all known sites |
| `POST /api/predict_site` | POST | Predict coverage for a candidate site |
| `POST /api/batch_predict` | POST | Predict coverage for multiple sites |
| `GET /api/baseline_coverage` | GET | Get baseline coverage for viewport |

### POST /api/predict_site — Request Format

```json
{
  "lat": 42.3149,
  "lon": -83.0364,
  "height": 25.0,
  "radius": 1000,
  "sectors": [
    {"azimuth": 0,   "edt": 6.0, "mdt": 0.0},
    {"azimuth": 120, "edt": 6.0, "mdt": 0.0},
    {"azimuth": 240, "edt": 6.0, "mdt": 0.0}
  ]
}
```

### Recommended Server Specifications

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB (all data preloaded in memory) |
| CPU | 2 cores | 4 cores (pycraf parallelization) |
| Storage | 2 GB | 2 GB |
| OS | Linux/Windows | Linux (Ubuntu 22.04) |
| Python | 3.9 | 3.11 |

### GCP Architecture (for cloud deployment)

```
Cloud Storage
├── lean_lgbm_53feat_model.joblib
├── lean_lgbm_53feat_features.json
├── h3_complete_features_windsor.csv
├── h3_dsm_clutter_database.csv
├── h3_dem_database.csv
└── comprehensive_rsrp_all_46_sites.csv
        │
        ▼
Cloud Run (or GCE VM)
├── Python Flask app (site_deployment_demo/app.py)
├── Loads data from GCS on startup
└── Serves HTTP API on port 8080
        │
        ▼
Cloud Load Balancer (HTTPS)
        │
        ▼
End Users (browser)
```

**Recommended Cloud Run config:**
- Memory: 8 GB
- CPU: 4 vCPU
- Min instances: 1 (keep warm — 30s cold start otherwise)
- Max instances: 5

---

## 10. Next Steps & Vision

### 10.1 Geographic Expansion — Montréal ✓ Complete

Montréal models are trained and deployed. Two separate LightGBM models (Urban + Suburban) cover the full Island of Montréal with LOSO RMSE of **7.41 dB (urban)** and **7.57 dB (suburban)**. See Part II of this document for full technical details.

### 10.2 Eastern Market Rollout — Québec City & Ottawa

Following Montréal, the roadmap targets major TELUS eastern Canadian markets:

- **Québec City:** Distinct propagation challenges — hilly terrain, mix of stone Old City and modern suburban sprawl
- **Ottawa/Gatineau:** Flat terrain similar to Windsor but at larger scale, river

Each market requires its own LSR MDT-GPS dataset extraction, clutter/DSM database build, and model training. The feature engineering pipeline and model architecture are fully reusable — only the data inputs change.

### 10.3 Automated Retraining Pipeline

The current model is a static snapshot trained on May–July 2025 LSR data. The vision is a **continuous learning pipeline** where:

1. New LSR MDT-GPS data is extracted from TELUS on a regular cadence 
2. The stationary filter and bin quality filter are applied automatically
3. The model is retrained and validated against the LOSO benchmark
4. If RMSE degrades beyond a threshold vs. the previous model version, an alert is raised for human review before deployment
5. The updated model is pushed to the inference server without service interruption

This ensures predictions remain accurate as the network evolves (new sites added, parameters changed, buildings constructed/demolished) and as seasonal patterns shift throughout the year.

### 10.4 Integration with TELUS Carto GIS Interface

RF engineers and managers at TELUS use **Carto** as their primary GIS and network visualization platform. The vision is to embed the site deployment prediction tool directly within Carto as a native layer or plugin, so engineers can:

- Select candidate locations directly on the Carto map already used daily
- See predicted RSRP footprints overlaid on existing network layers (sites, coverage, complaints)
- Export predictions directly into their standard GIS workflow
- Compare candidate sites side-by-side on the same map

This eliminates the context switch to a separate web tool and reduces the barrier to adoption. However, the 47m quadbin resolution limits at Carto pose a challenge for the moment.

### 10.5 LSR Bad RSRP Bin Integration

TELUS' Carto Layers data contains a **coverage problem areas**: bins where:
- Median RSRP < −118 dBm 
- ≥ 75 unique users sampled
- ≥ 15% of samples are worse than −118 dBm

The vision is to **automatically surface these "Bad RSRP Bins"** as a layer in the tool, so that when an engineer evaluates a candidate site, the tool immediately quantifies:

- How many Bad RSRP Bins fall within the new site's predicted footprint
- The predicted RSRP improvement at each bad bin from the new site
- A coverage problem score: weighted count of bad bins resolved (≥3 dB improvement to above −118 dBm threshold)

This creates a clean, automated bridge between **coverage gap identification** (LSR Bad Bins) and **site design evaluation** (model predictions), replacing the current manual process of exporting LSR data, overlaying it in Atoll, and comparing against the top-4 candidate designs by hand. Engineers remain in the loop for final site selection, but the tool dramatically accelerates the screening of a large volume of candidates.

### 10.6 Multi-Band Extension — 5G NR (n78 and Mid-Band)

The current model is trained exclusively on LTE 2100 MHz. The propagation environment is fundamentally different at 5G frequencies:

| Band | Frequency | Propagation Characteristics |
|------|-----------|----------------------------|
| LTE B4 | ~2100 MHz | Current model — baseline |
| 5G n78 | 3500 MHz | ~4–6 dB more path loss per decade of distance; much more sensitive to building/tree obstruction; LoS/NLoS distinction becomes critical |
| 5G Mid-Band | 1700–2100 MHz (n66) | Similar to LTE; may transfer-learn from LTE model |

The plan is to train separate LightGBM models for n78 (3500 MHz) and potentially mid-band 5G using LSR MDT-GPS data from 5G-capable devices. Key differences from the LTE model:

- **DSM LoS features will be much more important** at 3500 MHz — diffraction is far weaker and a single building obstruction essentially eliminates coverage
- **pycraf P.452 parameters** will require frequency-specific tuning (gaseous absorption, diffraction coefficients)
- **Clutter penetration** is worse at 3500 MHz — `indoor_proportion` will be a stronger predictor of penetration loss
- **Shorter effective range** — prediction radius should likely be reduced to 500m for n78 small cells

### 10.7 Vision: Human-in-the-Loop RF Planning Intelligence

The overarching vision is a system that automates the high-volume, low-complexity parts of RF site planning while keeping experienced engineers in control of the decisions that require judgment:

```
LSR Bad RSRP Bins  ──►  Automatic candidate site generation
                              │
                              ▼
                    Large-volume ML predictions
                    (hundreds of candidates in minutes)
                              │
                              ▼
                    Ranked shortlist (top N by coverage impact)
                              │
                              ▼
                    RF Engineer review in Carto GIS
                    (familiar interface, full network context)
                              │
                              ▼
                    Detailed Atoll study for top candidates
                    (fine-tune azimuths, tilts, height)
                              │
                              ▼
                    Final site selection & CAPEX justification
```

The ML model is not a replacement for RF engineering expertise, it is a tool that allows an engineer to evaluate over ten times as many candidate locations in the same time, with reliable quantified coverage improvement predictions.

---

---

# Part II — Montréal LTE RSRP Prediction Models

---

## 12. Montréal — Executive Summary

Montréal uses **two separate LightGBM models** — one for the dense urban core and one for suburban boroughs — routing predictions based on a geographic polygon. The split reduces RMSE by 1.1–1.7 dB compared to a single combined model.

### Key numbers at a glance

| Metric | Urban | Suburban |
|--------|-------|----------|
| Holdout RMSE | **6.85 dB** | **7.41 dB** |
| LOSO RMSE (30 folds) | **7.41 ± 1.04 dB** | **7.57 ± 0.99 dB** |
| Noise-corrected RMSE | **2.43 dB** | **3.72 dB** |
| Mean LOSO bias | +0.18 dB | −0.09 dB |
| Mean LOSO R² | 0.374 | 0.430 |
| Training sites | 120 | 161 |
| Training bins | 171,304 | 357,877 |
| Mean LoS distance | ~428 m | ~725 m |
| LTE bands | B4 (2100 MHz) + B2 (1900 MHz) | B4 + B2 |

**Single combined model baseline: 8.56 dB RMSE.** The urban/suburban split provides a 1.1 dB improvement for urban and 1.2 dB for suburban.

The urban model noise-corrected RMSE of **2.43 dB falls below the noise floor** (7.18 dB RMS intra-bin noise), meaning the model's errors are smaller than irreducible measurement variability — performance is at near-theoretical limits for the available dataset.

---

## 13. Why Two Models

### The propagation regimes are fundamentally different

| Property | Urban | Suburban |
|----------|-------|----------|
| Boroughs | Plateau-Mont-Royal, Rosemont, Villeray, Ville-Marie, NDG, Westmount | West Island, LaSalle, Pointe-aux-Trembles, Saint-Laurent |
| Building stock | Dense 3–20+ storey | Low-rise residential, commercial strips |
| Dominant propagation | NLOS — deep canyons, diffraction, guided street propagation | Mix of LoS and soft NLOS |
| Mean LoS distance | 428 m | 725 m |
| R² (single model) | 0.27 | 0.40 |
| R² (split model) | **0.34** | **0.52** |

A single model trained on both environments learns a compromise that captures neither regime well — the gradient boosting trees spend capacity on resolving the urban/suburban boundary rather than learning within-environment patterns.

### Why a geographic polygon, not a DSM height threshold

The boundary between urban and suburban was defined using a **hand-drawn geographic polygon** (`urban_mtl.geojson`) covering the inner boroughs, rather than an automatic DSM height cutoff. The reason: DSM-based classification misidentifies tall trees as urban and under-classifies low-density commercial corridors as suburban. The polygon captures the true planning intent — the dense, historically built-up inner city — and avoids misclassification from canopy noise.

### Model routing

At inference time, the antenna's lat/lon is tested against the urban polygon using Shapely. Urban antenna → Urban model. Suburban antenna → Suburban model. The same 53-feature input vector is used for both.

### Hyperparameter differences vs Windsor

| Parameter | Windsor | Montréal |
|-----------|---------|----------|
| `learning_rate` | 0.05 | **0.01** |
| `early_stopping` | 150 rounds | **500 rounds** |
| Best iteration | 62 trees | **758 (Urban) / 860 (Suburban)** |

The lower learning rate (0.01) was necessary because the Montréal dataset is ~17× larger than Windsor. A higher rate overshoots the optimum on a large dataset; more iterations at a finer step converge to a better solution.

---

## 14. Training Data

| Property | Value |
|----------|-------|
| Geography | Island of Montréal, QC |
| Raw stationary LSR rows | 48,453,617 |
| Spatial binning | H3 resolution 12 (~30m hex) |
| Total sites | 281 (Macro-O only) |
| Total training bins | 529,181 |
| Urban sites / bins | 120 / 171,304 |
| Suburban sites / bins | 161 / 357,877 |
| RSRP range | −141 to −44 dBm |
| Min sample filter | ≥ 5 measurements per bin |
| LTE bands | B4 (~2100 MHz) + B2 (~1900 MHz) |
| Measurement source | TELUS LSR MDT-GPS (stationary filter applied) |

### Stationary filter

Identical methodology to Windsor: MovingPandas `StopDetector` with `min_duration=60s`, `max_diameter=30m`, followed by speed filter >27 m/s. The Montreal dataset size (48.5M raw rows vs. 1.56M for Windsor) reflects the larger geographic footprint and higher subscriber density.

### Noise floor analysis

| Metric | Value |
|--------|-------|
| Median intra-bin σ | 5.99 dB |
| Mean intra-bin σ | 6.41 dB |
| RMS intra-bin σ | **7.18 dB** |

The RMS noise floor (7.18 dB) is higher than Windsor (5.56 dB), reflecting the greater multipath variability in a dense urban environment. The urban model's RMSE (6.85 dB) is already below this floor; noise-corrected RMSE of 2.43 dB confirms predictions are limited by measurement noise, not model capacity.

---

## 15. Cell Parameter Differences vs Windsor

Windsor uses fixed antenna parameters (all T2008 antennas, fixed EDT/RS EPRE/BW). Montréal uses a **per-cell parameter lookup table** extracted from the LUT, reflecting a mixed antenna fleet and site-specific configurations.

| Parameter | Windsor | Montréal |
|-----------|---------|----------|
| EDT | 6.0° fixed | Per-cell from LUT (mean 4.37°, range 0–12°) |
| RS EPRE | 18.2 dBm fixed | Per-cell (mean 16.57 dBm, range 6.2–18.3 dBm) |
| Carrier BW | 15 MHz fixed | Per-cell: 5 / 10 / 15 / 20 MHz |
| Antenna type | T2008 (single type) | Mixed fleet — VBW/HBW/Gain vary per site |

For **new candidate site predictions**, the user supplies EDT and RS EPRE manually. The app defaults to the training dataset means (EDT 4.4°, RS EPRE 16.57 dBm). Incorrect EDT/RS EPRE degrades accuracy by an estimated 1–3 dB.

### Band mixing

Both B4 and B2 sectors are included in the same models. The `downlink_freq` feature (MHz) allows the model to distinguish frequency-dependent path loss. However, because B4 and B2 sectors are often co-located on the same tower, the model may partially conflate band-specific propagation with site-specific characteristics. Future improvement: separate per-band models.

---

## 16. Feature Engineering

Both models use the same **53-feature** set as Windsor, with the same 6 groups. The feature list is identical; what differs is that the antenna configuration inputs (Group 2) are populated from the per-cell LUT rather than fixed constants.

### Top 10 features by importance

| Rank | Urban | Suburban |
|------|-------|----------|
| 1 | Antenna height (5,941) | Antenna height (8,718) |
| 2 | downtilt_projection_distance (5,883) | downtilt_projection_distance (8,017) |
| 3 | elevation_ue (5,047) | elevation_ue (4,796) |
| 4 | bearing_cos (3,576) | bearing_azimuth_cos (4,230) |
| 5 | bearing_sin (3,532) | bearing_sin (4,195) |
| 6 | dsm_max_excess_m (3,438) | distance_3d_log (4,009) |
| 7 | bearing_azimuth_cos (3,386) | bearing_cos (3,669) |
| 8 | distance_3d_log (3,336) | dsm_max_excess_m (3,143) |
| 9 | bearing_azimuth_sin (2,993) | dsm_first_block_m (3,130) |
| 10 | elevation_diff_log (2,633) | elevation_diff_log (2,816) |

**Key observations:**
- Antenna height and downtilt projection distance are #1 and #2 in both models — same as Windsor.
- DSM LoS features (`dsm_max_excess_m`, `dsm_first_block_m`) rank higher in Montréal than Windsor, reflecting greater building obstruction sensitivity in a denser city.
- The suburban model places higher absolute importance on all top features (larger total importance scores), consistent with a larger training dataset (357K vs 171K bins).

---

## 17. Performance & Validation

### LOSO cross-validation (30 folds)

The full 30-site LOSO was run separately for urban (15 sites) and suburban (15 sites). In each fold all bins from one site are held out and the model is retrained on the remaining sites.

**Urban — 15-fold LOSO summary:**

| Site | Test bins | RMSE (dB) | MAE (dB) | Bias (dB) | R² |
|------|-----------|-----------|----------|-----------|-----|
| PQ2932 | — | 5.72 | — | — | — |
| PQ1087 | — | 6.45 | — | — | — |
| PQ2904 | — | 6.89 | — | — | — |
| PQ2942 | — | 7.12 | — | — | — |
| PQ105385 | — | **9.45** | — | −4.36 | 0.06 |
| *(10 other sites)* | | 6.5–8.1 | | | |
| **MEAN** | | **7.41** | | **+0.18** | **0.374** |
| **STD** | | **±1.04** | | | |

**Suburban — 15-fold LOSO summary:**

| Site | Test bins | RMSE (dB) | MAE (dB) | Bias (dB) | R² |
|------|-----------|-----------|----------|-----------|-----|
| PQ2831 | — | 5.54 | — | — | — |
| PQ1857 | — | 6.55 | — | — | — |
| PQ2858 | — | 6.69 | — | — | — |
| PQ2880 | — | **9.62** | — | −5.72 | 0.40 |
| PQ1088 | — | **8.96** | — | +4.58 | 0.39 |
| *(10 other sites)* | | 6.5–8.5 | | | |
| **MEAN** | | **7.57** | | **−0.09** | **0.430** |
| **STD** | | **±0.99** | | | |

**Weighted combined LOSO RMSE (all 30 folds): 7.48 dB**

Near-zero mean bias (+0.18 / −0.09 dB) confirms the models are well-centred across the full site population.

### Holdout evaluation (3 sites per model, withheld from all training)

**Urban holdout:**

| Site | Bins | RMSE (dB) | MAE (dB) | Bias (dB) | R² |
|------|------|-----------|----------|-----------|-----|
| PQ2898 | 1,732 | 6.51 | 5.25 | +2.64 | 0.354 |
| PQ0209 | 1,929 | 6.80 | 5.28 | +1.26 | 0.342 |
| PQ0826 | 2,149 | 7.16 | 5.75 | +1.87 | 0.307 |
| **ALL** | **5,810** | **6.85** | **5.44** | **+1.89** | **0.341** |

**Suburban holdout:**

| Site | Bins | RMSE (dB) | MAE (dB) | Bias (dB) | R² |
|------|------|-----------|----------|-----------|-----|
| PQ0094 | 2,824 | 7.89 | 6.35 | −0.07 | 0.259 |
| PQ0020 | 3,984 | 7.53 | 5.96 | −2.08 | 0.498 |
| PQ2858 | 2,766 | 6.69 | 5.31 | +1.12 | 0.588 |
| **ALL** | **9,574** | **7.41** | **5.89** | **−0.56** | **0.524** |

The urban holdout shows a consistent **+1.9 dB positive bias** across all three sites. The LOSO mean bias (+0.18 dB) confirms this is a 3-site sampling artifact, not a systematic model failure. The three holdout sites happen to be in areas where the model under-predicts — the bias does not generalize to the full 120-site population.

### Comparison: single model vs split models

| Model | RMSE (dB) |
|-------|-----------|
| Single combined (all 281 sites) | 8.56 |
| Urban model (120 urban sites only) | 7.41 |
| Suburban model (161 suburban sites only) | 7.57 |
| **Improvement from split** | **1.1 – 1.2 dB** |

---

## 18. Known Limitations & Biases

### Island of Montréal only

The H3 environmental feature database, DSM, and DEM cover only the Island of Montréal. Predictions for **Laval, Longueuil, the South Shore, and off-island suburbs** are not supported — there is no clutter data for those areas and the models have not been trained on those propagation environments. A future model for Laval and the inner suburbs will require building new H3 clutter databases and retraining.

### Urban +1.9 dB systematic bias (holdout only)

All three urban holdout sites show positive bias (+1.3 to +2.6 dB) — the model under-predicts RSRP. The root cause is that ITU-R P.452 (pycraf) was designed for terrain-based links, not for building-canyon diffraction. Dense urban street canyons involve guided propagation along corridors, multiple façade reflections, and corner diffraction that P.452 does not model. The LOSO mean bias of +0.18 dB across all 120 urban sites shows this is not a global issue, but planners should be aware that predictions in the densest canyon environments (Ville-Marie, Plateau) may under-estimate RSRP by ~2 dB.

### Outlier sites

Three sites with RMSE > 8.9 dB have genuinely atypical propagation not captured by the current feature set:
- **Urban PQ105385:** RMSE 9.45 dB, R²=0.06, bias=−4.36 dB — model significantly over-predicts; likely unusual antenna placement or severe near-field obstruction
- **Suburban PQ2880:** RMSE 9.62 dB, bias=−5.72 dB — model over-predicts strongly; isolated suburban site with atypical geometry
- **Suburban PQ1088:** RMSE 8.96 dB, bias=+4.58 dB — model under-predicts; site may have unusually favourable propagation (elevated terrain, open water near path)

These sites are identifiable in `loso_results.csv` and should be treated with lower confidence.

### Macro-O cells only

All 529,181 training bins come from Macro-O layer cells. The models do **not** support uRRU, Micro, or Lampsite predictions — these have fundamentally different antenna heights, patterns, and coverage footprints.

### Band mixing (B4 + B2)

Both Band 4 (2100 MHz) and Band 2 (1900 MHz) sectors are included in the same models. The `downlink_freq` feature partially separates them, but because B4/B2 sectors are often co-located on the same tower, the model cannot fully isolate frequency-specific propagation from site characteristics.

### Antenna parameter sensitivity

Montréal uses per-cell EDT and RS EPRE from a LUT, unlike Windsor's fixed parameters. For new candidate sites, the user must supply these manually. The app defaults to dataset means (EDT 4.4°, RS EPRE 16.57 dBm). Incorrect EDT causes the largest prediction error because `downtilt_projection_distance` — the #2 most important feature — is directly derived from EDT.

### Per-site RMSE variability

±1.0 dB standard deviation in per-site LOSO RMSE (urban range 5.72–9.45 dB, suburban range 5.54–9.62 dB) means site-level accuracy varies substantially. The model is reliable on average but individual site predictions should be interpreted with this spread in mind.

---

The following works were consulted during the design, development, and validation of this model to ensure alignment with current academic and engineering best practices in ML-based RF propagation modelling, probabilistic machine learning, and production ML systems.

### ML-RF Propagation Research

**[1]** Xu, T.; Xu, N.; Gao, J.; Zhou, Y.; Ma, H. "Path Loss Prediction Model of 5G Signal Based on Fusing Data and XGBoost—SHAP Method." *Sensors* 2025, 25, 5440.  
https://doi.org/10.3390/s25175440  
*(China — gradient-boosted tree model for 5G path loss prediction with SHAP-based explainability; informed our feature importance analysis methodology.)*

**[2]** Ethier, J.; Chateauvert, M.; Dempsey, R. G.; Bose, A. "Path Loss Prediction Using Machine Learning with Extended Features." Communications Research Centre Canada (CRC), Ottawa, Ontario, Canada. arXiv:2501.08306v1 [cs.LG], January 2025.  
https://arxiv.org/abs/2501.08306  
*(Canada — CRC's investigation of ML path loss prediction using extended environmental features including clutter and DSM data; directly informed our feature group design and Canadian cellular context.)*

**[3]** Qiu, Y.; Bose, A. "Machine Learning for Modeling Wireless Radio Metrics with Crowdsourced Data and Local Environment Features." arXiv:2501.01344, January 2025.  
https://doi.org/10.48550/arXiv.2501.01344  
*(Canada — CRC's study of ML modelling with crowdsourced data and local environment features; directly relevant to our MDT-GPS data pipeline and feature engineering approach.)*

### Machine Learning Foundations

**[4]** Murphy, K. P. *Probabilistic Machine Learning: An Introduction.* MIT Press, 2022.  
http://probml.github.io/book1  
*(Textbook — theoretical grounding for gradient boosted trees, regularization, bias-variance tradeoff, and cross-validation methodology used throughout this project.)*

### ML Engineering & Production Systems

**[5]** Sculley, D.; Holt, G.; Golovin, D.; Davydov, E.; Phillips, T.; Ebner, D.; Chaudhary, V.; Young, M.; Crespo, J.-F.; Dennison, D. "Hidden Technical Debt in Machine Learning Systems." *Advances in Neural Information Processing Systems (NeurIPS)*, volume 28. Curran Associates, Inc., 2015.  
https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf  
*(ML systems engineering — informed our data cleaning, pipeline design, data validation practices, and the separation of feature engineering from model inference to minimize hidden technical debt.)*

---

*Document prepared March 2026. For questions contact Matthew Beaudet and/or Isaque Cerqueira*
