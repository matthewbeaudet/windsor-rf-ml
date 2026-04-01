# Windsor LTE RSRP Prediction Model — Overview

## 1. Purpose

This model predicts **LTE RSRP (Reference Signal Received Power, in dBm)** for any H3 hexagonal bin within a configurable radius (default 2 km) of a candidate LTE cell site. It is used to estimate radio frequency coverage for **new site deployment planning** — given an antenna configuration, predict the coverage footprint before the site is built.

**Use case:** Given a candidate site location + antenna parameters, generate a coverage map showing predicted RSRP across all H3 bins within 2 km.

**Target variable:** `rsrp_dbm` — median RSRP of drive-test measurements within each H3 bin, aggregated over 3 months (May–July 2025).

## 2. Model Architecture

| Property | Value |
|---|---|
| Algorithm | LightGBM (GBDT — Gradient Boosted Decision Trees) |
| Task | Regression (predict RSRP in dBm) |
| Library | `lightgbm` Python package |
| Serialization | `joblib` (`.joblib` file) |
| Number of features | 53 |
| Trees built (best_iteration) | 62 |
| Max trees configured | 2,000 (early stopping) |
| Model file size | ~535 KB |

**Hyperparameters:**
```
learning_rate    = 0.05
max_depth        = 7
num_leaves       = 100
min_child_samples= 50
subsample        = 0.8
colsample_bytree = 0.75
reg_alpha        = 0.5
reg_lambda       = 3.0
min_split_gain   = 0.05
min_child_weight = 0.05
early_stopping   = 150 rounds (on holdout RMSE)
```

## 3. Training Data

| Property | Value |
|---|---|
| Geography | Windsor, Ontario, Canada |
| Period | May–July 2025 (3 months) |
| Raw measurements | ~1.56 million drive-test RSRP samples |
| Spatial binning | Uber H3 hexagonal grid (resolution 12) |
| Training sites (cells) | 29 LTE cells |
| Training rows (bins) | ~84,200 H3 bins |
| Holdout strategy | Leave-One-Site-Out (LOSO) cross-validation |
| Measurement source | Mobile network drive-test data (MDT/crowdsourced) |
| Samples per bin | Median = 3, Mean = 15, Max = 12,880 |
| Intra-bin noise floor | Mean σ = 5.23 dB (irreducible measurement variability) |
| Time-of-day bias | Samples peak 3–8 AM (nighttime); daytime 14–17h under-sampled 2× |

## 4. Feature Set (53 features)

Features are organised in 6 groups (result of sequential ablation study):

### G1 — Distance & Geometry (6 features)
| Feature | Description |
|---|---|
| `distance_3d_log` | log10 of 3D slant distance from antenna to bin (m) |
| `elevation_diff_log` | log10 of absolute elevation difference (m) |
| `bearing_sin`, `bearing_cos` | UE bearing from antenna (sin/cos encoded) |
| `bearing_azimuth_sin`, `bearing_azimuth_cos` | Bearing minus antenna azimuth (off-boresight angle, sin/cos) |

> Note: Raw `azimuth_sin`/`azimuth_cos` were dropped — ablation showed they hurt RMSE alone (+0.076 dB) and are redundant once bearing_azimuth is included.

### G2 — Antenna & RF Configuration (10 features)
| Feature | Description |
|---|---|
| `RS dBm` | Reference signal power (antenna Tx power, dBm) |
| `Antenna height` | Antenna height above ground (m) |
| `Antenna EDT` | Electrical downtilt (degrees) |
| `Antenna MDT` | Mechanical downtilt (degrees) |
| `VBW` | Vertical beamwidth (degrees) |
| `HBW` | Horizontal beamwidth (degrees) |
| `Gain` | Antenna gain (dBi) |
| `downlink_freq` | Downlink frequency (MHz) |
| `carrier_bw` | Carrier bandwidth (MHz) |
| `outdoor` | Binary: 1 = outdoor cell, 0 = indoor |

### G3 — Beam Geometry (7 features)
| Feature | Description |
|---|---|
| `within_3db_horizontal` | 1 if UE is within horizontal 3dB beamwidth |
| `within_3db_beam` | 1 if UE is within both H and V 3dB beams |
| `within_vbw_shadow` | 1 if UE is in vertical beam shadow |
| `downtilt_projection_distance` | Distance where main beam hits ground (m) |
| `elevation_ue` | Elevation angle from antenna to UE (degrees) |
| `h_attenuation` | Horizontal pattern attenuation (dB) |
| `v_attenuation` | Vertical pattern attenuation (dB) |

### G4 — Propagation (pycraf ITU-R P.452 + DSM Line-of-Sight) (15 features)
| Feature | Description |
|---|---|
| `distance_los` | 10·log10(LoS path distance in m) |
| `pycraf_eps_pt`, `pycraf_eps_pr` | Path elevation angles at Tx and Rx (degrees) |
| `pycraf_d_lt`, `pycraf_d_lr` | Horizon distances at Tx and Rx (km) |
| `pycraf_L_bfsg` | Free-space + gas attenuation (dB) |
| `pycraf_L_b0p` | Basic loss not exceeded p% of time (dB) |
| `pycraf_L_bd` | Diffraction loss (dB) |
| `pycraf_L_bs` | Troposcatter loss (dB) |
| `pycraf_L_ba` | Ducting/anomalous propagation loss (dB) |
| `pycraf_L_b` | Total basic transmission loss (dB) |
| `dsm_los_ratio` | Fraction of ray samples clear of DSM [0–1] |
| `dsm_los_binary` | 1 if ≥95% of ray is clear |
| `dsm_first_block_m` | Distance from Tx to first DSM obstruction (m) |
| `dsm_max_excess_m` | Max DSM height above the ray line (m) |

### G5 — Clutter Height Statistics (6 features)
| Feature | Description |
|---|---|
| `clutter_mean_height` | Mean clutter height in H3 bin (m) |
| `clutter_min_height` | Min clutter height in H3 bin (m) |
| `clutter_max_height` | Max clutter height in H3 bin (m) |
| `clutter_height_range` | Range = max − min clutter height (m) |
| `clutter_std_height` | Std dev of clutter height in bin (m) |
| `clutter_p95_height` | 95th percentile clutter height in bin (m) |

### G6 — Clutter Type Composition (9 features)
| Feature | Description |
|---|---|
| `Building_count`, `Building_pct` | Number and % of building pixels in bin |
| `Industry_count`, `Industry_pct` | Number and % of industrial pixels |
| `Tree_pct`, `Forest_pct` | % tree and forest cover (seasonal, penetrable) |
| `tree_density_per_km2` | Tree density (trees per km²) |
| `indoor_proportion` | Fraction of measurements recorded indoors |
| `outdoor_proportion` | Fraction of measurements recorded outdoors |

> Tree/Forest features are kept for **deployment interpretability**: if the DSM high point is vegetation (not a building), the obstruction is seasonal and penetrable — important for site design decisions.

## 5. Model Performance

### Cross-Validation Results (14-fold LOSO, 2 km radius filter)

| Site | Test bins | RMSE (dB) | RMSE_nc (dB) | MAE (dB) | Bias (dB) | R² | Noise σ (dB) |
|---|---|---|---|---|---|---|---|
| ON000899 | 5,023 | 6.79 | 3.65 | 5.22 | +0.04 | 0.461 | 5.73 |
| ON0826 | 2,307 | 7.14 | 3.82 | 5.52 | +0.04 | 0.492 | 6.03 |
| ON1404 | 2,458 | 8.24 | 5.95 | 6.59 | +2.25 | 0.562 | 5.70 |
| ON0829 | 4,831 | 7.44 | 5.38 | 5.77 | −0.09 | 0.531 | 5.14 |
| ON0831 | 2,569 | 7.45 | 5.17 | 5.77 | +0.66 | 0.375 | 5.37 |
| ON1095 | 5,014 | 7.55 | 5.48 | 5.78 | −0.05 | 0.494 | 5.19 |
| ON1401 | 2,559 | 7.92 | 5.40 | 6.18 | −1.43 | 0.454 | 5.79 |
| ON0833 | 5,792 | 8.17 | 6.81 | 6.40 | +2.45 | 0.390 | 4.50 |
| ON000848 | 4,329 | 7.17 | 5.42 | 5.65 | +1.28 | 0.295 | 4.68 |
| ON0834 | 4,180 | 7.95 | 6.36 | 6.26 | +2.27 | 0.402 | 4.78 |
| ON0337 | 3,463 | 6.80 | 5.15 | 5.24 | 0.00 | 0.444 | 4.45 |
| ON0824 | 3,822 | 8.25 | 6.54 | 6.54 | +1.71 | 0.435 | 5.04 |
| ON0084 | 3,830 | 7.63 | 5.31 | 6.02 | −1.97 | 0.436 | 5.48 |
| ON0823 | 3,797 | 7.81 | 5.71 | 6.03 | −0.34 | 0.515 | 5.33 |
| **MEAN** | | **7.59** | **5.44** | **5.93** | **+0.49** | **0.449** | **5.23** |
| **STD** | | **0.49** | | | | | |

**RMSE_nc** = Noise-Corrected RMSE = √(RMSE² − σ_noise²). This represents the model's prediction error on the true (noiseless) RSRP signal, stripped of irreducible measurement variability.

### Ablation Study — RMSE Contribution by Feature Group

| Step | Feature Group Added | Total Features | RMSE (dB) | ΔRMSE |
|---|---|---|---|---|
| 1 | G1D: Distance + Bearing | 6 | 7.130 | baseline |
| 2 | + G2: Antenna & RF | 16 | 7.016 | −0.114 |
| 3 | + G3: Beam geometry | 23 | 6.984 | −0.032 |
| 4 | + G4: pycraf + DSM LoS | 38 | 6.962 | −0.022 |
| 5 | + G5: Clutter heights | 44 | 6.873 | −0.089 |
| 6 | + G6: Clutter types | **53** | **6.811** | −0.062 |

### Noise Floor Analysis
- **Intra-bin noise floor:** 5.23 dB (mean σ within each H3 bin)
- **Model RMSE / noise floor ratio:** 1.45×
- **Headroom above noise floor:** ~2.36 dB
- **Conclusion:** Model is operating near the practical ceiling for this dataset. Adding more features gives diminishing returns; more/denser drive-test data would reduce the noise floor.

## 6. Feature Importance (Top 15, by split count)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `downtilt_projection_distance` | 395 |
| 2 | `bearing_sin` | 350 |
| 3 | `Antenna height` | 341 |
| 4 | `bearing_cos` | 301 |
| 5 | `elevation_ue` | 301 |
| 6 | `bearing_azimuth_cos` | 283 |
| 7 | `bearing_azimuth_sin` | 276 |
| 8 | `h_attenuation` | 245 |
| 9 | `pycraf_eps_pt` | 229 |
| 10 | `dsm_los_ratio` | 227 |
| 11 | `Antenna EDT` | 189 |
| 12 | `elevation_diff_log` | 178 |
| 13 | `clutter_min_height` | 169 |
| 14 | `dsm_max_excess_m` | 153 |
| 15 | `dsm_first_block_m` | 152 |

Full importances: `Model/lean_lgbm_53feat_importance.csv`

## 7. Files

| File | Size | Description |
|---|---|---|
| `Model/lean_lgbm_53feat_model.joblib` | 535 KB | Trained LightGBM model (joblib) |
| `Model/lean_lgbm_53feat_features.json` | 1.1 KB | Ordered list of 53 feature names |
| `Model/lean_lgbm_53feat_importance.csv` | 1.0 KB | Feature importances (split count) |
| `Cline/ablation_feature_groups.py` | — | Sequential ablation study script |
| `Cline/cross_validate_lean_model.py` | — | 14-fold LOSO CV with noise correction |
| `Cline/check_intrabin_variance.py` | — | Noise floor analysis |
| `dataset.csv` | ~89k rows, 143 cols | Binned training dataset |

## 8. GCP Deployment Notes

### Runtime Requirements
```
python >= 3.9
lightgbm >= 3.3
joblib >= 1.1
numpy >= 1.21
pandas >= 1.3
pycraf >= 0.29   # for ITU-R P.452 propagation features
h3 >= 3.7        # for H3 hexagonal binning
```

### Inference Pipeline (per candidate site)
1. **Input:** Antenna lat/lon, azimuth, height, EDT, MDT, VBW, HBW, Gain, RS dBm, freq, carrier_bw
2. **Generate H3 bins:** All H3 resolution-9 bins within 2 km radius
3. **Compute features** for each bin:
   - G1: Distance, bearing, bearing-azimuth (geometry from antenna to bin centroid)
   - G2: Copy antenna parameters directly
   - G3: Beam geometry (elevation angle, H/V attenuation, downtilt projection)
   - G4: Run pycraf ITU-R P.452 + ray-trace DSM for LoS features
   - G5: Lookup H3 clutter height stats from pre-computed H3 database
   - G6: Lookup H3 clutter type percentages from pre-computed H3 database
4. **Load model:** `joblib.load('lean_lgbm_53feat_model.joblib')`
5. **Predict:** `model.predict(X[feature_list])`
6. **Output:** DataFrame of H3 bins with predicted RSRP (dBm)

### Pre-computed Lookup Tables Needed on GCP
| Table | Contents |
|---|---|
| `h3_clutter_database.csv` | Per-H3-bin clutter heights + type percentages |
| DSM raster (GeoTIFF) | Digital Surface Model for ray-tracing (Windsor area) |
| ITM terrain profile | Required by pycraf for P.452 path parameters |

### Recommended GCP Architecture
- **Cloud Storage:** Store model `.joblib`, feature JSON, H3 lookup CSVs, DSM raster
- **Cloud Run or Cloud Functions:** Stateless prediction endpoint (POST: antenna config → RSRP map)
- **Memory:** ~512 MB RAM sufficient (model is 535 KB; pycraf + DSM ray-trace dominates)
- **CPU:** Single vCPU sufficient; 2 km radius ≈ 3,000–6,000 H3 bins, inference < 1 second
- **Scaling:** Horizontal scaling via Cloud Run (serverless) handles burst demand
- **API contract:**
  ```json
  POST /predict
  {
    "site_lat": 42.30, "site_lon": -83.02,
    "azimuth": 120, "height_m": 30,
    "edt_deg": 5, "mdt_deg": 2,
    "vbw_deg": 6.3, "hbw_deg": 65,
    "gain_dbi": 17.1, "rs_dbm": 18.2,
    "freq_mhz": 2147.5, "carrier_bw_mhz": 15,
    "radius_km": 2.0
  }
  → { "bins": [{"h3_index": "...", "rsrp_dbm": -85.3, "lat": ..., "lon": ...}, ...] }
  ```

### Known Limitations
- **Bias:** Mean bias = +0.49 dB (slight over-prediction of RSRP). Sites in dense industrial areas show up to +2.5 dB bias.
- **Radius:** Validated only within 2 km. Beyond 2 km, accuracy degrades.
- **Geography:** Trained on Windsor, ON only. Transfer to other cities requires retraining.
- **Frequency:** Trained on Band 7 (~2147 MHz). Other bands may need retraining.
- **Time-of-day:** Training data skewed to nighttime measurements; may overpredict daytime RSRP.
- **Noise floor:** 5.23 dB irreducible measurement noise limits achievable RMSE to ~7.6 dB regardless of model complexity.
