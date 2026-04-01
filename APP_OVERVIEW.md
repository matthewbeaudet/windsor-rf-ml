# Windsor LTE Site Deployment Tool — App Overview

## 1. System Architecture

The system consists of two layers:

```
┌─────────────────────────────────────────────────────────┐
│            site_deployment_demo  (Flask Web App)         │
│  - Interactive Leaflet map (browser)                     │
│  - Flask REST API (Python, port 5000)                    │
│  - api/site_predictor.py  ← orchestration layer         │
└────────────────────┬────────────────────────────────────┘
                     │ imports
┌────────────────────▼────────────────────────────────────┐
│               rf_design_tool  (Core Library)             │
│  - data_loader.py     → loads model + H3 env features   │
│  - feature_engine.py  → computes all 53 features        │
│  - prediction_engine.py → runs ML model + best-server   │
│  - map_builder.py     → GeoJSON / export utilities       │
└─────────────────────────────────────────────────────────┘
                     │ reads
┌────────────────────▼────────────────────────────────────┐
│                  Static Data Assets                      │
│  - Model/lean_lgbm_53feat_model.joblib  (535 KB)        │
│  - h3_complete_features_windsor.csv     (H3 env DB)      │
│  - comprehensive_rsrp_all_46_sites.csv  (baseline RSRP) │
│  - DSM raster (GeoTIFF)                                  │
└─────────────────────────────────────────────────────────┘
```

**Request lifecycle for a click-to-predict:**
1. User clicks map → browser POSTs `{lat, lon, radius}` to `/api/predict_site`
2. Flask calls `SitePredictor.predict_site_deployment()`
3. `SitePredictor` → `PredictionEngine.predict_site_coverage()` (3 sectors)
4. `PredictionEngine` → `FeatureEngine.calculate_batch_features()` (53 features per bin)
5. Model inference: `lgbm.predict(X)` → RSRP per bin
6. Compare vs baseline → compute improvement map
7. Return GeoJSON to browser → Leaflet renders coloured dots

## 2. rf_design_tool (Core Library)

**Path:** `rf_design_tool/`  
**Role:** Reusable Python library — no web server, no UI. Imported by `site_deployment_demo`.

### Module breakdown

| Module | Class | Responsibility |
|---|---|---|
| `modules/data_loader.py` | `DataLoader` | Load model `.joblib`, feature list JSON, H3 env features CSV, baseline RSRP CSV |
| `modules/feature_engine.py` | `FeatureEngine` | Compute all 53 ML features for each (sector, H3 bin) pair |
| `modules/prediction_engine.py` | `PredictionEngine` | Run batch inference, select best-serving sector per bin, integrate with baseline |
| `modules/map_builder.py` | `MapBuilder` | Export results to CSV / GeoJSON |

### Feature calculation flow (`feature_engine.py`)
For each (sector config × H3 bin centroid) pair:
1. **G1** — compute 3D distance, elevation diff, bearing, bearing-azimuth (sin/cos)
2. **G2** — copy antenna params (RS dBm, height, EDT, MDT, VBW, HBW, Gain, freq, BW, outdoor)
3. **G3** — compute elevation angle, H/V attenuation from antenna pattern, downtilt projection
4. **G4** — call `pycraf` ITU-R P.452 for path loss + ray-trace DSM raster for LoS features
5. **G5/G6** — join H3 bin to pre-computed clutter height and clutter type database

### PredictionEngine logic
- Calls `FeatureEngine.calculate_batch_features()` for all (sector × bin) combinations
- Runs `model.predict(X)` → RSRP per (sector, bin)
- For each bin: selects the **best serving sector** (max RSRP) → one output row per bin
- `integrate_with_baseline()`: for each bin, takes `max(baseline_rsrp, designed_rsrp)` → computes improvement

## 3. site_deployment_demo (Flask Web App)

**Path:** `site_deployment_demo/`  
**Role:** Flask REST API + Leaflet.js web UI for interactive site deployment simulation.

### Key files

| File | Purpose |
|---|---|
| `app.py` | Flask server, all REST endpoints, startup initialization |
| `api/site_predictor.py` | `SitePredictor` class — orchestrates predictions + baseline comparison |
| `templates/index.html` | Leaflet.js map UI (single-page app) |
| `requirements.txt` | Python dependencies |
| `START_SERVER.bat` | Windows launcher |
| `sample_batch_test.csv` | Example CSV for batch prediction endpoint |

### SitePredictor initialization (on server startup)
On first request (or server start), the following are loaded into memory:
- LightGBM model + feature list (from `rf_design_tool` `DataLoader`)
- Environmental H3 features for all bins (`h3_complete_features_windsor.csv`) — **preloaded for fast inference**
- Baseline RSRP (`comprehensive_rsrp_all_46_sites.csv`) — dict keyed by `h3_index`
- Baseline coordinate index (H3 → lat/lon) built lazily on first `/api/baseline_coverage` call

### Standard antenna configuration (hardcoded in site_predictor.py)
Every clicked site uses a standard 3-sector macro configuration:

| Parameter | Value |
|---|---|
| Sectors | 3 (azimuths: 0°, 120°, 240°) |
| RS Power | 43.0 dBm |
| Frequency | 2100 MHz |
| Bandwidth | 20 MHz |
| EDT (electrical downtilt) | 6° |
| MDT (mechanical downtilt) | 0° |

> These are typical urban macro parameters. Custom configurations can be added via API.

### Auto-height detection
When no height is specified, antenna height is estimated from clutter data at the clicked H3 bin:
- **Building-dominant bin** (Building_count ≥ 10 AND ≥ Tree_count): use `clutter_p95_height`
- **Mixed bin** (some buildings): use `clutter_mean_height`
- **No buildings**: use `clutter_max_height` (may be tree/pole)

### Improvement threshold
A bin is counted as "improved" if `new_site_rsrp − baseline_rsrp ≥ 3.0 dB`.

## 4. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serve Leaflet map HTML |
| POST | `/api/predict_site` | Main prediction: click → coverage map |
| GET | `/api/baseline_coverage` | Baseline RSRP for map viewport (bbox filter) |
| GET | `/api/sites` | All existing site locations (merged from 2 CSVs) |
| POST | `/api/search_candidates` | Find top-N candidate locations within drawn polygon |
| POST | `/api/batch_predict` | Batch prediction from uploaded CSV |
| GET | `/api/health` | Health check |

### POST `/api/predict_site`
```json
Request:
{
  "lat": 42.30,
  "lon": -83.05,
  "height": 25.0,       // optional, auto-detected if omitted
  "radius": 1000,        // metres, default 1000
  "zoom": 13             // Leaflet zoom (used for radius scaling)
}

Response:
{
  "geojson": { "type": "FeatureCollection", "features": [...] },
  "footprint_geojson": { "type": "FeatureCollection", "features": [...] },
  "stats": {
    "total_bins": 3142,
    "improved_bins": 487,
    "mean_improvement": 4.2,
    "max_improvement": 18.7,
    "site_footprint_bins": 612
  },
  "site_height": 25.0,
  "site_location": { "lat": 42.30, "lon": -83.05 }
}
```

Each GeoJSON feature is a **Point** (lat/lon of H3 bin centroid) with properties:
- `improvement_db` — dB gain vs baseline
- `final_rsrp`, `baseline_rsrp`, `new_site_rsrp` — all in dBm
- `is_improved` — boolean (true if ≥3 dB)
- `serving_sector`, `sector_name` — which of the 3 sectors wins

### POST `/api/search_candidates`
Accepts a drawn polygon (lat/lon vertices), returns top-N H3 bins ranked by building height (best rooftop candidates). Uses Shapely for point-in-polygon. Results include rank, lat/lon, and `clutter_height`.

### POST `/api/batch_predict`
Accepts CSV with columns: `site_name, lat, lon, height` (height can be `auto`).  
Runs predictions in parallel (up to 4 threads). Returns JSON array with per-site stats + GeoJSON.

## 5. Data Dependencies

| File | Size (approx) | Role |
|---|---|---|
| `Model/lgbm_binned_model.joblib` | 535 KB | ML model (loaded by data_loader.py) |
| `Model/lean_lgbm_53feat_features.json` | 1 KB | Feature order list |
| `h3_complete_features_windsor.csv` | ~50 MB | **Main env DB** — clutter heights, DSM LoS ratios, clutter types, tree stats for all Windsor H3 bins |
| `comprehensive_rsrp_all_46_sites.csv` | ~10 MB | Baseline RSRP predictions for 46 existing sites |
| `dataset.csv` | ~30 MB | Training dataset — used only for site marker lat/lon display |
| `Cline/missing_sites_dataset.csv` | ~5 MB | 16 additional site locations |

> **No DSM raster file is needed at runtime.** The DSM-based LoS features (`dsm_los_ratio`, `dsm_los_binary`, `dsm_first_block_m`, `dsm_max_excess_m`) were pre-computed offline during dataset construction and are stored inside `h3_complete_features_windsor.csv`. At inference time, pycraf is called with `generic_heights=True` (flat-earth geometric model — no terrain raster lookup). All terrain/clutter information is served from the pre-computed H3 database.

## 6. GCP Deployment Notes

### Runtime requirements
```
python >= 3.9
flask >= 2.0
flask-cors
lightgbm >= 3.3
joblib >= 1.1
numpy >= 1.21
pandas >= 1.3
h3 >= 3.7
pycraf >= 0.29
shapely >= 1.8
rasterio          # for DSM raster access
gunicorn          # production WSGI server
```

### Recommended GCP architecture

```
Internet
   │
   ▼
Cloud Load Balancer
   │
   ▼
Cloud Run (containerized Flask app)
   │   - Stateless, scales 0→N instances
   │   - Min instances: 1 (to avoid cold start preloading delay)
   │   - Memory: 4 GB (model 535 KB + H3 features ~1 GB in memory + pycraf)
   │   - CPU: 2 vCPU (pycraf P.452 is CPU-bound per bin)
   │   - Startup: ~30s (preloads env features + baseline into RAM)
   │
   ▼
Cloud Storage (GCS bucket)
   ├── model/lean_lgbm_53feat_model.joblib
   ├── model/lean_lgbm_53feat_features.json
   ├── data/h3_complete_features_windsor.csv
   ├── data/comprehensive_rsrp_all_46_sites.csv
   └── data/windsor_dsm.tif
```

### Startup optimization
The biggest latency on GCP is the cold start:
- `h3_complete_features_windsor.csv` must be loaded into memory (~1 GB pandas DataFrame)
- Set `min-instances: 1` in Cloud Run to keep one warm instance always alive
- Or pre-load data into **Cloud Memorystore (Redis)** for sub-second lookups

### Key environment variables to configure
```
MODEL_PATH=gs://your-bucket/model/lean_lgbm_53feat_model.joblib
FEATURES_PATH=gs://your-bucket/model/lean_lgbm_53feat_features.json
H3_ENV_PATH=gs://your-bucket/data/h3_complete_features_windsor.csv
BASELINE_PATH=gs://your-bucket/data/comprehensive_rsrp_all_46_sites.csv
DSM_PATH=gs://your-bucket/data/windsor_dsm.tif
PORT=8080
```

### Containerization (Dockerfile sketch)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2",
     "--timeout", "120", "site_deployment_demo.app:app"]
```

### Scaling considerations
- **pycraf per-bin cost:** ~5–20 ms per bin (P.452 ITU path loss + ray-trace)
- **2 km radius at H3 res-12:** ~3,000–6,000 bins × 3 sectors = 9,000–18,000 pycraf calls
- **Target latency:** ~3–8 seconds per click at 2 vCPU with parallel pycraf workers
- **Caching opportunity:** Pre-compute pycraf features for all (site, bin) pairs offline and store in H3 database — eliminates runtime pycraf calls entirely
- **Stateless design:** Each Cloud Run instance loads model + data independently; no shared state needed between instances
