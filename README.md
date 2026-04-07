# Windsor RF ML — Site Deployment Prediction Tool

A machine-learning–powered RF planning tool that predicts the coverage impact of deploying a new cellular site anywhere in Windsor, Ontario. Engineers can click a location on an interactive map, and the model instantly shows which H3 bins would see ≥3 dB RSRP improvement, the new site's RF footprint, and the optimal antenna height.

> **Technical details:** See [MODEL_TECHNICAL_REFERENCE.md](MODEL_TECHNICAL_REFERENCE.md)

---

## Architecture

```
GitHub (this repo)          GCS Bucket: windsor-gcs
  ├── Dockerfile            ├── windsor/h3_complete_features_windsor.csv  (~400 MB)
  ├── cloudbuild.yaml       ├── windsor/h3_dsm_clutter_database.csv       (~200 MB)
  ├── requirements.txt      ├── windsor/h3_dem_database.csv               (~150 MB)
  ├── Model/                ├── windsor/comprehensive_rsrp_all_46_sites.csv (~80 MB)
  │   ├── lean_lgbm_53feat_model.joblib        ├── windsor/dataset.csv                     (~50 MB)
  │   └── lean_lgbm_53feat_features.json       └── windsor/Cline/missing_sites_dataset.csv
  ├── rf_design_tool/
  │   └── modules/
  │       ├── data_loader.py      ← downloads from GCS at startup on Cloud Run
  │       ├── feature_engine.py
  │       └── prediction_engine.py
  └── site_deployment_demo/
      ├── app.py             ← Flask backend
      ├── api/site_predictor.py
      └── templates/index.html
```

**Cloud Run** serves the Flask app via Gunicorn. On startup, `data_loader.py` detects the `GCS_BUCKET` environment variable and downloads the 6 large data files from GCS to `/tmp/` — this happens once per container instance. Locally, the files are read directly from disk.

---

## Running Locally

### Prerequisites
```
Python 3.11+
pip install -r requirements.txt
```

You also need the large data files in the repo root (same directory as `Dockerfile`):
- `h3_complete_features_windsor.csv`
- `h3_dsm_clutter_database.csv`
- `h3_dem_database.csv`
- `comprehensive_rsrp_all_46_sites.csv`
- `dataset.csv`
- `Cline/missing_sites_dataset.csv`

Contact the repo owner to obtain these from the `windsor-gcs` GCS bucket.

### Start the server
```bash
cd site_deployment_demo
python app.py
```
Then open **http://localhost:5000** in your browser.

Or use the batch file on Windows:
```
site_deployment_demo\START_SERVER.bat
```

---

## Deploying to Google Cloud Run

### One-time GCP setup (done once per project)

1. **Enable required APIs:**
   ```bash
   gcloud services enable cloudbuild.googleapis.com run.googleapis.com \
     artifactregistry.googleapis.com storage.googleapis.com
   ```

2. **Create Artifact Registry repository:**
   ```bash
   gcloud artifacts repositories create windsor-rf-ml \
     --repository-format=docker \
     --location=northamerica-northeast1
   ```

3. **Grant Cloud Build the Cloud Run Admin role:**
   ```bash
   PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
     --role="roles/run.admin"
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
     --role="roles/iam.serviceAccountUser"
   ```

4. **Upload data files to GCS** (see [GCS Data Layout](#gcs-data-layout) below).

5. **Connect GitHub repo to Cloud Build:**
   - Go to [GCP Console → Cloud Build → Triggers](https://console.cloud.google.com/cloud-build/triggers)
   - Click **Connect Repository** → Select GitHub → Choose `matthewbeaudet/windsor-rf-ml`
   - Create a trigger: Branch = `^main$`, Config = `cloudbuild.yaml`

### Deploy manually (first time or on-demand)
```bash
gcloud builds submit --config cloudbuild.yaml --project YOUR_PROJECT_ID
```

After the first manual deploy, every `git push origin main` auto-triggers Cloud Build.

---

## GCS Data Layout

Upload the following files to the `windsor-gcs` bucket **exactly** as shown:

```
gs://windsor-gcs/
└── windsor/
    ├── h3_complete_features_windsor.csv      # ~400 MB — env features for all Windsor H3 bins
    ├── h3_dsm_clutter_database.csv           # ~200 MB — DSM surface heights for LoS ray-trace
    ├── h3_dem_database.csv                   # ~150 MB — bare-ground terrain elevations
    ├── comprehensive_rsrp_all_46_sites.csv   # ~80 MB  — baseline network RSRP map (46 sites)
    ├── dataset.csv                           # ~50 MB  — site coordinates + training data
    └── Cline/
        └── missing_sites_dataset.csv         # additional 16 site coordinates
```

Upload command:
```bash
gsutil -m cp h3_complete_features_windsor.csv gs://windsor-gcs/windsor/
gsutil -m cp h3_dsm_clutter_database.csv      gs://windsor-gcs/windsor/
gsutil -m cp h3_dem_database.csv              gs://windsor-gcs/windsor/
gsutil -m cp comprehensive_rsrp_all_46_sites.csv gs://windsor-gcs/windsor/
gsutil -m cp dataset.csv                      gs://windsor-gcs/windsor/
gsutil -m cp Cline/missing_sites_dataset.csv  gs://windsor-gcs/windsor/Cline/
```

---

## Model

The production model is **LightGBM 53-feature min 5 samples per bin**  (`lean_lgbm_53feat_min5_model.joblib`).
Features include: H3-indexed clutter height statistics, building/tree counts, DSM LoS ray-trace metrics, DEM terrain elevation, pycraf path loss, fresnel zone clearance, and antenna geometry (azimuth, tilt, height, frequency).

See [MODEL_TECHNICAL_REFERENCE.md](MODEL_TECHNICAL_REFERENCE.md) for the full feature list and training methodology.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Interactive map UI |
| POST | `/api/predict_site` | Predict coverage for a new site location |
| POST | `/api/height_sweep` | Compare multiple antenna heights at one location |
| POST | `/api/batch_predict` | Batch predictions from a CSV file |
| POST | `/api/search_candidates` | Find top candidate locations inside a drawn polygon |
| GET | `/api/baseline_coverage` | Return current network RSRP for the map viewport |
| GET | `/api/sites` | Return all existing site locations |
| GET | `/api/health` | Health check |

---

## Repo Structure

```
windsor-rf-ml/
├── .gitignore
├── Dockerfile                          # Container definition for Cloud Run
├── cloudbuild.yaml                     # CI/CD pipeline (auto-deploy on git push)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── MODEL_TECHNICAL_REFERENCE.md        # Full model documentation
├── Antennas/
│   └── antenna_LUT_windsor.csv         # Antenna pattern lookup table
├── Model/
│   ├── lean_lgbm_53feat_model.joblib   # Production LightGBM model
│   ├── lean_lgbm_53feat_features.json  # Ordered feature list
│   ├── lean_lgbm_53feat_importance.csv # Feature importance
│   ├── lean_lgbm_53feat_min5_model.joblib
│   ├── lean_lgbm_53feat_min5_features.json
│   └── lean_lgbm_53feat_min5_importance.csv
├── rf_design_tool/
│   └── modules/
│       ├── __init__.py
│       ├── data_loader.py              # Loads data (GCS-aware)
│       ├── feature_engine.py           # Computes 53 features per bin
│       └── prediction_engine.py        # Runs ML inference
└── site_deployment_demo/
    ├── app.py                          # Flask app + all API routes
    ├── requirements.txt
    ├── README.md
    ├── START_SERVER.bat                # Windows local launch script
    ├── sample_batch_test.csv           # Example batch test input
    ├── api/
    │   ├── __init__.py
    │   └── site_predictor.py           # Orchestrates predictions
    └── templates/
        └── index.html                  # Interactive Leaflet map UI
```
