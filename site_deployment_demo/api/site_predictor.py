"""
Site Predictor - region-aware API wrapper for RF predictions.
Supports Windsor (single model) and Montréal (Urban/Suburban dual model).
"""

import sys
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import h3

parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from rf_design_tool.modules.data_loader import DataLoader
from rf_design_tool.modules.prediction_engine import PredictionEngine


class SitePredictor:
    """
    Predicts coverage impact of deploying a new site.
    Pass a RegionConfig to switch between Windsor and Montréal.
    """

    def __init__(self, region_config=None, baseline_file=None):
        from config.regions import WINDSOR
        self.cfg = region_config if region_config is not None else WINDSOR
        print(f"Initializing Site Predictor for {self.cfg.display_name}...")

        # ── Data loader ──────────────────────────────────────────────────────
        self.loader = DataLoader(
            str(parent_dir),
            h3_features_path=str(self.cfg.h3_features_path),
            terrain_elevation_fallback=self.cfg.terrain_elevation_m,
            dsm_path=str(self.cfg.dsm_path) if self.cfg.dsm_path else None,
            dem_path=str(self.cfg.dem_path) if self.cfg.dem_path else None,
        )

        # ── Load models ──────────────────────────────────────────────────────
        self._router = None   # set for Montreal dual-model

        if self.cfg.urban_model_path:
            # Montreal: load both Urban and Suburban models
            self._load_montreal_models()
            # self.model / self.features point to the router (same interface)
            self.model    = self._router
            self.features = json.load(open(self.cfg.urban_features_path))
        else:
            # Windsor: load single model (+ min5 variant if available)
            self._models = {}
            if self.cfg.model_path and Path(self.cfg.model_path).exists():
                self._models['standard'] = (
                    joblib.load(self.cfg.model_path),
                    json.load(open(self.cfg.features_path)),
                )
                print(f'  standard model loaded')
            if self.cfg.model_min5_path and Path(self.cfg.model_min5_path).exists():
                self._models['min5'] = (
                    joblib.load(self.cfg.model_min5_path),
                    json.load(open(self.cfg.features_min5_path)),
                )
                print(f'  min5 model loaded')
            self.model, self.features = self._models.get(
                'min5', next(iter(self._models.values())))

        # ── Environmental features ───────────────────────────────────────────
        print("  Preloading environmental features...")
        self.env_features = self.loader.load_environmental_features()
        print(f"  OK {len(self.env_features):,} bins loaded")

        # ── DSM database ─────────────────────────────────────────────────────
        print("  Preloading DSM database...")
        self.dsm_lookup = self.loader.load_dsm_database()
        if self.dsm_lookup:
            print(f"  OK DSM database: {len(self.dsm_lookup):,} bins")
        else:
            print("  WARNING DSM database not found -- DSM LoS features use precomputed values")

        # ── Baseline ─────────────────────────────────────────────────────────
        baseline_path = Path(baseline_file) if baseline_file else self.cfg.baseline_path
        if baseline_path and Path(baseline_path).exists():
            print(f"  Loading baseline: {baseline_path.name}")
            bl = pd.read_csv(baseline_path)
            self.baseline = bl
            # Support both column names used by Windsor and Montreal baselines
            rsrp_col = self.cfg.baseline_rsrp_col
            if rsrp_col not in bl.columns:
                # Fallback: try the other known name
                rsrp_col = 'predicted_rsrp' if rsrp_col == 'baseline_rsrp' else 'baseline_rsrp'
            self.baseline_dict = dict(zip(bl['h3_index'], bl[rsrp_col]))
            print(f"  OK Baseline: {len(self.baseline_dict):,} bins  "
                  f"({bl[rsrp_col].min():.1f} to {bl[rsrp_col].max():.1f} dBm)")
        else:
            print("  WARNING No baseline file -- improvement metrics unavailable")
            self.baseline = pd.DataFrame(columns=['h3_index', 'baseline_rsrp'])
            self.baseline_dict = {}

        print(f"Site Predictor ready for {self.cfg.display_name}.\n")

    # ── Montreal dual-model loader ────────────────────────────────────────────

    def _load_montreal_models(self):
        import json
        from shapely.geometry import shape
        from api.montreal_router import MontrealRouter

        urban_model    = joblib.load(self.cfg.urban_model_path)
        suburban_model = joblib.load(self.cfg.suburban_model_path)
        with open(self.cfg.urban_poly_path) as f:
            gj = json.load(f)
        # Handle both raw geometry and FeatureCollection
        if gj.get('type') == 'FeatureCollection':
            gj = gj['features'][0]['geometry']
        elif gj.get('type') == 'Feature':
            gj = gj['geometry']
        urban_poly = shape(gj)
        self._router = MontrealRouter(urban_model, suburban_model, urban_poly)
        print(f'  Montreal Urban + Suburban models loaded')

    # ── Prediction entry point ────────────────────────────────────────────────

    def predict_site_deployment(self, site_lat, site_lon, site_height=None,
                                radius_m=1000, h3_resolution=10,
                                model_variant='min5', edt=None):
        # Windsor model switching
        if self._router is None and model_variant in getattr(self, '_models', {}):
            self.model, self.features = self._models[model_variant]
            print(f'  Using model: {model_variant}')

        # For Montreal, select Urban or Suburban based on antenna location
        env_label = ''
        if self._router is not None:
            env_label = self._router.select_model(site_lat, site_lon)
            print(f'  Montreal model: {env_label}')

        if site_height is None:
            site_height = self._get_building_height(site_lat, site_lon)
            print(f"  Auto-detected height: {site_height:.1f}m")

        if not hasattr(self, 'engine'):
            self.engine = PredictionEngine(self.model, self.features, self.loader)
        else:
            # Update model reference in case Windsor variant was switched
            self.engine.model = self.model

        edt_val   = edt if edt is not None else self.cfg.default_edt_deg
        rs_power  = self.cfg.default_rs_epre_dbm
        print(f"  Site: height={site_height:.1f}m  RS={rs_power}dBm  "
              f"EDT={edt_val}°  radius={radius_m}m")

        sectors = [
            {'name': f'Sector {i+1}', 'azimuth': az,
             'edt': edt_val, 'mdt': 0,
             'rs_power': rs_power, 'frequency': 2100, 'bandwidth': 20}
            for i, az in enumerate([0, 120, 240])
        ]

        predictions = self.engine.predict_site_coverage(
            site_lat=site_lat, site_lon=site_lon, site_height=site_height,
            sectors=sectors, radius_m=radius_m, measured_only=False,
            h3_resolution=h3_resolution, dsm_lookup=self.dsm_lookup,
        )

        improvements_df   = self._calculate_improvements(predictions)
        geojson           = self._to_geojson(improvements_df)
        footprint_geojson = self._footprint_to_geojson(improvements_df)

        improved_mask  = improvements_df['improvement'] >= 3.0
        footprint_mask = improvements_df['new_site_rsrp'] > improvements_df['baseline_rsrp']
        has_baseline   = bool(self.baseline_dict)

        stats = {
            'total_bins':        int(len(improvements_df)),
            'improved_bins':     int(improved_mask.sum()),
            'mean_improvement':  float(improvements_df['improvement'].mean()),
            'max_improvement':   float(improvements_df['improvement'].max()),
            'site_footprint_bins': int(footprint_mask.sum()),
            'has_baseline':      has_baseline,
        }

        print(f"  OK {stats['total_bins']:,} bins  "
              f"improved: {stats['improved_bins']:,}  "
              f"footprint: {stats['site_footprint_bins']:,}")

        return {
            'geojson':          geojson,
            'footprint_geojson': footprint_geojson,
            'stats':            stats,
            'site_height':      site_height,
            'region':           self.cfg.name,
        }

    # ── Building height auto-detection ───────────────────────────────────────

    def _get_building_height(self, lat, lon):
        h3_idx = h3.latlng_to_cell(lat, lon, 12)
        if h3_idx not in self.env_features.index:
            return 20.0
        bin_data      = self.env_features.loc[h3_idx]
        building_count = bin_data.get('Building_count', 0)
        tree_count     = bin_data.get('Tree_count', 0)
        p95_height     = bin_data.get('clutter_p95_height', None)
        mean_height    = bin_data.get('clutter_mean_height', None)
        max_height     = bin_data.get('clutter_max_height', None)

        if building_count >= 10 and building_count >= tree_count:
            h = p95_height if (p95_height is not None and not pd.isna(p95_height)) else mean_height
        elif building_count > 0:
            h = mean_height if (mean_height is not None and not pd.isna(mean_height)) else max_height
        else:
            h = max_height
        return float(h) if h is not None else 20.0

    # ── Baseline lookup ───────────────────────────────────────────────────────

    def _get_baseline_rsrp(self, h3_idx):
        if h3_idx in self.baseline_dict:
            return self.baseline_dict[h3_idx]
        try:
            children = h3.cell_to_children(h3_idx, 12)
            vals = [self.baseline_dict[c] for c in children if c in self.baseline_dict]
            if vals:
                return max(vals)
        except Exception:
            pass
        return -120.0

    # ── Improvement calculation ───────────────────────────────────────────────

    def _calculate_improvements(self, predictions):
        results = []
        for _, row in predictions.iterrows():
            h3_idx    = row['h3_index']
            new_rsrp  = row['designed_rsrp']
            base_rsrp = self._get_baseline_rsrp(h3_idx)
            final     = max(base_rsrp, new_rsrp)
            improvement = final - base_rsrp
            results.append({
                'h3_index':       h3_idx,
                'baseline_rsrp':  base_rsrp,
                'new_site_rsrp':  new_rsrp,
                'final_rsrp':     final,
                'improvement':    improvement,
                'is_improved':    improvement >= 3.0,
                'serving_sector': row['serving_sector'],
                'sector_name':    row['sector_name'],
            })
        return pd.DataFrame(results)

    # ── GeoJSON serialisation ─────────────────────────────────────────────────

    @staticmethod
    def _h3_to_polygon(h3_index):
        boundary = h3.cell_to_boundary(h3_index)
        coords   = [[round(lon, 6), round(lat, 6)] for lat, lon in boundary]
        coords.append(coords[0])
        return {'type': 'Polygon', 'coordinates': [coords]}

    def _footprint_to_geojson(self, df):
        footprint = df[df['new_site_rsrp'] > df['baseline_rsrp']]
        features  = []
        for _, row in footprint.iterrows():
            features.append({
                'type': 'Feature',
                'geometry': self._h3_to_polygon(row['h3_index']),
                'properties': {
                    'h3_index':       row['h3_index'],
                    'new_site_rsrp':  round(row['new_site_rsrp'],  1),
                    'baseline_rsrp':  round(row['baseline_rsrp'],  1),
                    'improvement_db': round(row['improvement'],    1),
                    'sector_name':    row['sector_name'],
                },
            })
        return {'type': 'FeatureCollection', 'features': features}

    def _to_geojson(self, df):
        features = []
        for _, row in df.iterrows():
            features.append({
                'type': 'Feature',
                'geometry': self._h3_to_polygon(row['h3_index']),
                'properties': {
                    'improvement_db': round(row['improvement'],   1),
                    'final_rsrp':     round(row['final_rsrp'],    1),
                    'baseline_rsrp':  round(row['baseline_rsrp'], 1),
                    'new_site_rsrp':  round(row['new_site_rsrp'], 1),
                    'is_improved':    bool(row['is_improved']),
                    'serving_sector': int(row['serving_sector']),
                    'sector_name':    row['sector_name'],
                },
            })
        return {'type': 'FeatureCollection', 'features': features}
