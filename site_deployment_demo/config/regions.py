"""
Region configuration for the multi-region RF prediction server.
Adding a new region: add a RegionConfig entry to REGIONS dict.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Resolve paths relative to this file so the server works from any working directory
_HERE = Path(__file__).resolve().parent          # .../site_deployment_demo/config/
_REPO = _HERE.parent.parent                      # Windsor Mode Server root
_MTL  = _REPO.parent.parent / 'Montreal Full'   # G:/My Drive/Montreal Full


@dataclass
class RegionConfig:
    name: str
    display_name: str
    map_center: tuple                  # (lat, lon)
    map_zoom: int
    bounds: dict                       # min_lat/max_lat/min_lon/max_lon (for API filtering)
    terrain_elevation_m: float         # fallback when bin has no DEM data
    default_rs_epre_dbm: float         # RS EPRE for new-site predictions
    default_edt_deg: float             # default electrical downtilt in UI
    h3_features_path: Path
    baseline_path: Optional[Path]      # None = no baseline (raw RSRP shown)
    baseline_rsrp_col: str             # column name in baseline CSV
    bad_bins_path: Optional[Path] = None
    # Single-model region (Windsor)
    model_path: Optional[Path] = None
    model_min5_path: Optional[Path] = None
    features_path: Optional[Path] = None
    features_min5_path: Optional[Path] = None
    # DSM/DEM paths (None = use data loader defaults at repo root)
    dsm_path: Optional[Path] = None   # h3_index, p95_height
    dem_path: Optional[Path] = None   # h3_index, dem_mean
    # Dual-model region (Montreal)
    urban_model_path: Optional[Path] = None
    suburban_model_path: Optional[Path] = None
    urban_features_path: Optional[Path] = None
    suburban_features_path: Optional[Path] = None
    urban_poly_path: Optional[Path] = None


WINDSOR = RegionConfig(
    name='windsor',
    display_name='Windsor, ON',
    map_center=(42.3149, -83.0364),
    map_zoom=13,
    bounds=dict(min_lat=42.18, max_lat=42.42, min_lon=-83.15, max_lon=-82.88),
    terrain_elevation_m=183.0,
    default_rs_epre_dbm=18.2,
    default_edt_deg=6.0,
    h3_features_path=_REPO / 'h3_complete_features_windsor.csv',
    baseline_path=_REPO / 'comprehensive_rsrp_all_46_sites.csv',
    baseline_rsrp_col='predicted_rsrp',
    bad_bins_path=_HERE.parent / 'data' / 'bad_bins_wnd.csv',
    model_path=_REPO / 'Model' / 'lean_lgbm_53feat_model.joblib',
    model_min5_path=_REPO / 'Model' / 'lean_lgbm_53feat_min5_model.joblib',
    features_path=_REPO / 'Model' / 'lean_lgbm_53feat_features.json',
    features_min5_path=_REPO / 'Model' / 'lean_lgbm_53feat_min5_features.json',
)

MONTREAL = RegionConfig(
    name='montreal',
    display_name='Montréal, QC',
    map_center=(45.5017, -73.5673),
    map_zoom=12,
    bounds=dict(min_lat=45.40, max_lat=45.70, min_lon=-74.00, max_lon=-73.45),
    terrain_elevation_m=50.0,
    default_rs_epre_dbm=16.57,
    default_edt_deg=4.4,
    h3_features_path=_MTL / 'H3_Databases' / 'h3_complete_features_montreal.csv',
    baseline_path=_MTL / 'Model' / 'montreal_baseline_rsrp.csv',
    baseline_rsrp_col='baseline_rsrp',
    bad_bins_path=None,
    urban_model_path=_MTL / 'Model' / 'lgbm_montreal_53feat_urban_model.joblib',
    suburban_model_path=_MTL / 'Model' / 'lgbm_montreal_53feat_suburban_model.joblib',
    urban_features_path=_MTL / 'Model' / 'lgbm_montreal_53feat_urban_features.json',
    suburban_features_path=_MTL / 'Model' / 'lgbm_montreal_53feat_suburban_features.json',
    urban_poly_path=_MTL / 'urban_mtl.geojson',
    dsm_path=_MTL / 'H3_Databases' / 'h3_dsm_database_montreal.csv',
    dem_path=_MTL / 'H3_Databases' / 'h3_dem_database_montreal.csv',
)

REGIONS = {
    'windsor':  WINDSOR,
    'montreal': MONTREAL,
}
