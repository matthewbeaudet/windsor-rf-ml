"""
Site Deployment Tool - Flask Backend
Interactive demo for testing site deployment locations
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import threading
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from api.site_predictor import SitePredictor
from config.regions import REGIONS, WINDSOR

app = Flask(__name__)
CORS(app)

# ── Startup state ─────────────────────────────────────────────────────────────
predictor      = None
_startup_log   = []
_startup_ready = False
_startup_error = None
_current_region_cfg = WINDSOR   # set by /api/init before background thread starts


def _init_predictor_background(region_cfg):
    global predictor, _startup_ready, _startup_error, _baseline_with_coords, _clutter_height_index, _clutter_layer_index
    # Reset cached viewport indexes when region changes
    _baseline_with_coords = None
    _clutter_height_index = None
    _clutter_layer_index  = None
    try:
        _startup_log.append(f'> Initializing {region_cfg.display_name} RF ML Tool...')
        _startup_log.append('> Loading ML models and environmental data...')
        predictor = SitePredictor(region_config=region_cfg)
        _startup_log.append('> Baseline network loaded.')
        _startup_log.append('> System ready.')
        _startup_ready = True
    except Exception as e:
        _startup_error = str(e)
        _startup_log.append(f'> ERROR: {e}')


@app.route('/api/init', methods=['POST'])
def init_region():
    """
    Select region and start background initialization.
    Must be called before /api/status is polled.
    """
    global _current_region_cfg, _startup_log, _startup_ready, _startup_error, predictor, _ised_sites_cache
    region_name = (request.json or {}).get('region', 'windsor')
    cfg = REGIONS.get(region_name, WINDSOR)
    _current_region_cfg = cfg
    # Reset startup state and region-dependent caches
    predictor         = None
    _startup_log      = []
    _startup_ready    = False
    _startup_error    = None
    _ised_sites_cache = None   # force reload with correct region bounds
    get_sites._cached = None   # force reload with correct region sites
    threading.Thread(
        target=_init_predictor_background, args=(cfg,), daemon=True
    ).start()
    return jsonify({'region': cfg.name, 'display_name': cfg.display_name})

def _get_predictor():
    if predictor is None:
        from flask import abort
        abort(503, description="System still initializing — please wait.")
    return predictor

@app.route('/')
def index():
    """Serve the interactive map interface"""
    return render_template('index.html')

@app.route('/api/predict_site', methods=['POST'])
def predict_site():
    """
    Predict coverage impact for a potential site location
    
    Request JSON:
        {
            "lat": float,
            "lon": float,
            "height": float (optional, default 30m),
            "radius": float (optional, default 2000m)
        }
    
    Response JSON:
        {
            "geojson": GeoJSON FeatureCollection of improved bins,
            "stats": {
                "total_bins": int,
                "improved_bins": int,
                "mean_improvement": float,
                "max_improvement": float,
                "site_footprint_bins": int
            },
            "site_location": {"lat": float, "lon": float}
        }
    """
    try:
        data = request.json
        
        # Get parameters
        lat    = float(data['lat'])
        lon    = float(data['lon'])
        height = data.get('height', None)  # None triggers auto-detection from clutter
        if height is not None:
            height = float(height)
        radius        = float(data.get('radius', 1000))  # Default 1km
        zoom          = int(data.get('zoom', 13))         # Leaflet zoom level
        model_variant = str(data.get('model_variant', 'min5'))  # 'standard' or 'min5'
        edt           = float(data.get('edt', 6))         # Electrical downtilt from UI
        rs_power      = 43  # Hardcoded — RS power always 43 dBm, MDT always 0

        # Always use res-12 (full precision) — pycraf is now grouped by res-11 parents
        # so cost is ~1,400 pycraf calls / 8 workers ≈ 2s regardless of bin count.
        # Zoom still controls the prediction radius for visual clarity.
        h3_res = 12

        if height is None:
            print(f"Predicting site at ({lat:.4f}, {lon:.4f}), zoom={zoom}, H3 res {h3_res}, radius={radius}m")
        else:
            print(f"Predicting site at ({lat:.4f}, {lon:.4f}), height={height}m, zoom={zoom}, H3 res {h3_res}, radius={radius}m")

        # Run prediction
        result = _get_predictor().predict_site_deployment(
            site_lat=lat,
            site_lon=lon,
            site_height=height,
            radius_m=radius,
            h3_resolution=h3_res,
            model_variant=model_variant,
            edt=edt,
        )
        
        # Add site location to response
        result['site_location'] = {'lat': lat, 'lon': lon}
        result['model_variant'] = model_variant

        # Count bad IMSIs covered by the prediction footprint
        result['stats']['covered_bad_imsis'] = _covered_bad_imsis(result.get('footprint_geojson'))

        print(f"  {result['stats']['improved_bins']} bins improved by >=3dB  "
              f"/ {result['stats']['covered_bad_imsis']:,} bad IMSIs covered")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Pre-build baseline lookup with lat/lon at startup for fast viewport queries
_baseline_with_coords = None
# Cached clutter-height index (lat/lon for all bins with clutter_height > 0)
_clutter_height_index = None

def _get_baseline_with_coords():
    """Build baseline DataFrame with lat/lon from H3 index (cached)."""
    global _baseline_with_coords
    if _baseline_with_coords is None:
        import h3 as h3lib
        print("Building baseline coordinate index...")
        p = _get_predictor()
        df = p.baseline.copy()
        if df.empty:
            _baseline_with_coords = df
            return _baseline_with_coords
        # Normalise RSRP column to 'predicted_rsrp' for the API response
        rsrp_col = p.cfg.baseline_rsrp_col
        if rsrp_col in df.columns and rsrp_col != 'predicted_rsrp':
            df = df.rename(columns={rsrp_col: 'predicted_rsrp'})
        coords = [h3lib.cell_to_latlng(idx) for idx in df['h3_index']]
        df['lat'] = [round(c[0], 5) for c in coords]
        df['lon'] = [round(c[1], 5) for c in coords]
        _baseline_with_coords = df
        print(f"  OK Baseline coordinate index: {len(df):,} bins")
    return _baseline_with_coords


@app.route('/api/baseline_coverage', methods=['GET'])
def baseline_coverage():
    """
    Return baseline coverage for the current map viewport.
    Accepts bounding box params: min_lat, max_lat, min_lon, max_lon
    Only returns bins within the viewport for performance.
    """
    try:
        b = _current_region_cfg.bounds
        min_lat = float(request.args.get('min_lat', b['min_lat']))
        max_lat = float(request.args.get('max_lat', b['max_lat']))
        min_lon = float(request.args.get('min_lon', b['min_lon']))
        max_lon = float(request.args.get('max_lon', b['max_lon']))

        df = _get_baseline_with_coords()

        # Filter to viewport
        mask = (
            (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
            (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
        )
        viewport_df = df[mask]

        # Return as compact [lat, lon, rsrp] array
        points = viewport_df[['lat', 'lon', 'predicted_rsrp']].round({'predicted_rsrp': 1}).values.tolist()
        print(f"  Baseline viewport: {len(points):,} bins returned")
        return jsonify({'points': points})

    except Exception as e:
        print(f"Error in baseline_coverage: {e}")
        return jsonify({'error': str(e)}), 500


_ised_sites_cache = None

def _get_ised_sites(region_name=None):
    """Load and deduplicate ISED sites from local/GCS CSV (cached per region)."""
    global _ised_sites_cache
    if region_name is None:
        region_name = _current_region_cfg.name
    if _ised_sites_cache is not None:
        return _ised_sites_cache

    import os
    tmp_path = Path('/tmp/ISED_Overview_Table.csv')

    # Try local file first, then GCS
    if not tmp_path.exists():
        local_ised = Path(__file__).parent.parent / 'ISED Overview_Table.csv'
        if local_ised.exists():
            tmp_path = local_ised
            print(f"  OK Using local ISED CSV: {local_ised.name}")
        else:
            bucket_name = os.environ.get('GCS_BUCKET', 'windsor-rf-ml-data')
            try:
                from google.cloud import storage as gcs
                client = gcs.Client()
                blob = client.bucket(bucket_name).blob('ISED Overview_Table.csv')
                blob.download_to_filename(str(tmp_path))
                print(f"  OK Downloaded ISED_Overview_Table.csv from GCS bucket {bucket_name}")
            except Exception as e:
                print(f"  WARN GCS download failed for ISED CSV: {e}")
                _ised_sites_cache = []
                return _ised_sites_cache

    df = pd.read_csv(tmp_path, low_memory=False)

    # Filter to Windsor by coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    region_cfg = REGIONS.get(region_name, _current_region_cfg)
    b = region_cfg.bounds
    df = df[
        (df['latitude'].between(b['min_lat'], b['max_lat'])) &
        (df['longitude'].between(b['min_lon'], b['max_lon']))
    ].copy()
    print(f"  ISED {region_cfg.display_name} rows: {len(df):,}")

    # Round to 4 decimal places (~11m) to deduplicate co-located sectors
    df['lat_r'] = df['latitude'].round(4)
    df['lon_r'] = df['longitude'].round(4)

    sites = []
    for (lat_r, lon_r), group in df.groupby(['lat_r', 'lon_r']):
        def uniq(col):
            return sorted({str(v) for v in group[col].dropna() if str(v).strip()})
        max_h = None
        if 'max_ant_height' in group.columns:
            vals = group['max_ant_height'].dropna()
            if not vals.empty:
                max_h = float(vals.max())
        sites.append({
            'lat': float(lat_r),
            'lon': float(lon_r),
            'licensees': uniq('licensee_name'),
            'technologies': uniq('technology'),
            'bands': uniq('licence_category'),
            'sectors': int(len(group)),
            'max_ant_height': max_h,
        })

    print(f"  OK ISED sites deduplicated: {len(sites)} unique tower locations")
    _ised_sites_cache = sites
    return _ised_sites_cache


@app.route('/api/ised_sites', methods=['GET'])
def ised_sites():
    """Return deduplicated ISED colocation sites for the current region."""
    try:
        region_name = request.args.get('region', _current_region_cfg.name)
        sites = _get_ised_sites(region_name)
        return jsonify({'sites': sites})
    except Exception as e:
        print(f"Error in ised_sites: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sites', methods=['GET'])
def get_sites():
    """
    Return all existing site locations. Region-aware:
      - Windsor:  dataset.csv + Cline/missing_sites_dataset.csv
      - Montreal: mtl_cells_1900_2100.csv
    Accepts optional ?region= query param; falls back to _current_region_cfg.
    """
    region_name = request.args.get('region', _current_region_cfg.name)
    cache_key = f'_cached_{region_name}'
    if not hasattr(get_sites, cache_key) or getattr(get_sites, cache_key) is None:
        import pandas as pd
        base = Path(__file__).parent.parent

        if region_name == 'montreal':
            mtl_csv = Path(REGIONS['montreal'].h3_features_path).parent.parent / 'mtl_cells_1900_2100.csv'
            df = pd.read_csv(mtl_csv, usecols=['Latitude', 'Longitude', 'Site'], encoding='latin-1')
            df = df.dropna(subset=['Latitude', 'Longitude']).drop_duplicates(subset=['Site'])
            sites = [
                {'site_id': str(r['Site']), 'lat': round(float(r['Latitude']), 5), 'lon': round(float(r['Longitude']), 5)}
                for _, r in df.iterrows()
            ]
            print(f"  OK Sites loaded: {len(sites)} unique Montreal sites")
        else:
            cols = ['SiteID', 'antenna_lat', 'antenna_long']

            dataset_csv = Path('/tmp/dataset.csv')
            if not dataset_csv.exists():
                dataset_csv = base / 'dataset.csv'
            df1 = pd.read_csv(dataset_csv, usecols=cols)

            missing_csv = Path('/tmp/missing_sites_dataset.csv')
            if not missing_csv.exists():
                missing_csv = base / 'Cline' / 'missing_sites_dataset.csv'
            df2 = pd.read_csv(missing_csv, usecols=cols)

            df = pd.concat([df1, df2], ignore_index=True)
            df = df.dropna(subset=['antenna_lat', 'antenna_long'])
            df = df.drop_duplicates(subset=['SiteID'])
            sites = [
                {'site_id': row['SiteID'], 'lat': round(row['antenna_lat'], 5), 'lon': round(row['antenna_long'], 5)}
                for _, row in df.iterrows()
            ]
            print(f"  OK Sites loaded: {len(sites)} unique sites (merged from dataset.csv + missing_sites_dataset.csv)")

        setattr(get_sites, cache_key, sites)
    return jsonify({'sites': getattr(get_sites, cache_key)})


@app.route('/api/search_candidates', methods=['POST'])
def search_candidates():
    """
    Find the top-n candidate site locations within a drawn polygon.
    Candidates are H3 bins with the highest clutter_max_height inside the polygon.
    Uses Shapely for point-in-polygon testing.

    Request JSON:
        {
            "polygon": [[lat, lon], ...],   # polygon vertices (closed or open)
            "n": int                         # number of candidates (default 10, max 20)
        }

    Response JSON:
        {
            "candidates": [
                {"lat": float, "lon": float, "clutter_height": float, "rank": int},
                ...
            ],
            "n_bins_in_polygon": int
        }
    """
    try:
        from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
        import numpy as np
        import h3 as h3lib

        data = request.json
        polygon_coords = data['polygon']   # [[lat, lon], ...]
        n = min(int(data.get('n', 20)), 100)
        top_n_show = 5  # Always highlight top 5

        # Build Shapely polygon (x=lon, y=lat)
        poly = ShapelyPolygon([(lon, lat) for lat, lon in polygon_coords])
        min_lon, min_lat, max_lon, max_lat = poly.bounds

        # env_features is indexed by h3_index, no lat/lon columns
        # Compute lat/lon for all bins via H3 (vectorized via numpy)
        env_features = _get_predictor().env_features  # DataFrame indexed by h3_index

        if 'clutter_mean_height' not in env_features.columns:
            return jsonify({'error': 'clutter_mean_height not available in environmental features'}), 500

        # Cache building-dominant bins with lat/lon (built once per server lifetime).
        # Filter: Building_count >= 10 AND Building_count >= Tree_count
        #   → keeps bins where buildings are the dominant tall structure
        # Score:  clutter_mean_height (not max) — mean is stable; max is inflated by single trees/poles
        global _clutter_height_index
        if _clutter_height_index is None:
            has_building = 'Building_count' in env_features.columns
            has_tree     = 'Tree_count'     in env_features.columns
            has_mean     = 'clutter_mean_height' in env_features.columns

            if has_building and has_tree and has_mean:
                height_df = env_features[
                    (env_features['clutter_mean_height'] > 0) &
                    (env_features['Building_count'] >= 10) &
                    (env_features['Building_count'] >= env_features['Tree_count'])
                ].copy()
                score_col = 'clutter_mean_height'
                print(f"  Building index (Building >= 10 & >= Tree_count, score=mean_height): {len(height_df):,} bins...")
            elif has_building:
                height_df = env_features[
                    (env_features['clutter_max_height'] > 0) &
                    (env_features['Building_count'] >= 10)
                ].copy()
                score_col = 'clutter_max_height'
                print(f"  Building index (Building >= 10, score=max_height): {len(height_df):,} bins...")
            else:
                height_df = env_features[env_features['clutter_max_height'] > 0].copy()
                score_col = 'clutter_max_height'
                print(f"  Building index (no clutter filter): {len(height_df):,} bins...")

            height_df = height_df.copy()
            height_df['_score_col'] = score_col  # store which column to sort by
            coords = [h3lib.cell_to_latlng(idx) for idx in height_df.index]
            height_df['_lat'] = [c[0] for c in coords]
            height_df['_lon'] = [c[1] for c in coords]
            _clutter_height_index = height_df
            print(f"  OK Building index built: {len(_clutter_height_index):,} bins")
        height_df = _clutter_height_index
        # Determine which column was used for scoring
        score_col = height_df['_score_col'].iloc[0] if '_score_col' in height_df.columns else 'clutter_mean_height'

        # Bounding box filter (fast)
        bbox_mask = (
            (height_df['_lat'] >= min_lat) & (height_df['_lat'] <= max_lat) &
            (height_df['_lon'] >= min_lon) & (height_df['_lon'] <= max_lon)
        )
        bbox_df = height_df[bbox_mask]
        lat_arr = bbox_df['_lat'].values
        lon_arr = bbox_df['_lon'].values
        print(f"  Bounding box: {len(bbox_df):,} bins, checking polygon membership...")

        # Point-in-polygon
        in_poly = np.array([
            poly.contains(ShapelyPoint(lo, la))
            for la, lo in zip(lat_arr, lon_arr)
        ])
        poly_df = bbox_df[in_poly].copy()
        print(f"  Polygon: {len(poly_df):,} bins inside drawn area")

        # Top-n by mean building height (not max, to avoid tree spikes)
        poly_df = poly_df.dropna(subset=[score_col])
        top_n = poly_df.nlargest(n, score_col)

        candidates = []
        for rank, (h3_idx, row) in enumerate(top_n.iterrows(), 1):
            candidates.append({
                'rank': rank,
                'lat': round(float(row['_lat']), 5),
                'lon': round(float(row['_lon']), 5),
                'clutter_height': round(float(row[score_col]), 1),
                'h3_index': h3_idx
            })

        print(f"  Top {n} candidates found (max mean height: {top_n[score_col].max():.1f}m)")
        top5 = candidates[:5]
        return jsonify({
            'candidates': candidates,
            'top5': top5,
            'n_bins_in_polygon': int(len(poly_df))
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for CSV testing.
    Accepts CSV with columns: site_name, lat, lon, height
    Runs predictions in parallel for speed.
    
    CSV Format:
        site_name,lat,lon,height
        Downtown_20m,42.3149,-83.0364,20
        Downtown_25m,42.3149,-83.0364,25
        Eastside,42.3416,-82.9350,auto
    
    Returns JSON array of results for each site.
    """
    try:
        import io
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Get CSV data from request
        if 'file' in request.files:
            # File upload
            file = request.files['file']
            csv_data = file.read().decode('utf-8')
        elif 'csv_data' in request.json:
            # Direct CSV string
            csv_data = request.json['csv_data']
        else:
            return jsonify({'error': 'No CSV data provided. Include "file" or "csv_data"'}), 400
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Validate required columns
        required_cols = ['site_name', 'lat', 'lon', 'height']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return jsonify({'error': f'Missing required columns: {missing}'}), 400
        
        print(f"\n{'='*60}")
        print(f"BATCH PREDICTION: {len(df)} sites")
        print(f"{'='*60}")
        
        # Define prediction function for parallel execution
        def predict_one_site(row_idx, row):
            try:
                site_name = row['site_name']
                lat = float(row['lat'])
                lon = float(row['lon'])
                
                # Handle 'auto' height
                if isinstance(row['height'], str) and row['height'].lower() == 'auto':
                    height = None
                else:
                    height = float(row['height'])
                
                print(f"  [{row_idx+1}/{len(df)}] Predicting: {site_name} at ({lat:.4f}, {lon:.4f}), h={height if height else 'auto'}...")
                
                result = _get_predictor().predict_site_deployment(
                    site_lat=lat,
                    site_lon=lon,
                    site_height=height,
                    radius_m=1000,
                    h3_resolution=12  # Full resolution for batch testing
                )
                
                return {
                    'site_name': site_name,
                    'lat': lat,
                    'lon': lon,
                    'height': result['site_height'],
                    'improved_bins': result['stats']['improved_bins'],
                    'mean_improvement': round(result['stats']['mean_improvement'], 2),
                    'max_improvement': round(result['stats']['max_improvement'], 2),
                    'footprint_bins': result['stats']['site_footprint_bins'],
                    'total_bins': result['stats']['total_bins'],
                    'geojson': result['geojson'],                    # improved bins (>=3dB)
                    'footprint_geojson': result['footprint_geojson'],# all bins where new site wins
                    'status': 'success'
                }
            
            except Exception as e:
                import traceback
                print(f"  ERROR: {site_name} - {str(e)}")
                traceback.print_exc()
                return {
                    'site_name': row['site_name'],
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                    'height': row['height'],
                    'status': 'error',
                    'error': str(e)
                }
        
        # Run predictions in parallel (ThreadPoolExecutor for I/O-bound work)
        results = []
        max_workers = min(4, len(df))  # Use up to 4 parallel workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(predict_one_site, idx, row): idx
                for idx, row in df.iterrows()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"  FATAL ERROR on site {idx}: {e}")
                    results.append({
                        'site_name': df.iloc[idx]['site_name'],
                        'status': 'fatal_error',
                        'error': str(e)
                    })
        
        # Sort results by original CSV order (by site_name match)
        site_order = {row['site_name']: idx for idx, row in df.iterrows()}
        results.sort(key=lambda r: site_order.get(r['site_name'], 999))
        
        # Count successes
        successes = sum(1 for r in results if r['status'] == 'success')
        print(f"\nOK Batch complete: {successes}/{len(df)} successful\n")
        
        return jsonify({
            'results': results,
            'summary': {
                'total': len(df),
                'successful': successes,
                'failed': len(df) - successes
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def startup_status():
    """Startup progress — polled by the loading screen."""
    return jsonify({
        'ready': _startup_ready,
        'messages': _startup_log,
        'error': _startup_error,
    })


# Cached clutter layer index: DataFrame with lat/lon/median/p95, median > 6m
_clutter_layer_index = None

def _get_clutter_layer_index():
    """Build and cache the building height index (median/mean > 6m, Building_pct >= 75)."""
    global _clutter_layer_index
    if _clutter_layer_index is not None:
        return _clutter_layer_index

    import h3 as h3lib
    env = _get_predictor().env_features
    # Windsor has clutter_median_height; Montreal only has clutter_mean_height
    median_col = 'clutter_median_height' if 'clutter_median_height' in env.columns else 'clutter_mean_height'
    needed = {median_col, 'clutter_p95_height'}
    if not needed.issubset(env.columns):
        return None

    mask = env[median_col] > 6
    if 'Building_pct' in env.columns:
        mask = mask & (env['Building_pct'] >= 75)
    df = env[mask][[median_col, 'clutter_p95_height']].copy()
    coords = [h3lib.cell_to_latlng(idx) for idx in df.index]
    df['lat'] = [round(c[0], 5) for c in coords]
    df['lon'] = [round(c[1], 5) for c in coords]
    df = df.rename(columns={median_col: 'median', 'clutter_p95_height': 'p95'})
    df[['median', 'p95']] = df[['median', 'p95']].round(1)
    _clutter_layer_index = df
    print(f"  OK Building height index built: {len(df):,} bins ({median_col} > 6m, Building_pct >= 75%)")
    return _clutter_layer_index


@app.route('/api/clutter_heights', methods=['GET'])
def clutter_heights():
    """
    Return clutter median + p95 heights for bins in the viewport where median > 6m.
    Used by the high-zoom clutter height map layer.
    """
    try:
        b = _current_region_cfg.bounds
        min_lat = float(request.args.get('min_lat', b['min_lat']))
        max_lat = float(request.args.get('max_lat', b['max_lat']))
        min_lon = float(request.args.get('min_lon', b['min_lon']))
        max_lon = float(request.args.get('max_lon', b['max_lon']))

        df = _get_clutter_layer_index()
        if df is None:
            return jsonify({'points': [], 'error': 'clutter columns not available'})

        mask = (
            (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
            (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
        )
        vp = df[mask]
        points = vp[['lat', 'lon', 'median', 'p95']].values.tolist()
        print(f"  Clutter viewport: {len(points):,} bins")
        return jsonify({'points': points})
    except Exception as e:
        print(f"Error in clutter_heights: {e}")
        return jsonify({'error': str(e)}), 500


# ── Bad IMSI bins ─────────────────────────────────────────────────────────────
# Loaded once at startup from data/bad_bins_wnd.csv (or GCS fallback)
_bad_bins_df   = None   # DataFrame: h3_index, lat, lon, bad_ce_ims
_bad_bins_dict = None   # {h3_index: bad_ce_ims} for O(1) coverage lookup

def _load_bad_bins():
    global _bad_bins_df, _bad_bins_dict
    if _bad_bins_df is not None:
        return

    import os
    bucket_name = os.environ.get('GCS_BUCKET', 'windsor-rf-ml-data')
    blob_name   = os.environ.get('GCS_BAD_BINS_BLOB', 'weekly_bad_imsis')
    tmp_path    = Path('/tmp/bad_bins_wnd.csv')
    local_path  = Path(__file__).parent / 'data' / 'bad_bins_wnd.csv'

    # GCS-first: weekly BigQuery export overwrites this file
    if not tmp_path.exists():
        try:
            from google.cloud import storage as gcs
            client = gcs.Client()
            blob = client.bucket(bucket_name).blob(blob_name)
            blob.download_to_filename(str(tmp_path))
            print(f"  OK Downloaded {blob_name} from GCS bucket {bucket_name} (weekly data)")
        except Exception as e:
            print(f"  WARN GCS download failed: {e} -- using local fallback")

    source = tmp_path if tmp_path.exists() else local_path
    if not source.exists():
        _bad_bins_df   = pd.DataFrame(columns=['h3_index', 'lat', 'lon', 'bad_ce_ims'])
        _bad_bins_dict = {}
        return

    df = pd.read_csv(source)

    # BQ export has (quadbin, bad_ce_ims); local fallback already has h3_index
    if 'quadbin' in df.columns and 'h3_index' not in df.columns:
        import quadbin as qb
        import h3 as h3lib
        centers        = [qb.cell_to_point(int(c))['coordinates'] for c in df['quadbin']]
        df['lon']      = [c[0] for c in centers]
        df['lat']      = [c[1] for c in centers]
        df['h3_index'] = [h3lib.latlng_to_cell(lat, lon, 12)
                          for lat, lon in zip(df['lat'], df['lon'])]

    df['lat'] = df['lat'].round(5)
    df['lon'] = df['lon'].round(5)
    _bad_bins_df   = df
    _bad_bins_dict = dict(zip(df['h3_index'], df['bad_ce_ims']))
    print(f"  OK Bad IMSI bins loaded: {len(df):,} bins  "
          f"total={df['bad_ce_ims'].sum():,} IMSIs  "
          f"max={df['bad_ce_ims'].max():,}")

# Load eagerly (small file, fast)
threading.Thread(target=_load_bad_bins, daemon=True).start()


def _covered_bad_imsis(footprint_geojson):
    """Count bad IMSIs covered by a prediction footprint GeoJSON."""
    if _bad_bins_dict is None or not footprint_geojson:
        return 0
    total = 0
    for feat in footprint_geojson.get('features', []):
        h3_idx = feat.get('properties', {}).get('h3_index')
        if h3_idx:
            total += _bad_bins_dict.get(h3_idx, 0)
    return total


@app.route('/api/bad_imsi_bins', methods=['GET'])
def bad_imsi_bins():
    """Return bad IMSI bins within the viewport for map display."""
    try:
        _load_bad_bins()
        if _bad_bins_df is None or _bad_bins_df.empty:
            return jsonify({'points': []})

        b = _current_region_cfg.bounds
        min_lat = float(request.args.get('min_lat', b['min_lat']))
        max_lat = float(request.args.get('max_lat', b['max_lat']))
        min_lon = float(request.args.get('min_lon', b['min_lon']))
        max_lon = float(request.args.get('max_lon', b['max_lon']))

        df = _bad_bins_df
        mask = (
            (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
            (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
        )
        vp = df[mask]
        points = vp[['lat', 'lon', 'bad_ce_ims']].values.tolist()
        print(f"  Bad IMSI viewport: {len(points):,} bins")
        return jsonify({'points': points})
    except Exception as e:
        print(f"Error in bad_imsi_bins: {e}")
        return jsonify({'error': str(e)}), 500


# Cached poor-coverage index: baseline bins with predicted_rsrp <= -109.5
_poor_coverage_index = None

def _get_poor_coverage_index():
    """Build and cache the poor-coverage index (RSRP <= -109.5 dBm)."""
    global _poor_coverage_index
    if _poor_coverage_index is not None:
        return _poor_coverage_index

    import h3 as h3lib
    df = _get_baseline_with_coords()
    poor = df[df['predicted_rsrp'] <= -109.5][['lat', 'lon', 'predicted_rsrp']].copy()
    poor = poor.rename(columns={'predicted_rsrp': 'rsrp'})
    poor['rsrp'] = poor['rsrp'].round(1)
    _poor_coverage_index = poor
    print(f"  OK Poor coverage index built: {len(poor):,} bins (<= -109.5 dBm)")
    return _poor_coverage_index


@app.route('/api/poor_coverage', methods=['GET'])
def poor_coverage():
    """
    Return baseline bins with RSRP <= -109.5 dBm (poor + very poor) for the viewport.
    """
    try:
        b = _current_region_cfg.bounds
        min_lat = float(request.args.get('min_lat', b['min_lat']))
        max_lat = float(request.args.get('max_lat', b['max_lat']))
        min_lon = float(request.args.get('min_lon', b['min_lon']))
        max_lon = float(request.args.get('max_lon', b['max_lon']))

        df = _get_poor_coverage_index()
        mask = (
            (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
            (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
        )
        vp = df[mask]
        points = vp[['lat', 'lon', 'rsrp']].values.tolist()
        print(f"  Poor coverage viewport: {len(points):,} bins")
        return jsonify({'points': points})
    except Exception as e:
        print(f"Error in poor_coverage: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'predictor_loaded': predictor is not None,
        'baseline_bins': len(predictor.baseline_dict) if predictor else 0
    })

if __name__ == '__main__':
    print("="*60)
    print("Site Deployment Tool - Starting Flask Server")
    print("="*60)
    print("\nServer will be available at: http://localhost:5000")
    print("Open this URL in your web browser to use the tool")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    

@app.route('/api/height_sweep', methods=['POST'])
def height_sweep():
    """
    Run predictions for multiple antenna heights at a single location.
    Request JSON:
        {
            "lat": float,
            "lon": float,
            "heights": [float, ...],   # list of heights in metres
            "radius": float,           # prediction radius (default 1000)
            "model_variant": str       # 'standard' or 'min5'
        }
    Response JSON:
        {
            "results": [
                {"height": float, "improved_bins": int, "mean_improvement": float,
                 "max_improvement": float, "footprint_bins": int, "geojson": {...}},
                ...
            ],
            "best_height": float
        }
    """
    try:
        data = request.json
        lat           = float(data['lat'])
        lon           = float(data['lon'])
        heights       = [float(h) for h in data['heights']]
        radius        = float(data.get('radius', 1000))
        model_variant = str(data.get('model_variant', 'min5'))
        edt           = float(data.get('edt', 6))   # Electrical downtilt from UI

        if not heights:
            return jsonify({'error': 'No heights provided'}), 400
        if len(heights) > 20:
            return jsonify({'error': 'Maximum 20 heights per sweep'}), 400

        results = []
        for h in heights:
            print(f"  Height sweep: {h}m at ({lat:.4f},{lon:.4f}), EDT={edt} deg")
            result = _get_predictor().predict_site_deployment(
                site_lat=lat,
                site_lon=lon,
                site_height=h,
                radius_m=radius,
                h3_resolution=12,
                model_variant=model_variant,
                edt=edt,
            )
            # Improvement breakdown by dB band
            gjfeats = result.get('geojson', {}).get('features', [])
            bins_3to4 = sum(1 for f in gjfeats if f['properties'].get('is_improved') and 3 <= f['properties'].get('improvement_db', 0) < 4)
            bins_4to6 = sum(1 for f in gjfeats if f['properties'].get('is_improved') and 4 <= f['properties'].get('improvement_db', 0) < 6)
            bins_6plus = sum(1 for f in gjfeats if f['properties'].get('is_improved') and f['properties'].get('improvement_db', 0) >= 6)
            results.append({
                'height': h,
                'improved_bins':       result['stats']['improved_bins'],
                'mean_improvement':    round(result['stats']['mean_improvement'], 2),
                'max_improvement':     round(result['stats']['max_improvement'], 2),
                'footprint_bins':      result['stats'].get('site_footprint_bins', result['stats'].get('footprint_bins', 0)),
                'bins_3to4':           bins_3to4,
                'bins_4to6':           bins_4to6,
                'bins_6plus':          bins_6plus,
                'covered_bad_imsis':   _covered_bad_imsis(result.get('footprint_geojson')),
                'geojson':             result['geojson'],
                'footprint_geojson':   result.get('footprint_geojson'),
                'rsrp_geojson':        result.get('rsrp_geojson', result['geojson']),
            })

        best = max(results, key=lambda r: r['improved_bins'])
        return jsonify({'results': results, 'best_height': best['height']})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/save_png', methods=['POST'])
def save_png():
    """
    Save a base64 PNG from the client-side map capture.
    Request JSON:
        {
            "image": "data:image/png;base64,...",
            "filename": "sweep_30m.png"
        }
    Response JSON:
        {"saved": "exports/sweep_30m.png"}
    """
    try:
        import base64, re as _re
        data     = request.json
        img_data = data['image']
        filename = data.get('filename', 'export.png')
        # Sanitize filename
        filename = _re.sub(r'[^\w\-_.]', '_', filename)

        exports_dir = Path(__file__).parent / 'exports'
        exports_dir.mkdir(exist_ok=True)
        out_path = exports_dir / filename

        # Strip data URI header
        if ',' in img_data:
            img_data = img_data.split(',', 1)[1]
        with open(out_path, 'wb') as f:
            f.write(base64.b64decode(img_data))

        print(f"  PNG saved: {out_path}")
        return jsonify({'saved': str(out_path)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
