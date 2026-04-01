"""
Site Deployment Tool - Flask Backend
Interactive demo for testing site deployment locations
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import io
import pandas as pd
from pathlib import Path

# Add parent directory to path to import rf_design_tool modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.site_predictor import SitePredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for development

# Lazy-initialize predictor — loaded on first request so Cloud Run can bind the port first
predictor = None

def _get_predictor():
    global predictor
    if predictor is None:
        predictor = SitePredictor()
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
            print(f"Predicting site at ({lat:.4f}, {lon:.4f}), zoom={zoom} → H3 res {h3_res}, radius={radius}m")
        else:
            print(f"Predicting site at ({lat:.4f}, {lon:.4f}), height={height}m, zoom={zoom} → H3 res {h3_res}, radius={radius}m")

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
        
        print(f"  → {result['stats']['improved_bins']} bins improved by ≥3dB")
        
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
        df = _get_predictor().baseline.copy()
        coords = [h3lib.cell_to_latlng(idx) for idx in df['h3_index']]
        df['lat'] = [round(c[0], 5) for c in coords]
        df['lon'] = [round(c[1], 5) for c in coords]
        _baseline_with_coords = df
        print(f"  ✓ Baseline coordinate index built: {len(df):,} bins")
    return _baseline_with_coords


@app.route('/api/baseline_coverage', methods=['GET'])
def baseline_coverage():
    """
    Return baseline coverage for the current map viewport.
    Accepts bounding box params: min_lat, max_lat, min_lon, max_lon
    Only returns bins within the viewport for performance.
    """
    try:
        min_lat = float(request.args.get('min_lat', 42.2))
        max_lat = float(request.args.get('max_lat', 42.4))
        min_lon = float(request.args.get('min_lon', -83.2))
        max_lon = float(request.args.get('max_lon', -82.9))

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


@app.route('/api/sites', methods=['GET'])
def get_sites():
    """
    Return all existing site locations merged from:
      - dataset.csv          (original ~30 training sites)
      - Cline/missing_sites_dataset.csv  (16 additional sites)
    """
    if not hasattr(get_sites, '_cached'):
        import pandas as pd
        base = Path(__file__).parent.parent
        cols = ['SiteID', 'antenna_lat', 'antenna_long']

        # Primary dataset — prefer /tmp/ (GCS download) on Cloud Run
        dataset_csv = Path('/tmp/dataset.csv')
        if not dataset_csv.exists():
            dataset_csv = base / 'dataset.csv'
        df1 = pd.read_csv(dataset_csv, usecols=cols)

        # Additional 16 sites — prefer /tmp/ (GCS download) on Cloud Run
        missing_csv = Path('/tmp/missing_sites_dataset.csv')
        if not missing_csv.exists():
            missing_csv = base / 'Cline' / 'missing_sites_dataset.csv'
        df2 = pd.read_csv(missing_csv, usecols=cols)

        # Merge, deduplicate by SiteID
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.dropna(subset=['antenna_lat', 'antenna_long'])
        df = df.drop_duplicates(subset=['SiteID'])

        sites = [
            {'site_id': row['SiteID'], 'lat': round(row['antenna_lat'], 5), 'lon': round(row['antenna_long'], 5)}
            for _, row in df.iterrows()
        ]
        get_sites._cached = sites
        print(f"  ✓ Sites loaded: {len(sites)} unique sites (merged from dataset.csv + missing_sites_dataset.csv)")
    return jsonify({'sites': get_sites._cached})


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
            print(f"  ✓ Building index built: {len(_clutter_height_index):,} bins")
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
                    'geojson': result['geojson'],                    # improved bins (≥3dB)
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
        print(f"\n✓ Batch complete: {successes}/{len(df)} successful\n")
        
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
            print(f"  Height sweep: {h}m at ({lat:.4f},{lon:.4f}), EDT={edt}°")
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
                'improved_bins':    result['stats']['improved_bins'],
                'mean_improvement': round(result['stats']['mean_improvement'], 2),
                'max_improvement':  round(result['stats']['max_improvement'], 2),
                'footprint_bins':   result['stats'].get('site_footprint_bins', result['stats'].get('footprint_bins', 0)),
                'bins_3to4':        bins_3to4,
                'bins_4to6':        bins_4to6,
                'bins_6plus':       bins_6plus,
                'geojson':          result['geojson'],
                'footprint_geojson': result.get('footprint_geojson'),
                'rsrp_geojson':     result.get('rsrp_geojson', result['geojson']),
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
    app.run(debug=True, host='0.0.0.0', port=5000)
