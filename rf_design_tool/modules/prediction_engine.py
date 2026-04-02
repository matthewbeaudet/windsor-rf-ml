"""
Prediction Engine
Handles multi-sector RF predictions and baseline integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from .feature_engine import FeatureEngine

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Multi-sector RF prediction engine with baseline integration"""
    
    def __init__(self, model, feature_names: List[str], data_loader):
        """
        Initialize prediction engine
        
        Args:
            model: Trained LightGBM model
            feature_names: List of feature names expected by model
            data_loader: DataLoader instance
        """
        self.model = model
        self.feature_names = feature_names
        self.data_loader = data_loader
        self.feature_engine = FeatureEngine()
        
        logger.info(f"Prediction engine initialized with {len(feature_names)} features")
    
    def predict_site_coverage(self,
                             site_lat: float,
                             site_lon: float,
                             site_height: float,
                             sectors: List[Dict],
                             radius_m: float = 2000,
                             measured_only: bool = True,
                             h3_resolution: int = 12,
                             dsm_lookup: dict = None) -> pd.DataFrame:
        """
        Predict coverage for a multi-sector site

        Args:
            site_lat, site_lon: Site location
            site_height: Antenna height (meters)
            sectors: List of sector configurations
            radius_m: Prediction radius (meters)
            measured_only: If True, only predict on bins with measurements
            h3_resolution: H3 resolution for prediction grid (9-12).
                           Lower = fewer bins = faster. Default 12 (full detail).

        Returns:
            DataFrame with predictions for each H3 bin
        """
        logger.info(f"Predicting coverage for site at ({site_lat:.4f}, {site_lon:.4f})")
        logger.info(f"  Sectors: {len(sectors)}, Radius: {radius_m}m, Height: {site_height}m, H3 res: {h3_resolution}")

        # Get H3 bins within radius at requested resolution
        h3_bins_all = self.data_loader.get_h3_bins_in_radius(site_lat, site_lon, radius_m,
                                                               resolution=h3_resolution)
        
        # Filter to only bins with measurements if requested
        if measured_only:
            measured_rsrp = self.data_loader.load_measured_rsrp()
            h3_bins = [h3_idx for h3_idx in h3_bins_all if h3_idx in measured_rsrp]
            logger.info(f"  Prediction area: {len(h3_bins):,} bins WITH measurements (from {len(h3_bins_all):,} total)")
        else:
            h3_bins = h3_bins_all
            logger.info(f"  Prediction area: {len(h3_bins):,} H3 bins")
        
        # Load environmental features
        env_features = self.data_loader.load_environmental_features()
        
        # Calculate features for all sector-bin combinations
        features_df = self.feature_engine.calculate_batch_features(
            antenna_lat=site_lat,
            antenna_lon=site_lon,
            antenna_height=site_height,
            sector_configs=sectors,
            h3_bins=h3_bins,
            env_features_df=env_features,
            dsm_lookup=dsm_lookup,
        )
        
        logger.info(f"  Calculated features for {len(features_df):,} sector-bin combinations")
        
        # Align features with model expectations
        X = self._align_features(features_df)
        
        # Predict RSRP — use native booster to avoid sklearn wrapper version issues
        predictions = self.model.booster_.predict(X)
        features_df['predicted_rsrp'] = predictions
        
        # Group by H3 bin and find best serving sector
        results = []
        for h3_idx in h3_bins:
            bin_preds = features_df[features_df['h3_index'] == h3_idx]
            
            if len(bin_preds) == 0:
                continue
            
            # Find best serving sector
            best_idx = bin_preds['predicted_rsrp'].idxmax()
            best_row = bin_preds.loc[best_idx]
            
            results.append({
                'h3_index': h3_idx,
                'designed_rsrp': best_row['predicted_rsrp'],
                'serving_sector': best_row['sector_id'],
                'sector_name': best_row['sector_name']
            })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"  Predictions complete for {len(results_df):,} bins")
        
        return results_df
    
    def integrate_with_baseline(self, 
                                designed_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate designed site predictions with measured baseline
        
        Args:
            designed_predictions: DataFrame with designed site predictions
        
        Returns:
            DataFrame with combined network analysis
        """
        logger.info("Integrating designed site with measured baseline...")
        
        # Load measured RSRP
        measured_rsrp = self.data_loader.load_measured_rsrp()
        
        results = []
        
        for _, row in designed_predictions.iterrows():
            h3_idx = row['h3_index']
            designed = row['designed_rsrp']
            
            # Check if we have measured baseline
            measured = measured_rsrp.get(h3_idx, None)
            
            if measured is not None:
                # Has baseline measurement - can calculate real improvement
                final_rsrp = max(measured, designed)
                improvement = final_rsrp - measured
                
                if improvement > 0.5:  # Threshold for improvement
                    status = 'improved'
                else:
                    status = 'no_change'
            else:
                # No baseline measurement - unknown baseline
                # Don't claim improvement without knowing what was there before
                final_rsrp = designed
                improvement = 0  # Set to 0 since we don't know the baseline
                status = 'unknown_baseline'  # Mark as unknown rather than claiming new coverage
            
            results.append({
                'h3_index': h3_idx,
                'measured_rsrp': measured,
                'designed_rsrp': designed,
                'final_rsrp': final_rsrp,
                'improvement_db': improvement,
                'status': status,
                'serving_sector': row['serving_sector'],
                'sector_name': row['sector_name']
            })
        
        combined_df = pd.DataFrame(results)
        
        # Calculate statistics
        stats = self._calculate_statistics(combined_df, measured_rsrp)
        
        logger.info(f"  Combined network analysis complete")
        logger.info(f"    Improved bins: {stats['improved_bins']:,}")
        logger.info(f"    New coverage bins: {stats['new_coverage_bins']:,}")
        logger.info(f"    No change bins: {stats['no_change_bins']:,}")
        
        return combined_df, stats
    
    def _align_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align calculated features with model's expected features
        
        Args:
            features_df: DataFrame with calculated features
        
        Returns:
            DataFrame with only model features in correct order
        """
        # Create feature matrix with model's expected features
        X = pd.DataFrame()
        
        for feat_name in self.feature_names:
            if feat_name in features_df.columns:
                X[feat_name] = features_df[feat_name]
            else:
                # Feature missing - fill with default
                logger.warning(f"Feature '{feat_name}' not found, filling with 0")
                X[feat_name] = 0
        
        # Fill NaN values
        X = X.fillna(0)
        
        return X
    
    def _calculate_statistics(self, 
                              combined_df: pd.DataFrame,
                              measured_rsrp: Dict) -> Dict:
        """Calculate statistics for the combined network"""
        
        stats = {}
        
        # Count bins by status
        stats['total_bins'] = len(combined_df)
        stats['improved_bins'] = (combined_df['status'] == 'improved').sum()
        stats['new_coverage_bins'] = (combined_df['status'] == 'new_coverage').sum()
        stats['no_change_bins'] = (combined_df['status'] == 'no_change').sum()
        
        # Baseline network stats
        stats['baseline_bins'] = len(measured_rsrp)
        if measured_rsrp:
            stats['baseline_mean_rsrp'] = np.mean(list(measured_rsrp.values()))
            stats['baseline_min_rsrp'] = np.min(list(measured_rsrp.values()))
            stats['baseline_max_rsrp'] = np.max(list(measured_rsrp.values()))
        
        # Combined network stats
        stats['combined_mean_rsrp'] = combined_df['final_rsrp'].mean()
        stats['combined_min_rsrp'] = combined_df['final_rsrp'].min()
        stats['combined_max_rsrp'] = combined_df['final_rsrp'].max()
        
        # Improvement stats (only for improved bins)
        improved = combined_df[combined_df['status'] == 'improved']
        if len(improved) > 0:
            stats['mean_improvement'] = improved['improvement_db'].mean()
            stats['max_improvement'] = improved['improvement_db'].max()
            stats['median_improvement'] = improved['improvement_db'].median()
        else:
            stats['mean_improvement'] = 0
            stats['max_improvement'] = 0
            stats['median_improvement'] = 0
        
        # New coverage stats
        new_cov = combined_df[combined_df['status'] == 'new_coverage']
        if len(new_cov) > 0:
            stats['new_coverage_mean_rsrp'] = new_cov['final_rsrp'].mean()
        else:
            stats['new_coverage_mean_rsrp'] = 0
        
        # Sector contribution
        sector_contribution = combined_df.groupby('serving_sector').size().to_dict()
        stats['sector_contribution'] = sector_contribution
        
        # Signal quality distribution
        def classify_rsrp(rsrp):
            if rsrp >= -81.5:
                return 'Excellent'
            elif rsrp >= -88.5:
                return 'Good'
            elif rsrp >= -95.5:
                return 'Fair'
            elif rsrp >= -102.5:
                return 'Moderate'
            elif rsrp >= -109.5:
                return 'Poor'
            else:
                return 'Very Poor'
        
        combined_df['signal_quality'] = combined_df['final_rsrp'].apply(classify_rsrp)
        stats['signal_quality_dist'] = combined_df['signal_quality'].value_counts().to_dict()
        
        # Calculate baseline signal quality distribution for comparison
        if measured_rsrp:
            baseline_quality = pd.Series(list(measured_rsrp.values())).apply(classify_rsrp)
            stats['baseline_signal_quality_dist'] = baseline_quality.value_counts().to_dict()
        
        return stats
    
    def export_to_csv(self, combined_df: pd.DataFrame, output_path: str):
        """Export predictions to CSV"""
        # Add lat/lon for each H3 bin
        import h3
        
        lats = []
        lons = []
        for h3_idx in combined_df['h3_index']:
            lat, lon = h3.cell_to_latlng(h3_idx)
            lats.append(lat)
            lons.append(lon)
        
        export_df = combined_df.copy()
        export_df['latitude'] = lats
        export_df['longitude'] = lons
        
        export_df.to_csv(output_path, index=False)
        logger.info(f"Exported predictions to: {output_path}")
    
    def export_to_geojson(self, combined_df: pd.DataFrame, output_path: str):
        """Export predictions to GeoJSON"""
        import h3
        import json
        
        features = []
        
        for _, row in combined_df.iterrows():
            h3_idx = row['h3_index']
            
            # Get H3 polygon
            boundary = h3.cell_to_boundary(h3_idx)
            coords = [[lon, lat] for lat, lon in boundary]
            coords.append(coords[0])  # Close polygon
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [coords]
                },
                'properties': {
                    'h3_index': h3_idx,
                    'measured_rsrp': float(row['measured_rsrp']) if pd.notna(row['measured_rsrp']) else None,
                    'designed_rsrp': float(row['designed_rsrp']),
                    'final_rsrp': float(row['final_rsrp']),
                    'improvement_db': float(row['improvement_db']),
                    'status': row['status'],
                    'serving_sector': int(row['serving_sector']),
                    'sector_name': row['sector_name']
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"Exported predictions to: {output_path}")


if __name__ == "__main__":
    # Test prediction engine
    from data_loader import DataLoader
    
    print("\n" + "="*60)
    print("Testing Prediction Engine")
    print("="*60)
    
    # Initialize
    loader = DataLoader()
    model, features = loader.load_model()
    
    engine = PredictionEngine(model, features, loader)
    
    # Test prediction
    test_sectors = [
        {'name': 'Sector 1', 'azimuth': 0, 'edt': 6, 'mdt': 0, 
         'rs_power': 43.0, 'frequency': 2100, 'bandwidth': 20},
        {'name': 'Sector 2', 'azimuth': 120, 'edt': 6, 'mdt': 0,
         'rs_power': 43.0, 'frequency': 2100, 'bandwidth': 20},
        {'name': 'Sector 3', 'azimuth': 240, 'edt': 6, 'mdt': 0,
         'rs_power': 43.0, 'frequency': 2100, 'bandwidth': 20}
    ]
    
    # Predict coverage
    predictions = engine.predict_site_coverage(
        site_lat=42.3,
        site_lon=-83.0,
        site_height=30,
        sectors=test_sectors,
        radius_m=500  # Small radius for testing
    )
    
    print(f"\n✓ Predicted coverage for {len(predictions)} bins")
    
    # Integrate with baseline
    combined, stats = engine.integrate_with_baseline(predictions)
    
    print(f"\n✓ Integrated with baseline")
    print(f"  Improved: {stats['improved_bins']}")
    print(f"  New coverage: {stats['new_coverage_bins']}")
    print(f"  No change: {stats['no_change_bins']}")
    
    print("\n✅ Prediction engine test passed!")
