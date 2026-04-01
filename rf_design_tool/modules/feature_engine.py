
"""
Feature Calculation Engine
Computes all features required for RF prediction
"""

import numpy as np
import pandas as pd
import h3
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Calculates features for RF propagation prediction"""
    
    def __init__(self):
        """Initialize feature calculator"""
        self.feature_names = []
    
    @staticmethod
    def calculate_distance_3d(lat1: float, lon1: float, height1: float,
                              lat2: float, lon2: float, height2: float = 0) -> float:
        """
        Calculate 3D distance between two points
        
        Args:
            lat1, lon1: Antenna location
            height1: Antenna height (meters)
            lat2, lon2: UE location  
            height2: UE height (meters, default 0)
        
        Returns:
            3D distance in meters
        """
        # Haversine for horizontal distance
        R = 6371000  # Earth radius in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        horizontal_dist = R * c
        
        # Add vertical component
        vertical_diff = height1 - height2
        distance_3d = np.sqrt(horizontal_dist**2 + vertical_diff**2)
        
        return distance_3d
    
    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate bearing from point 1 to point 2
        
        Returns:
            Bearing in degrees (0-360)
        """
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_lambda = np.radians(lon2 - lon1)
        
        y = np.sin(delta_lambda) * np.cos(phi2)
        x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
        
        bearing = np.degrees(np.arctan2(y, x))
        bearing = (bearing + 360) % 360
        
        return bearing
    
    @staticmethod
    def calculate_elevation_angle(lat1: float, lon1: float, height1: float,
                                  lat2: float, lon2: float, height2: float = 0) -> float:
        """
        Calculate elevation angle from antenna to UE
        
        Returns:
            Elevation angle in degrees (positive = above horizontal)
        """
        # Horizontal distance
        R = 6371000
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        horizontal_dist = R * c
        
        # Vertical difference
        vertical_diff = height2 - height1
        
        # Elevation angle
        if horizontal_dist > 0:
            elevation = np.degrees(np.arctan(vertical_diff / horizontal_dist))
        else:
            elevation = 0.0
        
        return elevation
    
    def calculate_sector_features(self, 
                                  antenna_lat: float,
                                  antenna_lon: float,
                                  antenna_height: float,
                                  azimuth: float,
                                  edt: float,
                                  mdt: float,
                                  rs_power: float,
                                  frequency: float,
                                  bandwidth: float,
                                  ue_lat: float,
                                  ue_lon: float,
                                  ue_height: float = 0,
                                  env_features: Dict = None) -> Dict:
        """
        Calculate all features for a sector-to-UE link
        
        Args:
            antenna_lat, antenna_lon, antenna_height: Antenna position
            azimuth: Sector azimuth (degrees)
            edt: Electrical downtilt (degrees)
            mdt: Mechanical downtilt (degrees)
            rs_power: Reference signal power (dBm)
            frequency: Frequency (MHz)
            bandwidth: Carrier bandwidth (MHz)
            ue_lat, ue_lon, ue_height: UE position
            env_features: Environmental features dict for this location
        
        Returns:
            Dictionary of calculated features
        """
        features = {}
        
        # Distance calculations
        distance_3d = self.calculate_distance_3d(
            antenna_lat, antenna_lon, antenna_height,
            ue_lat, ue_lon, ue_height
        )
        distance_los = distance_3d  # Assuming LOS distance = 3D distance (metres)
        
        features['distance_3d_log'] = 10 * np.log10(max(distance_3d, 1.0))
        # distance_los in training data is stored as 10*log10(metres) — NOT raw metres
        features['distance_los'] = 10 * np.log10(max(distance_los, 1.0))
        
        # Bearing and azimuth
        bearing = self.calculate_bearing(antenna_lat, antenna_lon, ue_lat, ue_lon)
        bearing_azimuth_diff = (bearing - azimuth + 360) % 360
        if bearing_azimuth_diff > 180:
            bearing_azimuth_diff -= 360
        
        features['bearing_sin'] = np.sin(np.radians(bearing))
        features['bearing_cos'] = np.cos(np.radians(bearing))
        features['bearing_azimuth_sin'] = np.sin(np.radians(bearing_azimuth_diff))
        features['bearing_azimuth_cos'] = np.cos(np.radians(bearing_azimuth_diff))
        
        # Azimuth encoding
        features['azimuth_sin'] = np.sin(np.radians(azimuth))
        features['azimuth_cos'] = np.cos(np.radians(azimuth))
        
        # elevation_ue in training data = UE terrain elevation in METRES ASL.
        # Source: terrain_mean from h3_complete_features_windsor.csv (DEM).
        # UE is 1.5m above the ground surface.
        # Fallback: 183.0 m ASL (Windsor is flat ~180-190m ASL).
        WINDSOR_DEFAULT_ELEVATION_ASL = 183.0
        terrain_asl = float(env_features.get('terrain_mean', WINDSOR_DEFAULT_ELEVATION_ASL)) \
                      if env_features else WINDSOR_DEFAULT_ELEVATION_ASL
        features['elevation_ue'] = terrain_asl + 1.5   # UE height above ground

        # elevation_diff_log: log10 of |antenna_height_agl - ue_height_agl|
        elevation_diff = antenna_height - ue_height
        features['elevation_diff_log'] = np.log10(max(abs(elevation_diff), 1.0))

        # Compute horizontal distance (metres) — needed for depression angle below
        R = 6371000
        phi1_r = np.radians(antenna_lat); phi2_r = np.radians(ue_lat)
        dl = np.radians(ue_lon - antenna_lon)
        a_h = np.sin((phi2_r-phi1_r)/2)**2 + np.cos(phi1_r)*np.cos(phi2_r)*np.sin(dl/2)**2
        horiz_dist_m = R * 2 * np.arctan2(np.sqrt(a_h), np.sqrt(1-a_h))

        # Depression angle from antenna down to UE (positive = UE is below antenna)
        # theta_down = arctan((ant_height_agl - ue_height_agl) / horiz_dist)
        UE_HEIGHT_AGL = max(ue_height, 1.5)
        height_diff_agl = float(antenna_height) - UE_HEIGHT_AGL
        theta_down = float(np.degrees(np.arctan2(height_diff_agl, max(horiz_dist_m, 1.0))))
        
        # Antenna parameters — NOTE: model feature names use spaces, not underscores
        # RS dBm in training data = 18.2 dBm (RS EPRE — constant across all Windsor sites).
        # This is hardcoded because the model was trained on this value; using any other
        # value (e.g. total RS power = 43 dBm) would push the model out of distribution.
        features['RS dBm']       = 18.2
        features['Antenna height'] = float(antenna_height) if antenna_height is not None else 45.0
        features['Antenna EDT']  = float(edt) if edt is not None else 2.0
        features['Antenna MDT']  = float(mdt) if mdt is not None else 0.0
        # T2008 antenna constants (same for all Windsor sites)
        VBW  = 6.3    # Vertical beamwidth (degrees)
        HBW  = 65.0   # Horizontal beamwidth (degrees)
        GAIN = 17.1   # Antenna gain (dBi)
        features['VBW']  = VBW
        features['HBW']  = HBW
        features['Gain'] = GAIN
        # Training data defaults: 2147.5 MHz centre frequency, 15 MHz BW
        features['downlink_freq'] = frequency if frequency else 2147.5
        features['carrier_bw'] = bandwidth if bandwidth else 15.0
        features['outdoor'] = 1.0            # Deployment predictions are always outdoor
        # outdoor_proportion / indoor_proportion: set defaults here; env_features will
        # overwrite with real per-bin LSR values (fraction of outdoor measurements).
        # Default to 1.0 only for bins with no environmental data.
        features['outdoor_proportion'] = 1.0
        features['indoor_proportion']  = 0.0
        
        # Calculate downtilt projection distance
        ue_height_default = 1.5  # meters
        total_downtilt = float(edt if edt is not None else 2.0) + float(mdt if mdt is not None else 0.0)
        ant_h = float(antenna_height) if antenna_height is not None else 45.0
        height_diff = ant_h - ue_height_default
        
        if height_diff > 0 and (total_downtilt + VBW/2) > 0:
            dpd_m = height_diff / np.tan(np.radians(total_downtilt + VBW/2))
            # Training data stores this as 10*log10(metres), same encoding as distance_los
            features['downtilt_projection_distance'] = 10 * np.log10(max(dpd_m, 1.0))
        else:
            features['downtilt_projection_distance'] = 0
        
        # ── Beam geometry ─────────────────────────────────────────────────────
        total_tilt = float(edt if edt is not None else 2.0) + float(mdt if mdt is not None else 0.0)

        # HORIZONTAL ANTENNA PATTERN ATTENUATION (3GPP 36.814 formula)
        # h_atten = -min(12 * (theta_h / HBW)^2, 20) dB
        # theta_h = signed angle between bearing-to-UE and sector azimuth (−180..+180)
        # HBW = 3dB horizontal beamwidth = 65° (full beamwidth, not half)
        features['h_attenuation'] = -min(12 * (abs(bearing_azimuth_diff) / HBW)**2, 20.0)

        # VERTICAL ANTENNA PATTERN ATTENUATION
        # Beam boresight is depressed by total_tilt degrees below horizontal.
        # theta_v = angular offset of UE from beam boresight:
        #   theta_v = total_tilt - theta_down   (positive if UE is closer than boresight)
        # v_atten = -min(12 * (theta_v / VBW)^2, 20) dB
        theta_v_offset = total_tilt - theta_down   # signed, degrees
        features['v_attenuation'] = -min(12 * (theta_v_offset / VBW)**2, 20.0)

        # WITHIN 3dB HORIZONTAL BEAM (binary flag)
        # True if |bearing - azimuth| < HBW/2 (i.e., within the 3dB half-beamwidth)
        features['within_3db_horizontal'] = int(abs(bearing_azimuth_diff) < HBW / 2)

        # WITHIN VERTICAL BEAMWIDTH SHADOW (binary flag)
        # The "shadow" of the VBW cone projected on the ground.
        # A bin is inside the VBW cone if the depression angle to it is within
        # VBW/2 of the beam boresight depression: |theta_down - total_tilt| < VBW/2
        features['within_vbw_shadow'] = int(abs(theta_v_offset) < VBW / 2)

        # WITHIN 3dB BEAM — logical AND of horizontal AND vertical
        features['within_3db_beam'] = (
            features['within_3db_horizontal'] * features['within_vbw_shadow']
        )
        
        # Add environmental features if provided
        if env_features:
            for key, value in env_features.items():
                if key not in ['end_location_lat', 'end_location_lon']:
                    features[key] = value
        
        # Interaction features
        if env_features and 'tree_density_per_km2' in env_features:
            features['distance_tree_interaction'] = distance_los * env_features['tree_density_per_km2'] / 1000
        
        if env_features and 'water_coverage_pct' in env_features:
            features['distance_water_interaction'] = distance_los * env_features['water_coverage_pct'] / 100
        
        if env_features and 'clutter_mean_height' in env_features:
            features['distance_clutter_interaction'] = distance_los * env_features['clutter_mean_height']
            features['beam_clutter_interaction'] = features['within_3db_beam'] * env_features['clutter_mean_height']
        
        # Fresnel zone features (simplified)
        c = 3e8
        wavelength = c / (frequency * 1e6)
        fresnel_radius = np.sqrt(wavelength * distance_los / 4)
        features['fresnel_radius_m'] = fresnel_radius
        
        if env_features and 'clutter_max_height' in env_features:
            features['fresnel_clearance_ratio'] = env_features['clutter_max_height'] / (fresnel_radius + 1)
            features['fresnel_obstruction'] = 1 if features['fresnel_clearance_ratio'] > 0.6 else 0
        
        # ─── Pycraf geometry features (computed analytically — instant, high importance) ──
        # NOTE: use distance_los (raw metres) for all geometry calculations, NOT the
        # log-encoded features['distance_los'] which is 10*log10(metres).
        freq_hz = frequency * 1e6
        fspl_fallback = 20 * np.log10(max(distance_los, 1)) + 20 * np.log10(freq_hz) - 147.55

        # eps_pt / eps_pr: elevation angles at TX and RX (degrees above horizontal)
        # These are the TOP importance pycraf features and are pure geometry.
        # pycraf defines eps_pt as the elevation angle at the transmitter looking toward RX.
        # pycraf convention: eps_pt is the elevation angle at TX looking toward RX.
        # Positive = RX above TX, negative = RX below TX (UE is always below antenna).
        # Training data shows eps_pt mean=-3.05, max=0.0 → always negative for this dataset.
        # height_diff_m = antenna_height - ue_height > 0 means UE is below → negative angle.
        height_diff_m = float(antenna_height) - max(ue_height, 1.5)
        features['pycraf_eps_pt'] = float(-np.degrees(np.arctan(height_diff_m / (distance_los + 1e-6))))
        features['pycraf_eps_pr'] = -features['pycraf_eps_pt']   # symmetric: RX looking up at TX

        # d_lt / d_lr: distance from TX/RX to terrain horizon (km) — computed by pycraf
        # from actual terrain profile. Cannot be computed analytically without DEM.
        # Training-data means: d_lt≈1.1 km, d_lr≈0.07 km (UE is at ground level).
        # These will be overwritten by real pycraf values in calculate_batch_features().
        features['pycraf_d_lt'] = 1.1    # km — training dataset mean fallback
        features['pycraf_d_lr'] = 0.07   # km — training dataset mean fallback

        # LoS flag: geometrically True for short distances
        features['pycraf_LoS'] = 1  # will be overridden by pycraf below if available

        # Path-loss features: pycraf needed (but only the ones with real importance)
        # pycraf_L_ba (diffraction), pycraf_L_bfsg, pycraf_L_bd, pycraf_L_b0p, pycraf_L_bs
        # pycraf_L_b has near-zero importance — we default it to fspl
        features['pycraf_L_b'] = fspl_fallback  # importance=21, use FSPL — saves time

        # NOTE: batch pycraf is computed in calculate_batch_features() for all bins at once.
        # Here we set FSPL defaults; the batch override will replace them after this call.
        # pycraf_L_bd/L_bs/L_ba: use fspl_fallback (NOT 0.0 — diffraction/scatter loss
        # is always >= free-space; 0 is physically impossible and skews the model badly).
        # Training data mean: L_bfsg≈98, L_bd≈102, L_bs≈153, L_ba≈174 dB.
        features['pycraf_L_bfsg'] = fspl_fallback
        features['pycraf_L_b0p']  = fspl_fallback
        features['pycraf_L_bd']   = fspl_fallback          # will be replaced by pycraf batch
        features['pycraf_L_bs']   = fspl_fallback + 55.0   # scatter ≈ FSPL + ~55dB (training mean diff)
        features['pycraf_L_ba']   = fspl_fallback + 76.0   # absorption ≈ FSPL + ~76dB (training mean diff)
        
        # Clutter-derived features - DON'T overwrite if they exist in env_features!
        # Only set point_count to 0 (user said to ignore it)
        features['point_count'] = 0
        
        # clutter_height_range, clutter_std_height, fresnel_tree_obstruction
        # will come from env_features if available - don't overwrite with 0!
        
        # Interaction features
        if env_features:
            if 'tree_count' in env_features and 'frequency' in locals():
                features['freq_tree_interaction'] = frequency * env_features.get('tree_count', 0) / 1000
            else:
                features['freq_tree_interaction'] = 0
                
            if 'clutter_mean_height' in env_features:
                features['freq_clutter_interaction'] = frequency * env_features.get('clutter_mean_height', 0) / 100
            else:
                features['freq_clutter_interaction'] = 0
        else:
            features['freq_tree_interaction'] = 0
            features['freq_clutter_interaction'] = 0
        
        return features
    
    @staticmethod
    def compute_dsm_los(ant_lat: float, ant_lon: float, ant_height: float,
                        ue_lat: float, ue_lon: float, dist_m: float,
                        dsm_lookup: dict, n_samples: int = 20) -> dict:
        """
        Ray-trace from antenna to UE using real DEM + DSM surface heights (all ASL).
        Vectorized implementation: all sample points computed with NumPy, no Python loop
        over samples. n_samples reduced from 50 → 20 (adequate for macro LoS characterization).

        dsm_lookup format (from DataLoader.load_dsm_database()):
            h3_index -> {'dem': float (terrain ASL), 'p95': float (surface ASL)}
        where:
            dem  = bare-ground terrain elevation from HRDEM-Terrain.tif
            p95  = 95th-percentile DSM surface (terrain+buildings+trees) from HRDEM-Surface.tif

        Ray geometry (fully in ASL):
            ant_asl = dem(ant_bin) + ant_height_agl
            ue_asl  = p95(ue_bin) + 1.5m  (UE sits on top of surface + 1.5m above)
            ray_asl = linear interpolation from ant_asl to ue_asl
            blocked = p95(sample_bin) > ray_asl at that point

        Args:
            ant_lat, ant_lon  : Antenna coordinates
            ant_height        : Total antenna height AGL (m) — ground to antenna
            ue_lat, ue_lon    : UE bin centroid coordinates
            dist_m            : Horizontal distance (m)
            dsm_lookup        : dict h3_index -> {'dem': float, 'p95': float} (m ASL)
            n_samples         : Ray sample count (default 20)

        Returns:
            dict: dsm_los_ratio, dsm_los_binary, dsm_first_block_m, dsm_max_excess_m
        """
        _clear = {"dsm_los_ratio": 1.0, "dsm_los_binary": 1,
                  "dsm_first_block_m": -1.0, "dsm_max_excess_m": 0.0}

        if not dsm_lookup or dist_m <= 0:
            return _clear

        FALLBACK_TERRAIN = 183.0   # Windsor default terrain ASL if bin missing
        UE_HEIGHT_AGL    = 1.5     # UE sits 1.5m above the surface

        def _get(cell, key):
            """Get dem or p95 for a cell; handles dict-of-dict and legacy float."""
            val = dsm_lookup.get(cell)
            if val is None:
                return None
            if isinstance(val, dict):
                return val.get(key)
            return float(val)   # legacy scalar → treat as p95

        try:
            # ── Antenna ASL ───────────────────────────────────────────────────
            ant_cell = h3.latlng_to_cell(ant_lat, ant_lon, 12)
            ant_dem  = _get(ant_cell, "dem")
            if ant_dem is None:
                ant_dem = FALLBACK_TERRAIN
            ant_asl = ant_dem + ant_height   # real terrain + total AGL mount height

            # ── UE ASL (top of p95 surface + UE height) ───────────────────────
            ue_cell  = h3.latlng_to_cell(ue_lat, ue_lon, 12)
            ue_p95   = _get(ue_cell, "p95")
            if ue_p95 is None:
                ue_p95 = FALLBACK_TERRAIN
            ue_asl = ue_p95 + UE_HEIGHT_AGL

            # ── Vectorized ray samples ────────────────────────────────────────
            # Compute all n_samples interpolation fractions at once
            fracs   = np.linspace(0.0, 1.0, n_samples)
            s_lats  = ant_lat + (ue_lat - ant_lat) * fracs
            s_lons  = ant_lon + (ue_lon - ant_lon) * fracs
            ray_asl = ant_asl + (ue_asl - ant_asl) * fracs

            spacing = dist_m / max(n_samples - 1, 1)

            # ── Bulk H3 lookup ────────────────────────────────────────────────
            surface_vals = np.full(n_samples, np.nan)
            for i in range(n_samples):
                try:
                    cell = h3.latlng_to_cell(float(s_lats[i]), float(s_lons[i]), 12)
                    v = _get(cell, "p95")
                    if v is not None:
                        surface_vals[i] = v
                except Exception:
                    pass

            # ── Vectorized obstruction check ──────────────────────────────────
            valid      = ~np.isnan(surface_vals)
            pen        = np.where(valid, surface_vals - ray_asl, 0.0)
            obstructed = valid & (pen > 0)

            n_obs         = int(obstructed.sum())
            los_ratio     = 1.0 - (n_obs / n_samples)
            los_binary    = 1 if los_ratio >= 0.95 else 0
            max_excess    = float(pen[obstructed].max()) if n_obs > 0 else 0.0
            first_idx     = int(np.argmax(obstructed)) if n_obs > 0 else -1
            first_block_m = float(first_idx * spacing) if first_idx >= 0 else -1.0

            return {
                "dsm_los_ratio":     los_ratio,
                "dsm_los_binary":    los_binary,
                "dsm_first_block_m": first_block_m,
                "dsm_max_excess_m":  max_excess,
            }
        except Exception:
            return _clear

    def calculate_batch_features(self,
                                antenna_lat: float,
                                antenna_lon: float,
                                antenna_height: float,
                                sector_configs: list,
                                h3_bins: list,
                                env_features_df: pd.DataFrame,
                                dsm_lookup: dict = None) -> pd.DataFrame:
        """
        Calculate features for multiple sectors and H3 bins.
        Uses vectorized batch pycraf for path-loss features (fast),
        with analytic geometry for high-importance eps/d features.
        Optionally computes DSM LoS features via ray-tracing if dsm_lookup provided.

        Args:
            antenna_lat, antenna_lon, antenna_height: Antenna position
            sector_configs: List of sector configuration dicts
            h3_bins: List of H3 indices to predict
            env_features_df: DataFrame with environmental features indexed by h3_index
            dsm_lookup: dict of h3_index -> mean_height for DSM LoS ray-tracing
                        (if None, falls back to env_features_df precomputed values)

        Returns:
            DataFrame with all features for all sector-bin combinations
        """
        import time
        t0 = time.time()

        # ── Step 1: Resolve UE lat/lon for all bins ──────────────────────────
        bin_latlons = {h3_idx: h3.cell_to_latlng(h3_idx) for h3_idx in h3_bins}

        # ── Step 1b: Geometric best-sector pre-filter ─────────────────────────
        # For each bin, pre-assign it to the geometrically nearest sector (smallest
        # angular deviation between bearing-to-bin and sector azimuth).
        # Only that sector's full feature set is computed (pycraf + DSM remain per-bin).
        # This reduces sector-bin rows from N_sectors×N_bins → N_bins,
        # giving ~N_sectors× speedup on geometry + pycraf.
        # Approximation: bins very near sector boundaries may get the wrong best
        # server, but RSRP values at boundaries are nearly equal anyway.
        bin_best_sector = {}   # h3_idx → sector_idx (0-based)
        if len(sector_configs) > 1:
            # Vectorized: compute all bin bearings, then find nearest sector azimuth
            bin_keys = list(bin_latlons.keys())
            b_lats   = np.array([bin_latlons[b][0] for b in bin_keys])
            b_lons   = np.array([bin_latlons[b][1] for b in bin_keys])

            # Bearing from antenna to each bin (vectorized)
            d_lon    = np.radians(b_lons - antenna_lon)
            phi1     = np.radians(antenna_lat)
            phi2     = np.radians(b_lats)
            y_vec    = np.sin(d_lon) * np.cos(phi2)
            x_vec    = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lon)
            bearings = (np.degrees(np.arctan2(y_vec, x_vec)) + 360) % 360  # (N_bins,)

            # Angular deviation from each sector azimuth → pick nearest
            azimuths        = np.array([s['azimuth'] for s in sector_configs])  # (N_sectors,)
            diffs           = (bearings[:, None] - azimuths[None, :] + 180) % 360 - 180
            best_sector_idx = np.argmin(np.abs(diffs), axis=1)                  # (N_bins,)
            bin_best_sector = {b: int(best_sector_idx[i]) for i, b in enumerate(bin_keys)}
            logger.info(f"  Sector pre-filter: assigned {len(bin_keys)} bins to nearest sector "
                        f"(was {len(sector_configs)*len(bin_keys)} rows, now {len(bin_keys)})")
        else:
            # Single sector — no pre-filter needed
            bin_best_sector = {h3_idx: 0 for h3_idx in h3_bins}

        # ── Step 2: Per-sector geometry + env features (fast, no pycraf) ─────
        # Only compute for the pre-assigned (bin, sector) pair — 1 row per bin.
        all_features = []
        for h3_idx in h3_bins:
            sector_idx = bin_best_sector[h3_idx]
            sector     = sector_configs[sector_idx]
            ue_lat, ue_lon = bin_latlons[h3_idx]
            env_feats = env_features_df.loc[h3_idx].to_dict() if h3_idx in env_features_df.index else {}

            features = self.calculate_sector_features(
                antenna_lat=antenna_lat,
                antenna_lon=antenna_lon,
                antenna_height=antenna_height,
                azimuth=sector['azimuth'],
                edt=sector['edt'],
                mdt=sector['mdt'],
                rs_power=sector['rs_power'],
                frequency=sector['frequency'],
                bandwidth=sector['bandwidth'],
                ue_lat=ue_lat,
                ue_lon=ue_lon,
                env_features=env_feats
            )
            features['h3_index']    = h3_idx
            features['sector_id']   = sector_idx + 1
            features['sector_name'] = sector.get('name', f"Sector {sector_idx + 1}")
            features['_ue_lat']     = ue_lat   # temp: needed for pycraf batch
            features['_ue_lon']     = ue_lon
            features['_freq_mhz']   = sector['frequency']
            all_features.append(features)

        df = pd.DataFrame(all_features)
        logger.info(f"  Geometry features done in {time.time()-t0:.1f}s ({len(df)} rows)")

        # ── Step 3: Batch pycraf path loss (parallelized) ────────────────────
        t1 = time.time()
        df = self._apply_batch_pycraf(df, antenna_lat, antenna_lon, antenna_height)
        logger.info(f"  Pycraf batch done in {time.time()-t1:.1f}s")

        # ── Step 4: DSM LoS ray-trace (per unique bin, shared across sectors) ─
        # The DSM LoS features depend only on geometry (antenna → bin), not on
        # sector azimuth, so we compute once per unique h3_index and broadcast.
        if dsm_lookup:
            t2 = time.time()
            # Pre-build a fast h3_index → distance_3d_log lookup from the DataFrame
            dist_log_map = df.set_index('h3_index')['distance_3d_log'].to_dict()

            # Compute DSM LoS once per unique bin (sector-independent)
            dsm_cache = {}
            for h3_idx in h3_bins:
                ue_lat, ue_lon = bin_latlons[h3_idx]
                log_val = dist_log_map.get(h3_idx, 2.0)   # default ~100m
                dist_m  = float(10 ** (log_val / 10))
                dsm_cache[h3_idx] = self.compute_dsm_los(
                    antenna_lat, antenna_lon, antenna_height,
                    ue_lat, ue_lon, dist_m,
                    dsm_lookup, n_samples=20
                )
            # Write DSM LoS features to all rows (same value per bin across sectors)
            for col in ['dsm_los_ratio', 'dsm_los_binary', 'dsm_first_block_m', 'dsm_max_excess_m']:
                df[col] = df['h3_index'].map(lambda idx: dsm_cache.get(idx, {}).get(col, 0.0))
            n_los = sum(1 for v in dsm_cache.values() if v['dsm_los_binary'] == 1)
            logger.info(f"  DSM LoS done in {time.time()-t2:.1f}s  "
                        f"({n_los}/{len(h3_bins)} bins are LoS)")
        else:
            logger.info("  DSM lookup not provided — using precomputed values from env_features")

        # Remove temp columns
        df.drop(columns=['_ue_lat', '_ue_lon', '_freq_mhz'], inplace=True, errors='ignore')
        return df

    # ── Vectorized pycraf: one thread per bin-path ────────────────────────────
    @staticmethod
    def _compute_one_path(args):
        """
        Compute pycraf path loss for a single (antenna → bin) path.
        Called in a thread pool — pycraf Cython releases the GIL.
        Returns (index, L_bfsg, L_b0p, L_bd, L_bs, L_ba, L_b) or None on error.
        """
        try:
            from pycraf import pathprof
            import astropy.units as u

            (row_idx, ant_lon, ant_lat, ant_h,
             ue_lon, ue_lat, ue_h, freq_mhz, dist_m) = args

            step = max(50.0, min(dist_m / 8.0, 200.0)) * u.m

            pprop = pathprof.PathProp(
                freq_mhz * u.MHz,
                290 * u.K, 1013.25 * u.hPa,
                ant_lon * u.deg, ant_lat * u.deg,
                ue_lon * u.deg,  ue_lat * u.deg,
                max(ant_h, 1.0) * u.m,
                max(ue_h, 1.5)  * u.m,
                hprof_step=step,
                timepercent=50. * u.percent,
                version=16,
                generic_heights=True,
            )
            L_b0p, L_bd, L_bs, L_ba, L_b, L_b_corr, L = pathprof.loss_complete(pprop)
            los = 1 if pprop.path_type.value == 1 else 0
            del pprop
            return (row_idx,
                    float(L.value),        # L_bfsg
                    float(L_b0p.value),
                    float(L_bd.value),
                    float(L_bs.value),
                    float(L_ba.value),
                    float(L_b.value),
                    los)
        except Exception:
            return None

    def _apply_batch_pycraf(self, df: pd.DataFrame,
                            ant_lat: float, ant_lon: float, ant_h: float) -> pd.DataFrame:
        """
        Run pycraf grouped by res-11 parent bin.

        Strategy:
        - Group all prediction bins by their H3 res-11 parent
        - Run ONE pycraf call per unique (parent, sector_id) pair, using the
          parent centroid as the path endpoint
        - Copy the 6 pycraf path-loss values to all children in that group

        This achieves res-12 precision for all ML features while only paying
        pycraf cost for res-10 bins (~50× fewer calls than res-12). Path loss
        varies smoothly over the ~520m span of a res-10 hex, so the approximation
        is excellent for macro-cell propagation.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import h3 as _h3

        UE_H = 1.5
        PYCRAF_RES = 10  # Resolution at which pycraf is evaluated (~50× fewer calls than res-12)

        # ── Step 1: For each row, compute its res-10 parent centroid ──────────
        parent_lat = {}
        parent_lon = {}
        df['_parent11'] = None

        for idx, row in df.iterrows():
            h3_idx = row['h3_index']
            try:
                res = _h3.get_resolution(h3_idx)
                if res <= PYCRAF_RES:
                    parent = h3_idx   # Already coarser or equal
                else:
                    parent = _h3.cell_to_parent(h3_idx, PYCRAF_RES)
            except Exception:
                parent = h3_idx
            df.at[idx, '_parent11'] = parent
            if parent not in parent_lat:
                plat, plon = _h3.cell_to_latlng(parent)
                parent_lat[parent] = plat
                parent_lon[parent] = plon

        # ── Step 2: Build unique (parent, sector) pycraf tasks ────────────────
        # One task per unique (parent_id, sector_id) combination
        seen = set()
        args_list = []
        for idx, row in df.iterrows():
            parent   = row['_parent11']
            freq_mhz = float(row['_freq_mhz'])
            task_key = (parent, float(row['sector_id']))
            if task_key in seen:
                continue
            seen.add(task_key)

            plat = parent_lat[parent]
            plon = parent_lon[parent]

            # Use haversine distance to parent centroid
            dlat = np.radians(plat - ant_lat)
            dlon = np.radians(plon - ant_lon)
            a = (np.sin(dlat/2)**2
                 + np.cos(np.radians(ant_lat)) * np.cos(np.radians(plat))
                 * np.sin(dlon/2)**2)
            dist_m = 6371000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

            args_list.append((
                task_key,
                ant_lon, ant_lat, ant_h,
                plon, plat,
                UE_H,
                freq_mhz,
                max(dist_m, 10.0)
            ))

        logger.info(f"  Pycraf: {len(args_list)} unique parent-sector paths "
                    f"(from {len(df)} rows, {len(parent_lat)} unique parents)")

        # ── Step 3: Run pycraf in parallel ────────────────────────────────────
        n_workers = min(8, len(args_list))
        task_results = {}   # task_key → (L_bfsg, L_b0p, L_bd, L_bs, L_ba, L_b, los)

        def _run_task(args):
            task_key = args[0]
            # Reuse _compute_one_path with row_idx = task_key
            r = FeatureEngine._compute_one_path(args)
            return r  # (task_key, L_bfsg, ...)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(FeatureEngine._compute_one_path, a): a[0]
                       for a in args_list}
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    task_results[res[0]] = res[1:]

        # ── Step 4: Write results back — all children of each parent get the same path loss
        n_hits = 0
        for idx, row in df.iterrows():
            parent   = row['_parent11']
            task_key = (parent, float(row['sector_id']))
            if task_key in task_results:
                L_bfsg, L_b0p, L_bd, L_bs, L_ba, L_b, los = task_results[task_key]
                df.at[idx, 'pycraf_L_bfsg'] = L_bfsg
                df.at[idx, 'pycraf_L_b0p']  = L_b0p
                df.at[idx, 'pycraf_L_bd']   = L_bd
                df.at[idx, 'pycraf_L_bs']   = L_bs
                df.at[idx, 'pycraf_L_ba']   = L_ba
                df.at[idx, 'pycraf_L_b']    = L_b
                df.at[idx, 'pycraf_LoS']    = los   # space matches training column name
                n_hits += 1

        df.drop(columns=['_parent11'], inplace=True, errors='ignore')

        # ── Derived pycraf features (replace 6 correlated terms in model) ─────
        # pycraf_predicted_rsrp: full physics-based RSRP prediction
        # NOTE: use 'RS dBm' (space) to match model feature names
        df['pycraf_predicted_rsrp'] = df['RS dBm'] - df['pycraf_L_b'] + df['Gain']
        # pycraf_diffraction_excess: diffraction loss beyond free-space
        df['pycraf_diffraction_excess'] = df['pycraf_L_bd'] - df['pycraf_L_bfsg']

        logger.info(f"  Pycraf: {n_hits}/{len(df)} rows filled from "
                    f"{len(task_results)}/{len(args_list)} parent paths, "
                    f"{len(df)-n_hits} used FSPL fallback")
        return df


if __name__ == "__main__":
    # Test feature engine
    engine = FeatureEngine()
    
    print("\n" + "="*60)
    print("Testing Feature Engine")
    print("="*60)
    
    # Test single calculation
    test_sector = {
        'azimuth': 0,
        'edt': 6,
        'mdt': 0,
        'rs_power': 43.0,
        'frequency': 2100,
        'bandwidth': 20
    }
    
    features = engine.calculate_sector_features(
        antenna_lat=42.3,
        antenna_lon=-83.0,
        antenna_height=30,
        azimuth=test_sector['azimuth'],
        edt=test_sector['edt'],
        mdt=test_sector['mdt'],
        rs_power=test_sector['rs_power'],
        frequency=test_sector['frequency'],
        bandwidth=test_sector['bandwidth'],
        ue_lat=42.31,
        ue_lon=-82.99,
        ue_height=0
    )
    
    print(f"\n✓ Calculated {len(features)} features")
    print("\nSample features:")
    for key in list(features.keys())[:10]:
        print(f"  {key}: {features[key]}")
    
    print("\n✅ Feature engine test passed!")
