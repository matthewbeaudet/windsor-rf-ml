# 🗼 Site Deployment Tool - Windsor LTE Network

Interactive web-based tool for testing and evaluating potential cell site locations using ML-powered RF predictions.

## 🎯 Overview

This tool allows RF engineers to:
- **Click anywhere** on an interactive map to test a potential site location
- **See immediate predictions** of coverage improvements (bins with ≥3dB improvement)
- **Visualize impact** with color-coded hexagonal bins
- **Compare locations** quickly without complex RF planning software

### Key Features:
- ✅ ML-powered predictions using trained LightGBM model
- ✅ Pycraf path loss calculations for accuracy
- ✅ 3-sector site configuration (0°, 120°, 240°)
- ✅ Best-server selection vs existing 46-site network
- ✅ Interactive OpenStreetMap interface
- ✅ Real-time statistics and metrics
- ✅ Fast predictions (10-30 seconds)

---

## 📁 Project Structure

```
site_deployment_demo/
├── app.py                          # Flask backend server
├── api/
│   ├── __init__.py
│   └── site_predictor.py          # Prediction engine wrapper
├── templates/
│   └── index.html                  # Interactive map interface
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Navigate to the demo folder
cd "G:/My Drive/Windsor/Mode Server (Original)/site_deployment_demo"

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Server

```bash
# Start Flask development server
python app.py
```

You should see:
```
Site Deployment Tool - Starting Flask Server
============================================================

Server will be available at: http://localhost:5000
Open this URL in your web browser to use the tool
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

---

## 💡 How to Use

### Basic Workflow:

1. **Open the tool** in your web browser
2. **Click anywhere** on the Windsor map
3. **Wait 10-30 seconds** for predictions to calculate
4. **View results**:
   - Green hexagons: >6dB improvement
   - Yellow hexagons: 4-6dB improvement
   - Orange hexagons: 3-4dB improvement
5. **Check statistics** in the right panel:
   - Number of bins improved
   - Mean/max improvement
   - Site footprint size
6. **Click "Clear Results"** to test another location
7. **Repeat** to compare different locations

### Understanding the Results:

- **Bins Improved**: Number of hexagonal bins with ≥3dB RSRP improvement
- **Mean Improvement**: Average improvement across all improved bins
- **Site Footprint**: Number of bins where new site becomes best server
- **Total Bins Predicted**: All bins within 2km prediction radius

---

## 🔧 Technical Details

### Site Configuration:
- **Antenna Height**: 30 meters (default)
- **Sectors**: 3 sectors at 0°, 120°, 240° azimuths
- **Electrical Downtilt**: 6°
- **RS Power**: 43 dBm
- **Frequency**: 2100 MHz
- **Bandwidth**: 20 MHz

### Prediction Process:
1. User clicks location on map
2. System finds all H3 bins within 2km radius
3. ML model predicts RSRP for 3 sectors
4. Pycraf calculates path loss features
5. Best-server selection (max RSRP per bin)
6. Compare vs baseline 46-site network
7. Identify bins with ≥3dB improvement
8. Display results on map

### Performance:
- **Prediction Radius**: 2 km (configurable)
- **Bins per Prediction**: ~1,000-5,000
- **Prediction Time**: 10-30 seconds
- **Baseline Network**: 46 existing sites

---

## 📊 Data Requirements

The tool requires these files in the parent directory:

### Required Files:
1. **`comprehensive_rsrp_all_46_sites.csv`**
   - Baseline network predictions
   - Contains: h3_index, predicted_rsrp, tower_id
   
2. **`h3_complete_features_windsor.csv`**
   - Environmental features for all H3 bins
   - Contains: clutter heights, tree data, water coverage, etc.
   
3. **`Model/lgbm_binned_model.joblib`**
   - Trained ML model
   
4. **`Model/Residual/lgbm_binned_model.joblib`**
   - Residual model

### Existing Modules Used:
- `rf_design_tool/modules/data_loader.py`
- `rf_design_tool/modules/prediction_engine.py`
- `rf_design_tool/modules/feature_engine.py`

---

## 🎨 User Interface

### Map Controls:
- **Zoom**: Mouse wheel or +/- buttons
- **Pan**: Click and drag
- **Click**: Test site location
- **Popup**: Click hexagons for details

### Color Legend:
- 🟢 **Green**: >6 dB improvement (Excellent)
- 🟡 **Yellow**: 4-6 dB improvement (Very Good)
- 🟠 **Orange**: 3-4 dB improvement (Good)

### Info Panel:
- Instructions
- Real-time statistics
- Clear results button
- Network configuration details

---

## 🔍 API Endpoints

### `GET /`
Returns the interactive map interface

### `POST /api/predict_site`
Predicts coverage for a site location

**Request Body:**
```json
{
    "lat": 42.3149,
    "lon": -83.0364,
    "height": 30,      // Optional, default 30m
    "radius": 2000     // Optional, default 2000m
}
```

**Response:**
```json
{
    "geojson": {
        "type": "FeatureCollection",
        "features": [...]
    },
    "stats": {
        "total_bins": 1523,
        "improved_bins": 245,
        "mean_improvement": 4.2,
        "max_improvement": 8.7,
        "site_footprint_bins": 189
    },
    "site_location": {
        "lat": 42.3149,
        "lon": -83.0364
    }
}
```

### `GET /api/health`
Health check endpoint

---

## 🚧 Future Enhancements

### Phase 2 (Planned):
- [ ] Draw search ring/polygon
- [ ] Upload poor IMSI CSV data
- [ ] Find tallest buildings in search ring
- [ ] Load competitor tower database
- [ ] Compare multiple candidate sites
- [ ] Rank top 4 candidates
- [ ] Export results to CSV/KML

### Phase 3 (Planned):
- [ ] Adjust site parameters (height, azimuth, power)
- [ ] Multi-site deployment analysis
- [ ] Cost-benefit analysis
- [ ] Integration with Atoll
- [ ] Historical comparison

---

## 🐛 Troubleshooting

### Issue: Server won't start
**Solution**: Check if port 5000 is already in use
```bash
# Windows: Find process using port 5000
netstat -ano | findstr :5000

# Kill the process
taskkill /PID <process_id> /F
```

### Issue: Predictions take too long
**Solution**: Reduce prediction radius in the code:
```python
# In templates/index.html, line ~210
radius: 1000  // Change from 2000 to 1000
```

### Issue: No improvements shown
**Solution**: Try clicking in areas with poorer existing coverage (farther from existing sites)

### Issue: Import errors
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

---

## 📝 Notes

- This is an **MVP demo** for rapid site evaluation
- Not intended to replace detailed RF planning tools like Atoll
- **Purpose**: Screen 100+ candidates down to top 4 for detailed analysis
- Predictions are ML-based estimates, not ray-tracing accuracy
- Best used for **comparative analysis** between locations

---

## 👥 Usage Workflow

### Typical Engineering Process:

1. **Identify coverage gap** (e.g., from poor IMSI reports)
2. **Use this tool** to test 10-20 potential locations
3. **Compare results** - which location improves most bins?
4. **Select top 4** candidates based on:
   - Bins improved (≥3dB)
   - Site footprint
   - Practical considerations (access, colocation)
5. **Export candidates** for detailed Atoll analysis
6. **Final selection** based on comprehensive planning

### Time Savings:
- **Before**: 1-2 hours per candidate in Atoll
- **With This Tool**: 30 seconds per candidate
- **Benefit**: Quickly eliminate poor candidates, focus on best options

---

## 📧 Support

For issues or questions, contact the RF Engineering team.

---

**Built with**: Flask, Leaflet.js, OpenStreetMap, LightGBM, Pycraf
**Version**: 1.0.0 MVP
**Last Updated**: February 2026
