# Plan: Weekly BigQuery Bad IMSI Bins → Map Layer

## Status: App code DONE — BigQuery scheduled query pending setup

---

## Architecture

```
BigQuery: bi-srv-cte-pga-pr-eb8a61.netscout.lsr_panda_weekly_qk
    → Scheduled Query (exports quadbin + bad_ce_ims as CSV, weekly)
    → GCS: gs://windsor-rf-ml-data/weekly_bad_imsis   (overwritten weekly)
    → Cloud Run: downloads at startup, converts quadbin → H3 res-12 in Python
    → Local CSV (data/bad_bins_wnd.csv): fallback if GCS unavailable
```

---

## Step 1 — BigQuery Scheduled Query (one-time GCP setup)

Run this in BigQuery console, then schedule weekly (e.g. Monday 02:00).

**Source table:** `bi-srv-cte-pga-pr-eb8a61.netscout.lsr_panda_weekly_qk`

**Filter logic:** `imsi_count_below118 >= 100 AND perc_bad_sessions >= 25 AND strat_check = 'Windsor'`

```sql
EXPORT DATA OPTIONS (
  uri       = 'gs://windsor-rf-ml-data/weekly_bad_imsis',
  format    = 'CSV',
  overwrite = true,
  header    = true
) AS
SELECT
  quadbin,
  imsi_count_below118 AS bad_ce_ims
FROM `bi-srv-cte-pga-pr-eb8a61.netscout.lsr_panda_weekly_qk`
WHERE imsi_count_below118 >= 100
  AND perc_bad_sessions   >= 25
  AND strat_check          = 'Windsor';
```

**IAM requirement:** The BigQuery service account must have `Storage Object Creator`
on the `windsor-rf-ml-data` GCS bucket to run the EXPORT DATA statement.

**Schedule:** Weekly, e.g. Monday 02:00 UTC — set up in BigQuery → Scheduled Queries.

---

## Step 2 — App code changes (DONE)

### `site_deployment_demo/app.py` — `_load_bad_bins()` (lines 672–716)

- **GCS-first**: always tries `gs://windsor-rf-ml-data/bad_bins_wnd.csv` at startup
- **Quadbin conversion**: if downloaded CSV has a `quadbin` column (BQ export format),
  converts quadbin → lat/lon (via `quadbin` package) → H3 res-12 (via `h3.latlng_to_cell`)
- **Local fallback**: falls back to `data/bad_bins_wnd.csv` (existing H3 format) if GCS fails

### `site_deployment_demo/requirements.txt` + `requirements.txt`
- Added `quadbin>=0.3` (Carto Python package for quadbin → center-point conversion)

No other files need changes. Frontend, `/api/bad_imsi_bins`, and `_covered_bad_imsis()` are unchanged.

---

## Step 3 — Verification

1. Run the EXPORT DATA query manually once in BQ console
2. Confirm file exists: `gsutil ls gs://windsor-rf-ml-data/bad_bins_wnd.csv`
3. Confirm file has `quadbin,bad_ce_ims` header
4. Restart Cloud Run → check logs for: `✓ Downloaded bad_bins_wnd.csv from GCS (weekly data)`
5. Check log line: `✓ Bad IMSI bins loaded: X bins  total=Y IMSIs  max=Z`
6. Enable Bad IMSI Bins layer on map → confirm bins appear and counts look correct
7. Set up BigQuery scheduled query (weekly)

---

## Notes

- The local `data/bad_bins_wnd.csv` (H3 format) stays as permanent fallback — no need to delete it
- Weekly GCS overwrite means Docker never needs rebuilding for data refreshes
- The `quadbin` → H3 conversion runs once at startup (~1–2 seconds for ~36k bins)
- Adding more regions later: just add another `strat_check = 'Montreal'` export to a separate GCS path
