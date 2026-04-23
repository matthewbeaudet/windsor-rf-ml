"""
MontrealRouter — dual-model wrapper for the Montreal Urban/Suburban split.

Presents the same .booster_.predict(X) interface as a single LGBMRegressor
so it drops into PredictionEngine without modification.

Usage:
    router = MontrealRouter(urban_model, suburban_model, urban_poly)
    router.select_model(antenna_lat, antenna_lon)   # once per site click
    predictions = router.booster_.predict(X)        # called by PredictionEngine
"""

from shapely.geometry import Point


class MontrealRouter:

    class _Booster:
        def __init__(self, router):
            self._r = router

        def predict(self, X):
            return self._r._active_model.booster_.predict(X)

    def __init__(self, urban_model, suburban_model, urban_poly):
        self._urban    = urban_model
        self._suburban = suburban_model
        self._poly     = urban_poly
        self._active_model = suburban_model   # safe default until select_model is called
        self.booster_  = self._Booster(self)
        # Expose feature names from the urban model (both share the same feature list)
        self.feature_name_ = urban_model.feature_name_

    def select_model(self, antenna_lat: float, antenna_lon: float) -> str:
        """
        Choose Urban or Suburban model based on antenna location.
        Call once per prediction request before the engine runs predict().
        Returns 'urban' or 'suburban' for logging.
        """
        if self._poly.contains(Point(antenna_lon, antenna_lat)):
            self._active_model = self._urban
            return 'urban'
        self._active_model = self._suburban
        return 'suburban'
