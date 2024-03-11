from ._feature_transform import FeatureTransform
from ._rmstd_tr_roc import RollingMeanStdTrRocFeatureTransform
from ._stft_tr_roc import StftTrRocFeatureTransform
from ._technical_trend_indicators import TechnicalTrendIndicatorsFeatureTransform

FEATURE_TRANSFORM_TYPES = [
    RollingMeanStdTrRocFeatureTransform,
    TechnicalTrendIndicatorsFeatureTransform,
    StftTrRocFeatureTransform,
]
