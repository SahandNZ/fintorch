from ._feature_transform import FeatureTransform
from ._rmstd_tr_roc import RollingMeanStdTrRocFeatureTransform
from ._stft_tr_roc import StftTrRocFeatureTransform
from ._technical_analysis import TechnicalAnalysisFeatureTransform

FEATURE_TRANSFORM_TYPES = [
    RollingMeanStdTrRocFeatureTransform,
    TechnicalAnalysisFeatureTransform,
    StftTrRocFeatureTransform,
]
