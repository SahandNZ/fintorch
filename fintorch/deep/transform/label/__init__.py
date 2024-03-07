from ._forward_backward_min import ForwardBackwardMinimumLabelTransform
from ._forward_ichimoku import ForwardIchimokuLabelTransform
from ._forward_middle_sma import ForwardMiddleSmaLabelTransform
from ._forward_roc import ForwardRocLabelTransform
from ._label_transform import LabelTransform
from ._next_fractal import NextFractalLabelTransform
from ._triple_barrier import TripleBarrierLabelTransform
from ._up_down import UpDownLabelTransform

LABEL_TRANSFORM_TYPES = [
    ForwardBackwardMinimumLabelTransform,
    ForwardIchimokuLabelTransform,
    ForwardMiddleSmaLabelTransform,
    ForwardRocLabelTransform,
    NextFractalLabelTransform,
    TripleBarrierLabelTransform,
    UpDownLabelTransform,
]
