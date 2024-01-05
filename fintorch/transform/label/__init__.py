from fintorch.transform.label.forward_backward_min import ForwardBackwardMinimumLabelTransform
from fintorch.transform.label.forward_ichimoku import ForwardIchimokuLabelTransform
from fintorch.transform.label.forward_middle_sma import ForwardMiddleSmaLabelTransform
from fintorch.transform.label.forward_roc import ForwardRocLabelTransform
from fintorch.transform.label.next_fractal import NextFractalLabelTransform
from fintorch.transform.label.up_down import UpDownLabelTransform

LABEL_TRANSFORMS = [
    ForwardBackwardMinimumLabelTransform(),
    ForwardIchimokuLabelTransform(),
    ForwardMiddleSmaLabelTransform(),
    ForwardRocLabelTransform(),
    NextFractalLabelTransform(),
    UpDownLabelTransform()
]
