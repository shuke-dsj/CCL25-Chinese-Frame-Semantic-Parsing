"""Central Task2 model construction used by train, analysis and prediction."""

from .model import Model as BaselineModel
from .model_boundary_auxiliary import BoundaryAuxiliaryModel
from .model_content_aware import ContentAwareModel
from .model_frame_prototype import FramePrototypeConditionedModel
from .model_target_conditioned import TargetConditionedModel


def build_task2_model(config, variant='baseline'):
    if variant == 'baseline':
        return BaselineModel(config)
    if variant == 'target_conditioned':
        return TargetConditionedModel(config)
    if variant == 'content_aware':
        return ContentAwareModel(config)
    if variant == 'frame_prototype_conditioned':
        return FramePrototypeConditionedModel(config)
    if variant == 'boundary_auxiliary':
        return BoundaryAuxiliaryModel(config)
    raise ValueError(f'unknown Task2 model variant: {variant}')
