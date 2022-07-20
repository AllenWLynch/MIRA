from mira.topic_model.accessibility_model import AccessibilityModel
from mira.topic_model.expression_model import ExpressionModel
from mira.topic_model.covariate_model import CovariateModel
from mira.topic_model.dirichlet_process import \
        ExpressionDirichletProcessModel, AccessibilityDirichletProcessModel
from mira.topic_model.dirichlet_model import \
        ExpressionDirichletModel, AccessibilityDirichletModel
from mira.topic_model.base import BaseModel, logger
import numpy as np

def TopicModel(
    n_samples, n_features,
    endogenous_key = None,
    exogenous_key = None,
    counts_layer = None,
    covariates_keys = None,
    extra_features_keys = None,
    latent_space = 'dirichlet',
    feature_type = 'expression',
    automatically_set_params = True,
    **kwargs,
):

    assert(latent_space in ['dp', 'dirichlet'])
    assert(feature_type in ['expression','accessibility'])

    if latent_space == 'dp' and n_samples < 40000:
        logger.warn(
            'The dirichlet process model is intended for atlas-level experiments.\n'
            'For smaller datasets, please use the "dirichlet" latent space, and use the tuner'
            'to find the optimal number of topics.'
        )
        
        latent_space = 'dirichlet-process'

    basename = 'model'
    if isinstance(covariates_keys, (list, np.ndarray)) \
            and len(covariates_keys) > 0:
        basename = 'covariate-model'
        baseclass = CovariateModel
    else:
        baseclass = BaseModel

    if feature_type == 'expression':
        feature_model = ExpressionModel
    elif feature_type == 'accessibility':
        feature_model = AccessibilityModel

    generative_map = {
        ('expression','dirichlet') : ExpressionDirichletModel,
        ('expression','dirichlet-process') : ExpressionDirichletProcessModel,
        ('accessibility', 'dirichlet') : AccessibilityDirichletModel,
        ('accessibility', 'dirichlet-process') : AccessibilityDirichletProcessModel,
    }

    generative_model = generative_map[(feature_type, latent_space)]

    _class = type(
        '_'.join([latent_space, feature_type, basename]),
        (generative_model, feature_model, baseclass),
        {}
    )

    instance = _class(
        endogenous_key = endogenous_key,
        exogenous_key = exogenous_key,
        counts_layer = counts_layer,
        covariates_keys = covariates_keys,
        extra_features_keys = extra_features_keys,
    )

    if automatically_set_params:
        instance.set_params(
            **instance.recommend_parameters(n_samples, n_features)
        )

    instance.set_params(**kwargs)

    return instance