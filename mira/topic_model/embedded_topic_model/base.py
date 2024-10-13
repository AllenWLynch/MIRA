import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from mira.topic_model.base import encoder_layer, ConcatLayer, _mask_drop
from mira.topic_model.base import logger
from mira.topic_model.modality_mixins.accessibility_encoders import LSIEncoder, apply_dan
from mira.topic_model.embedded_topic_model.lsi_embed import PointwiseMutualInfoTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

# CLIFF: 
# New implementation of the embedded topic model
# The `ProjectionDecoder` class is the decoder model which takes a fixed embedding matrix.
# 
# In this current implementation,
# I am passing an hidden layer output from the encoder to the batch effect model as well as the topic compositions (intermediate).
# I did this because I thought there could be some technical signal in the counts which is not explained by only cell type 
# and batch label - think of an orthogonal stochastic process - like in some cells the Tn5 transposase acts a little differently.
# With this change, the encoder can pass some information directly to the batch effect model, bypassing the biological latent space.
# I only tried this on one dataset and it didn't seem to make a difference, but it could be worth exploring.
# 
# Essentially, the peak embeddings cannot be trained because the inner products take too much time to compute, so 
# whichever embeddings are chosen should be pretty good - I think concatenating the Cistrome and dataset-specific embeddings could
# work pretty well.
#
# Another difference between what I've implemented and the Blei paper is that I'm using "ProdEmbeddedTopicModel" - e.g.
# I do the projection in log-space, sum across the topics, then do the softmax. This is inline with the original implementation
# and the ProdLDA paper. I did it this way mostly for consistency, but I think there are some compelling reasons to favor 
# the ProdLDA approach. I can outline some geometric reasons in the future.
#
class ProjectionDecoder(nn.Module):
    
    def __init__(self, 
        covariates_hidden = 32,
        covariates_dropout = 0.05, 
        mask_dropout = 0.05,
        *,
        num_exog_features, 
        num_topics, 
        num_covariates, 
        topics_dropout, 
        embedding_matrix,
        is_trainable, # you don't need to train the decoder embeddings so ignore.
    ):
        super().__init__()

        assert embedding_matrix.shape[1] == num_exog_features
        projection_dim = embedding_matrix.shape[0]

        self.beta = nn.Linear(num_topics, projection_dim, bias = False)
        self.bn = nn.BatchNorm1d(projection_dim)
        self.overall_bn = nn.BatchNorm1d(num_exog_features)

        self.drop1 = nn.Dropout(covariates_dropout)
        self.drop2 = nn.Dropout(topics_dropout)
        self.mask_drop = partial(_mask_drop, dropout_rate = mask_dropout)

        self.num_topics = num_topics
        self.num_covariates = num_covariates

        self.projection_matrix = nn.Parameter(torch.Tensor(embedding_matrix)).requires_grad_(False)
        
        if num_covariates > 0:

            self.batch_effect_model = nn.Sequential(
                ConcatLayer(1),
                encoder_layer(
                    num_topics + num_covariates, 
                    covariates_hidden, 
                    dropout=covariates_dropout, 
                    nonlin=True
                ),
                nn.Linear(covariates_hidden, projection_dim),
                nn.BatchNorm1d(projection_dim, affine = False),
            )

            self.batch_effect_gamma = nn.Parameter(
                torch.zeros(projection_dim)
            )


    @property
    def is_correcting(self):
        return self.num_covariates > 0


    def forward(self, theta, covariates, intermediate, nullify_covariates = False):
        
        X1 = self.drop1(theta)
        X2 = self.drop2(theta)
        
        if self.is_correcting:
            
            self.covariate_signal = self._get_batch_effect_lowdim(X1, covariates, 
                nullify_covariates = nullify_covariates)

            self.biological_signal = self._get_biological_effect_lowdim(X1)

        return F.softmax(
                    self.project(
                        self._get_biological_effect_lowdim(X2) + \
                        self.mask_drop(
                            self._get_batch_effect_lowdim(X2, covariates, nullify_covariates = nullify_covariates), 
                            training = self.training
                        )
                    ),
                    dim=1
                )
    

    def project(self, X):
        return self.overall_bn(X @ self.projection_matrix)

    def get_topic_activations(self):
        X = torch.eye(self.num_topics).to(self.beta.weight.device)
        return self.get_biological_effect(X).detach().cpu().numpy()

    def _get_biological_effect_lowdim(self, theta):
        return self.bn(self.beta(theta))

    def _get_batch_effect_lowdim(self, theta, covariates, nullify_covariates = False):
        
        if not self.is_correcting or nullify_covariates: 
            batch_effect = theta.new_zeros(1)
            batch_effect.requires_grad = False
        else:
            batch_effect = self.batch_effect_gamma * self.batch_effect_model(
                    (theta, covariates)
                )
        return batch_effect
    
    def get_biological_effect(self, theta):
        return self.project(self._get_biological_effect_lowdim(theta))
    
    def get_batch_effect(self, theta, covariates, intermediate, nullify_covariates = False):
        return self.project(self._get_batch_effect_lowdim(theta, covariates, nullify_covariates = nullify_covariates))

    def get_softmax_denom(self, theta, covariates, intermediate, include_batcheffects = True):
        return self.project(
            self._get_biological_effect_lowdim(theta) + \
            self._get_batch_effect_lowdim(theta, covariates, nullify_covariates = not include_batcheffects)
        ).exp().sum(-1)


# CLIFF:
# This is the main class for the embedded topic model. When this is mixed in with the base topic model,
# it tweaks the encoder and decoder models to use the embeddings. It also adds a method to "fit" the embeddings.
# The process of fitting could be using PMI-SVD, downloading them from Cistrome, etc.
#
class EmbeddedModelMixin:

    def _get_decoder_model(self):
        return partial(
                    ProjectionDecoder,
                    embedding_matrix = self._embedding_matrix,
                    is_trainable = False, #self._embeddings_trainable,
                )
    
    def _get_encoder_model(self):
        return super()._get_encoder_model()
    
    def preprocess_endog(self, X):
        return super().preprocess_endog(X)
    
    def _fit_embeddings(self, dataset, **feature_attrs):
        raise NotImplementedError()

    def _get_dataset_statistics(self, dataset, training_bar = True, **feature_attrs):
        super()._get_dataset_statistics(dataset, training_bar = training_bar)

        # if you've already fit the embeddings, don't do it again
        # when retraining the model.
        try:
            self._embeddings_transformer
        except AttributeError:
            (self._embedding_matrix, self._embeddings_trainable, self._embeddings_transformer) \
                = self._fit_embeddings(dataset, **feature_attrs)
        

    def _get_save_data(self):
        data = super()._get_save_data()
        
        data['fit_params']['_embedding_matrix'] = self._embedding_matrix
        data['fit_params']['_embeddings_trainable'] = self._embeddings_trainable
        data['fit_params']['_embeddings_transformer'] = self._embeddings_transformer

        return data
    
 
# CLIFF: The specialization to use PMI-SVD for the embeddings calculated *from the data*.
class LSIEmbeddingModel(EmbeddedModelMixin):
    '''
    Like the "Base" topic model, this class is agnostic to the modality of the data.
    '''

    lsi_dim = 500

    def _get_encoder_model(self):
        return partial(LSIEncoder, input_dim = self.lsi_dim)
    

    def _fit_embeddings(self, dataset, **feature_attrs):

        X_matrix = sparse.vstack([
            x['endog_features']
            for x in dataset
        ])

        '''def _get_embeddings(svd_pipeline, alpha=0.5):
            svd = svd_pipeline.steps[1][1]
            # Take the power of the diagonal elements of S
            S_alpha = np.power(svd.singular_values_, alpha)
            embeds = S_alpha[:,None] * svd.components_

            return embeds / np.linalg.norm(embeds, axis=0, keepdims=True)

        svd_pipeline = Pipeline([
            ('tfidf', PointwiseMutualInfoTransformer()),
            ('svd', TruncatedSVD(n_components= self.lsi_dim, random_state=42)),
        ])'''
        
        # CLIFF: this is how I calculate the embeddings from the data.
        # There are probably a bunch of ways to do this, not clearn which is the best.
        # Empirically, this way seems to work well.
        # Above is another way to go about it based on your repo...
        def _get_embeddings(svd_pipeline):
            
            tfidf = svd_pipeline.steps[0][1]
            svd = svd_pipeline.steps[1][1]
            embeds = svd.singular_values_[:,None] * svd.components_ * 1/tfidf.idf_[None,:]
            #return embeds/np.linalg.norm(embeds, axis=0, keepdims=True)
            return embeds

        svd_pipeline = Pipeline([
            ('tfidf', TfidfTransformer()),
            ('svd', TruncatedSVD(n_components= self.lsi_dim, random_state=1776)),
        ])

        logger.info('Fitting LSI pipeline.')
        svd_pipeline.fit(X_matrix)

        return (
            _get_embeddings(svd_pipeline),
            np.zeros(X_matrix.shape[1]).astype(bool), # don't train the embeddings
            svd_pipeline,
        )

    def preprocess_endog(self, X):

        if not sparse.isspmatrix(X):
            X = sparse.csr_matrix(X)

        return self._embeddings_transformer\
                    .transform(X)\
                    .astype(np.float32)
    


# CLIFF: A customized encoder model to partially use pre-trained embeddings.
# The `MixedEmbeddingEncoder` class is a wrapper around some base encoder model.
# It takes a pre-trained embedding matrix and uses it to transform the in-vocab features.
# The out-of-vocab features are passed directly to the base encoder model.
# Finally, the transformed in-vocab features are concatenated with the `extra features` 
# and passed to the base encoder model.
class MixedEmbeddingEncoder(nn.Module):
    
    def __init__(self,
        base_encoder,
        *,
        embedding_matrix,
        oov_size,
        num_extra_features,
        num_endog_features, # list this here so it is captured and not passed to the base encoder
        **encoder_kwargs,
    ):
        super().__init__()

        (dim, in_vocab_size) = embedding_matrix.shape

        self.pretrained_embedding = nn.Embedding(in_vocab_size + 1, dim, padding_idx=0)
        
        # add a zero column to the left of the embedding matrix
        padded_embedding_matrix = np.hstack([np.zeros((dim, 1)), embedding_matrix])
        self.pretrained_embedding.weight = nn.Parameter(torch.Tensor(padded_embedding_matrix.T))
        
        # TODO: Experiment with this - I've found that it's better to allow the model to learn the embeddings
        # - the pretrained embeddings are not directly suited for the DAN model.
        self.pretrained_embedding.requires_grad_(True)
        # this sort of defeats the purpose of the separated embeddings, but you need this class to do these 
        # experiments.

        self.base_encoder = base_encoder(
            num_endog_features = oov_size,
            num_extra_features = num_extra_features + dim,
            # manipulate the underlying encoder so that it learns embeddings for OOV features
            # and uses the transformed pretrained embeddings as "extra features".
            **encoder_kwargs
        )

    @property
    def intermediate_output(self):
        return self.base_encoder.intermediate_output

    def split_X(self, X, extra_features):
        (iv, oov) = X # iv : in_vocab, oov : out_of_vocab
        iv = apply_dan(self.pretrained_embedding, iv)
        iv = torch.cat([iv, extra_features], dim = 1)
        return (iv, oov)

    def forward(self, X, read_depth, covariates, extra_features):
        iv, oov = self.split_X(X, extra_features)
        return self.base_encoder(oov, read_depth, covariates, iv)
    
    def get_topic_comps(self, idx, read_depth, covariates, extra_features):
        iv, oov = self.split_X(idx, extra_features)
        return self.base_encoder.get_topic_comps(oov, read_depth, covariates, iv)
    
    def sample_posterior(self, X, read_depth, covariates, extra_features, n_samples = 100):
        iv, oov = self.split_X(X, extra_features)
        return self.base_encoder.sample_posterior(oov, read_depth, covariates, iv, n_samples = n_samples)


# CLIFF: This is a specialized model which uses pre-trained embeddings which are 
# stored in the AnnData object.
#
# pass the varm key with the embeddings as the `feature_embeddings_key` argument.
#
class PretrainedEmbeddingModel(EmbeddedModelMixin):
    '''
    Like the "Base" topic model, this class is agnostic to the modality of the data.
    '''

    def _get_encoder_model(self):
        return partial(
            MixedEmbeddingEncoder,
            super()._get_encoder_model(),
            embedding_matrix = self._embedding_matrix[:, ~self._embeddings_trainable],
            oov_size = self._embeddings_trainable.sum(),
        )


    def _fit_embeddings(self, dataset,
                        *,
                        feature_embeddings,
                        trainable_embeddings, 
                        **_ # collect the rest of the arguments
                    ):
        return (
            feature_embeddings,
            trainable_embeddings, # the mask vector marks all of the embeddings which are all 0 as `TRUE`
                                  # so unknown embeddings should be set to 0s in the AnnData.
            None,
        )


    def preprocess_endog(self, X):
        
        if not sparse.isspmatrix(X):
            X = sparse.csc_matrix(X)

        # split the matrices into in-vocab and out-of-vocab
        in_vocab = X[:, ~self._embeddings_trainable].tocsr()
        out_of_vocab = X[:, self._embeddings_trainable].tocsr() # I x OOV

        return (
            super().preprocess_endog(in_vocab),
            super().preprocess_endog(out_of_vocab)
        )
        
