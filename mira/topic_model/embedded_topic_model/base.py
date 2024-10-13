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
        is_trainable,
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


    def forward(self, theta, covariates, nullify_covariates = False):
        
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
    
    def get_batch_effect(self, theta, covariates, nullify_covariates = False):
        return self.project(self._get_batch_effect_lowdim(theta, covariates, nullify_covariates = nullify_covariates))

    def get_softmax_denom(self, theta, covariates, include_batcheffects = True):
        return self.project(
            self._get_biological_effect_lowdim(theta) + \
            self._get_batch_effect_lowdim(theta, covariates, nullify_covariates = not include_batcheffects)
        ).exp().sum(-1)



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
    

class LSIEmbeddingModel(EmbeddedModelMixin):
    '''
    Like the "Base" topic model, this class is agnostic to the modality of the data.
    '''

    lsi_dim = 500

    def _get_encoder_model(self):
        return partial(LSIEncoder, input_dim = self.lsi_dim)
    

    def _fit_embeddings(self, dataset, **feature_attrs):
        
        def _get_embeddings(svd_pipeline, alpha=0.5):
            svd = svd_pipeline.steps[1][1]
            # Take the power of the diagonal elements of S
            S_alpha = np.power(svd.singular_values_, alpha)
            embeds = S_alpha[:,None] * svd.components_

            return embeds / np.linalg.norm(embeds, axis=0, keepdims=True)
        

        X_matrix = sparse.vstack([
            x['endog_features']
            for x in dataset
        ])

        svd_pipeline = Pipeline([
            ('tfidf', PointwiseMutualInfoTransformer()),
            ('svd', TruncatedSVD(n_components= self.lsi_dim)),
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
        
        # Experiment with this - I've found that it's better to allow the model to learn the embeddings
        # - the pretrained embeddings are not directly suited for the DAN model.
        self.pretrained_embedding.requires_grad_(True)

        self.base_encoder = base_encoder(
            num_endog_features = oov_size,
            num_extra_features = num_extra_features + dim,
            # manipulate the underlying encoder so that it learns embeddings for OOV features
            # and uses the transformed pretrained embeddings as "extra features".
            **encoder_kwargs
        )

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
            trainable_embeddings,
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
        
