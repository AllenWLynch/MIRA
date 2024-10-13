from mira.topic_model.base import get_fc_stack, encoder_layer
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def apply_dan(embedding_nn, idx):
    return torch.nan_to_num(embedding_nn(idx).sum(1)/idx.sum(1, keepdim=True), nan=0.0)

class EncoderBase(nn.Module):

    def topic_comps(self, idx, read_depth, covariates, extra_features):
        theta = self.forward(idx, read_depth, covariates, extra_features)[0]
        theta = theta.exp()/theta.exp().sum(-1, keepdim = True)
       
        return theta.detach().cpu().numpy()


    def sample_posterior(self, X, read_depth, covariates, extra_features,
            n_samples = 100):

        theta_loc, theta_scale = self.forward(X, read_depth, covariates, extra_features)
        theta_loc, theta_scale = theta_loc.detach().cpu().numpy(), theta_scale.detach().cpu().numpy()

        # theta = z*std + mu
        theta = np.random.randn(*theta_loc.shape, n_samples)*theta_scale[:,:,None] + theta_loc[:,:,None]
        theta = np.exp(theta)/np.exp(theta).sum(-2, keepdims = True)
        
        return theta.mean(-1)


class DANEncoder(EncoderBase):

    def __init__(self, 
        embedding_size = None,
        *,
        num_endog_features, num_topics, embedding_dropout,
        hidden, dropout, num_layers, num_exog_features, num_covariates, num_extra_features,
        ):
        super().__init__()

        if embedding_size is None:
            embedding_size = hidden

        self.word_dropout_rate = embedding_dropout
        self.embedding = nn.Embedding(num_endog_features + 1, embedding_size, padding_idx=0)
        self.num_topics = num_topics
        self.calc_readdepth = True
        self.fc_layers = get_fc_stack(
            layer_dims = [embedding_size + 1 + num_covariates + num_extra_features, 
                *[hidden]*(num_layers-2), 2*num_topics],
            dropout = dropout, skip_nonlin = True
        )

    def forward(self, idx, read_depth, covariates, extra_features):
       
        if self.training:
            corrupted_idx = torch.multiply(
                torch.empty_like(idx).bernoulli_(1-self.word_dropout_rate),
                idx
            )
        else:
            corrupted_idx = idx

        if self.calc_readdepth: # for compatibility with older models
            #read_depth = (corrupted_idx > 0).sum(-1, keepdim=True)
            pass
        
        ave_embeddings = apply_dan(self.embedding, corrupted_idx)

        X = torch.hstack([ave_embeddings, read_depth.log(), covariates, extra_features]) #inject read depth into model
        X = self.fc_layers(X)

        theta_loc = X[:, :self.num_topics]
        theta_scale = F.softplus(X[:, self.num_topics:(2*self.num_topics)])  

        return theta_loc, theta_scale



class DANSkipEncoder(EncoderBase):

    def __init__(self, embedding_size = None,*, num_endog_features, num_topics, embedding_dropout,
        hidden, dropout, num_layers, num_exog_features, num_covariates, num_extra_features):
        super().__init__()

        assert num_layers > 2, 'Cannot use SkipEncoder with less than three layers.'

        if embedding_size is None:
            embedding_size = hidden

        self.word_dropout_rate = embedding_dropout
        self.embedding = nn.Embedding(num_endog_features + 1, embedding_size, padding_idx=0)
        self.embedding_bn = nn.BatchNorm1d(embedding_size)

        self.num_topics = num_topics
        self.calc_readdepth = True

        self.output_layer = encoder_layer(
                hidden, 2*num_topics, 
                dropout = dropout, 
                nonlin = False
            )

        hidden_input = embedding_size + 1 + num_covariates + num_extra_features
        self.fc_layers = get_fc_stack(
            layer_dims = [hidden_input, *[hidden]*(num_layers-2)],
            dropout = dropout, 
            skip_nonlin = False
        )


    def forward(self, idx, read_depth, covariates, extra_features):
       
        if self.training:
            corrupted_idx = torch.multiply(
                torch.empty_like(idx).bernoulli_(1-self.word_dropout_rate),
                idx
            )
        else:
            corrupted_idx = idx

        embeddings = self.embedding_bn(apply_dan(self.embedding, corrupted_idx))

        X = torch.hstack([embeddings, read_depth.log(), covariates, extra_features]) #inject read depth into model
        
        #print(X, X.min(), X.max(), torch.isfinite(X).all())

        X = self.fc_layers(X)

        #print(X, X.min(), X.max(), torch.isfinite(X).all())

        X = self.output_layer(
            X + embeddings # skip connection
        )

        theta_loc = X[:, :self.num_topics]
        theta_scale = F.softplus(X[:, self.num_topics:(2*self.num_topics)])  

        return theta_loc, theta_scale


class LSIEncoder(EncoderBase):

    def __init__(self, embedding_size = None,*, input_dim, 
        num_endog_features, num_topics, embedding_dropout,
        hidden, dropout, num_layers, num_exog_features, num_covariates, num_extra_features):
        super().__init__()

        if embedding_size is None:
            embedding_size = hidden

        output_batchnorm_size = 2*num_topics
        self.num_topics = num_topics
        self.fc_layers = get_fc_stack(
            layer_dims = [input_dim + 1 + num_covariates + num_extra_features, 
            embedding_size, *[hidden]*(num_layers-2), output_batchnorm_size],
            dropout = dropout, skip_nonlin = True
        )

        self.fc_layers[-2].register_forward_hook(self.hook_intermediate_output)


    def hook_intermediate_output(self, module, input, output):
        self.intermediate_output_ = output

    @property
    def intermediate_output(self):
        return self.intermediate_output_

    def forward(self, X, read_depth, covariates, extra_features):

        X = torch.hstack([X, torch.log(read_depth), covariates, extra_features])

        X = self.fc_layers(X)

        theta_loc = X[:, :self.num_topics]
        theta_scale = F.softplus(X[:, self.num_topics:(2*self.num_topics)])# + 1e-5

        return theta_loc, theta_scale
