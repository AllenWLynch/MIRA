from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np


def multiply_by_rows(matrix, row_coefs):
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)

def multiply_by_columns(matrix, col_coefs):
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())


class PointwiseMutualInfoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, exponent=1, neg_val=1):
        self.exponent = exponent
        self.neg_val = neg_val

    def fit(self, X, y=None):
        counts = X.astype(float)
        self.sum_c_ = np.array(counts.sum(axis=0))[0, :]

        if self.exponent != 1:
            self.sum_c_ = self.sum_c_ ** self.exponent

        self.sum_total_ = self.sum_c_.sum()
        return self

    def transform(self, X):
        """
        Calculates positive PMI
        args:
            - X: sparse array csr format
        returns:
            pmi array in csr format
        """
        counts = X.astype(float)
        sum_w = np.array(counts.sum(axis=1))[:, 0]
        sum_c = self.sum_c_

        sum_w = np.reciprocal(sum_w)
        sum_c = np.reciprocal(sum_c)

        pmi = csr_matrix(counts)
        pmi = multiply_by_rows(pmi, sum_w)
        pmi = multiply_by_columns(pmi, sum_c)
        pmi = pmi * self.sum_total_

        # take log, eliminate negative values
        data = pmi.data
        mask = data > 0 
        log_data = np.zeros_like(data)
        log_data[mask] = np.log(data[mask])
        # set negative values to zero
        mask = log_data < 0
        log_data[mask] = 0
        pmi.data = log_data
        # remove zeros from sparse matrix
        pmi.eliminate_zeros() 
        return pmi

