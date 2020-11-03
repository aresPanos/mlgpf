from typing import Optional, Tuple, TypeVar

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import os
from itertools import product

from gpflow.base import Module, Parameter
from gpflow.config import default_float, default_jitter, set_default_float
from gpflow.utilities import ops, positive, triangular
Data = TypeVar('Data',  Tuple[tf.Tensor, tf.Tensor], tf.Tensor)


class MLGPF_model(Module):
    def __init__(self,
                 num_points: int,
                 num_labels: int,
                 num_dim: int,
                 num_factors: int = 30,
                 num_inducings: int = 400,
                 lengthscales: tf.Tensor = None,
                 Z_init: tf.Tensor = None,
                 jitter: np.float = 1e-6,
                 q_mu: tf.Tensor = None,
                 q_sqrt: tf.Tensor = None,
                 use_se: bool = False,
                 use_linear: bool = False,
                 use_se_plus_linear: bool = False,
                 N_GH: int = 10):

        x_gh, w_gh = np.polynomial.hermite.hermgauss(N_GH)
        self.x_gh_3d, self.w_gh = tf.cast(np.sqrt(2)*x_gh[:, np.newaxis, np.newaxis], dtype=default_float()), tf.cast(w_gh / np.sqrt(np.pi), dtype=default_float())
        self.jitter_eye = tf.cast(jitter*np.eye(num_inducings), dtype=default_float())
        self.num_points = tf.cast(num_points, dtype=default_float())
        self.const_PM = tf.cast(num_factors*num_inducings, dtype=default_float())
        self.const_P = tf.cast(num_factors, dtype=default_float())
        self.M_ones = tf.ones([num_inducings, 1], dtype=default_float())
        self.P_ones_2d = tf.ones([num_factors, 1],  dtype=default_float())
        self.P_ones_3d = tf.ones([num_factors, 1 ,1], dtype=default_float())
        self.use_se = use_se
        self.use_linear = use_linear
        self.use_se_plus_linear = use_se_plus_linear
        
        lengthscales = np.ones(num_dim) if lengthscales is None else lengthscales
        q_mu = 0.5 * np.ones((num_inducings, num_factors)) if q_mu is None else q_mu
        q_sqrt = tfp.math.fill_triangular_inverse(np.array([0.2*np.eye(num_inducings) for _ in range(num_factors)])) if q_sqrt is None else q_sqrt
        Z_init = 0.2 * tf.np.random.rand(num_inducings, num_dim) if Z_init is None else Z_init
        
        if lengthscales.size != num_dim:
            print('Dimension mismatch: Variable \"lengthscales\" must be of size ({},)' .format(num_dim) )
            sys.exit(1)
            
        if q_mu.shape[0] != num_inducings or q_mu.shape[1] != num_factors:
            print('Dimension mismatch: Variable \"q_mu\" must be of size ({},{})' .format(num_inducings, num_factors) )
            sys.exit(1)
            
        if q_sqrt.shape[0] != num_factors or q_sqrt.shape[1] != int(0.5*num_inducings*(num_inducings + 1)):
            print('Dimension mismatch: Variable \"q_sqrt\" must be of size ({},{})' .format(num_factors, int(0.5*num_inducings*(num_inducings + 1))) )
            sys.exit(1)
            
        if Z_init.shape[0] != num_inducings or Z_init.shape[1] != num_dim:
            print('Dimension mismatch: Variable \"Z_init\" must be of size ({},{})' .format(num_inducings, num_dim) )
            sys.exit(1)
            

        self.Phi = Parameter(np.random.randn(num_labels, num_factors)/np.sqrt(num_factors), dtype=default_float())
        self.bias_vec = Parameter(np.random.randn(num_labels, 1)/np.sqrt(num_factors), dtype=default_float())
        self.Z_unorm = Parameter(Z_init, dtype=default_float())
        self.lengthscales = Parameter(lengthscales, transform=positive(), dtype=default_float())
        self.q_mu = Parameter(q_mu, dtype=default_float())
        self.q_sqrt = Parameter(q_sqrt, dtype=default_float())
        
        if self.use_se_plus_linear:
            self.se_var = Parameter(1., transform=positive(), dtype=default_float())
            self.linear_var = Parameter(1., transform=positive(), dtype=default_float())
        
        
    def se_kernel(self, batch_X):
        Z_norm = tf.nn.l2_normalize(self.Z_unorm, 1)
        Z_norm_ells_T = tf.transpose(Z_norm*self.lengthscales) # D x M
        Xb_Lambda = batch_X.__mul__(self.lengthscales)
        Z_dot_Z = tf.linalg.matmul(Z_norm_ells_T, Z_norm_ells_T, transpose_a=True) # M x M
        X_dot_Z = tf.sparse.sparse_dense_matmul(Xb_Lambda, Z_norm_ells_T) # minibatch x M
        
        Z_mul_sum = tf.linalg.diag_part(Z_dot_Z) # (M, )
            
        H_p = self.M_ones * Z_mul_sum
        A_p = H_p + tf.transpose(H_p) - 2.*Z_dot_Z

        A_k2_tmp = tf.sparse.reduce_sum(tf.square(Xb_Lambda), 1) - 2.*tf.transpose(X_dot_Z) # M x minibatch
        A_k2 = tf.transpose(A_k2_tmp) + Z_mul_sum # minibatch x M
        
        K_mm = tf.exp(-A_p) + self.jitter_eye # M x M

        K_mn = tf.exp(-A_k2) # minibatch_size x M
        K_mn = tf.transpose(K_mn) # M x minibatch_size
        
        return K_mm, K_mn, 1.
        
        
    def linear_kernel(self, batch_X):
        Z_norm = tf.nn.l2_normalize(self.Z_unorm, 1)
        Z_norm_ells_T = tf.transpose(Z_norm*self.lengthscales) # D x M
        Xb_Lambda = batch_X.__mul__(self.lengthscales)
        Xb_Lambda_sq_sum = tf.sparse.reduce_sum(tf.square(Xb_Lambda), 1)
        K_mm = tf.linalg.matmul(Z_norm_ells_T, Z_norm_ells_T, transpose_a=True) + self.jitter_eye # M x M
        K_mn = tf.sparse.sparse_dense_matmul(Xb_Lambda, Z_norm_ells_T) # minibatch x M
        K_mn = tf.transpose(K_mn) # M x minibatch_size
        
        return K_mm, K_mn, Xb_Lambda_sq_sum
    
    
    def se_plus_linear_kernel(self, batch_X):
        denom = self.se_var + self.linear_var
        norm_se_var, norm_linear_var = self.se_var / denom, self.linear_var / denom
        Z_norm = tf.nn.l2_normalize(self.Z_unorm, 1)
        Z_norm_ells_T = tf.transpose(Z_norm*self.lengthscales) # D x M
        Xb_Lambda = batch_X.__mul__(self.lengthscales)
        Z_dot_Z = tf.linalg.matmul(Z_norm_ells_T, Z_norm_ells_T, transpose_a=True) # M x M
        X_dot_Z = tf.sparse.sparse_dense_matmul(Xb_Lambda, Z_norm_ells_T) # minibatch x M
        Z_dot_X = tf.transpose(X_dot_Z) # M x minibatch_size
        Xb_Lambda_sq_sum = tf.sparse.reduce_sum(tf.square(Xb_Lambda), 1)
        
        Z_mul_sum = tf.linalg.diag_part(Z_dot_Z) # (M, )
            
        H_p = self.M_ones * Z_mul_sum
        A_p = H_p + tf.transpose(H_p) - 2.*Z_dot_Z

        A_k2_tmp = Xb_Lambda_sq_sum - 2.*Z_dot_X # M x minibatch
        A_k2 = tf.transpose(A_k2_tmp) + Z_mul_sum # minibatch x M
        
        K_mm = norm_se_var * tf.exp(-A_p) + norm_linear_var * Z_dot_Z + self.jitter_eye # M x M

        K_mn = norm_se_var * tf.exp(-A_k2) # minibatch_size x M
        K_mn = tf.transpose(K_mn) + norm_linear_var * Z_dot_X # M x minibatch_size
        
        return K_mm, K_mn, norm_linear_var * Xb_Lambda_sq_sum + norm_se_var


    def neg_elbo(self, batch_XY) -> tf.Tensor:
        mask_minus_ones = tf.where(tf.sparse.to_dense(batch_XY[1]), -1, 1)
        mask_minus_ones = tf.transpose(mask_minus_ones)
        mask_minus_ones = tf.cast(mask_minus_ones, default_float())
        q_sqrt_transf = tfp.math.fill_triangular(self.q_sqrt)
        
        if self.use_linear :
            K_mm, K_mn, var_kern = self.linear_kernel(batch_XY[0])
        elif self.use_se_plus_linear:
            K_mm, K_mn, var_kern = self.se_plus_linear_kernel(batch_XY[0])
        else:          
            K_mm, K_mn, var_kern = self.se_kernel(batch_XY[0])

        Lp = tf.linalg.cholesky(K_mm) # M x M
        alpha = tf.linalg.triangular_solve(Lp, self.q_mu, lower=True) # M x P
        Lq_diag = tf.linalg.diag_part(q_sqrt_transf) # P x M
        Lp_full = self.P_ones_3d*Lp[None, :, :] # P x M x M
        LpiLq = tf.linalg.triangular_solve(Lp_full, q_sqrt_transf, lower=True) # M x P
        sum_log_sqdiag_Lp = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(Lp))))
        KL_div = 0.5 * (self.const_P * sum_log_sqdiag_Lp + tf.reduce_sum(tf.square(alpha)) - self.const_PM - tf.reduce_sum(tf.math.log(tf.square(Lq_diag))) + tf.reduce_sum(tf.square(LpiLq)))
        
        A = tf.linalg.triangular_solve(Lp, K_mn, lower=True) # M x minibatch_size
        fvar = var_kern - tf.reduce_sum(tf.square(A), 0) # minibatch_size
        fvar = self.P_ones_2d*fvar[None, :] # P x minibatch_size
        A = tf.linalg.triangular_solve(tf.transpose(Lp), A, lower=False) # M x minibatch_size
        fmean = tf.linalg.matmul(A, self.q_mu, transpose_a=True) # minibatch_size x P - Marginal mean
        
        A = self.P_ones_3d*A[None, :, :] # P x M x minibatch_size
        LTA = tf.linalg.matmul(q_sqrt_transf, A, transpose_a=True) # P x M x minibatch_size
        fvar = fvar + tf.reduce_sum(tf.square(LTA), 1) # P x minibatch_size - Marginal variance
        
        # Expectations computation
        sum_dot_phi = (tf.linalg.matmul(self.Phi, fmean, transpose_b=True) + self.bias_vec)*mask_minus_ones # K x minibatch_size
        sum_dot_phi_sqrt = tf.math.sqrt(tf.matmul(tf.square(self.Phi), fvar))*mask_minus_ones # K x minibatch_size
        sum_dot_phi_sqrt = tf.expand_dims(sum_dot_phi_sqrt, 0)
        sum_E_q_all = tf.reduce_sum(self.w_gh * tf.reduce_sum(tf.nn.softplus(sum_dot_phi - self.x_gh_3d*sum_dot_phi_sqrt), [1, 2]))
        
        scale = self.num_points / tf.cast(tf.shape(fvar)[1], default_float())

        return scale * sum_E_q_all + 0.5 * KL_div


    def predict_scores(self, X_test) -> tf.Tensor:       
        if self.use_linear :
            K_mm_test, K_mn_test, _ = self.linear_kernel(X_test)
        elif self.use_se_plus_linear:
            K_mm_test, K_mn_test, _ = self.se_plus_linear_kernel(X_test)
        else:          
            K_mm_test, K_mn_test, _ = self.se_kernel(X_test)
            
        Lp_test = tf.linalg.cholesky(K_mm_test)
        A_test = tf.linalg.triangular_solve(Lp_test, K_mn_test, lower=True) # M x n_test
        A_test = tf.linalg.triangular_solve(tf.transpose(Lp_test), A_test, lower=False) # M x n_test
        fmean_test = tf.matmul(A_test, self.q_mu, transpose_a=True) # n_test x P
        
        return tf.matmul(fmean_test, self.Phi, transpose_b=True) + tf.squeeze(self.bias_vec) # n_test x K

