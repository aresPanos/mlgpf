import numpy as np
import sys
from scipy.sparse import csr_matrix, lil_matrix
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.datasets import load_svmlight_file
from xclib.utils.sparse import ll_to_sparse
from xclib.evaluation import xc_metrics
import time


def read_data_helper(filename: str, header=True, dtype='float32', zero_based=True):
    with open(filename, 'rb') as f:
        _l_shape = None
        if header:
            line = f.readline().decode('utf-8').rstrip("\n")
            line = line.split(" ")
            num_samples, num_feat, num_labels = int(line[0]), int(line[1]), int(line[2])
            _l_shape = (num_samples, num_labels)
        else:
            num_samples, num_feat, num_labels = None, None, None
        features, labels = load_svmlight_file(f, n_features=num_feat, multilabel=True)
        labels = ll_to_sparse(labels, dtype=dtype, zero_based=zero_based, shape=_l_shape)
    return features, labels, num_samples, num_feat, num_labels


def read_data(dataset: str):
    data_dir = '../data/' + dataset 
    
    X_train, Y_train, Ntrain, num_dim_tr, num_labels_tr = read_data_helper(data_dir + '/train.txt')
    X_test, Y_test, Ntest, num_dim_tst, num_labels_tst = read_data_helper(data_dir + '/test.txt')
    
    if X_train.shape[1] != X_test.shape[1] or Y_train.shape[1] != Y_test.shape[1]:
        print('IncompatibleDimensionsError: Something went wrong during file reading.')
        sys.exit(1)

    get_nz_rows = Y_train.getnnz(1) != 0
    X_train = X_train[get_nz_rows]
    Y_train = Y_train[get_nz_rows]
    
    get_nz_rows = Y_test.getnnz(1) != 0
    X_test = X_test[get_nz_rows]
    Y_test = Y_test[get_nz_rows]
    
    X_train = normalize(X_train)
    X_test = normalize(X_test)    
    
    return X_train, Y_train, X_test, Y_test


def compute_metrics(model, X_feat, Y_labels, wts, batch_size=50000): 
    '''
    Compute P@k, nDCG@k and their propensity scored versions PSP@k and PSnDCG@k, respectively. k=1,...,5
    '''
    metrics_all = xc_metrics.Metrics(Y_labels, wts)
    
    if Y_labels.shape[0] > 20000 or Y_labels.shape[1] > 20000:
        count = 0
        dataset_tf = tf.data.Dataset.from_tensor_slices(X_feat).batch(batch_size)        
        mtr_scores = lil_matrix((Y_labels.shape[0], Y_labels.shape[1]), dtype=np.float64)
        start_time = time.time()
        for batch_X in dataset_tf:
            mtr_scores_tmp = model.predict_scores(batch_X).numpy()
            for n_test in range(batch_X.shape[0]):
                Prob_scores_n_test = mtr_scores_tmp[n_test]
                top5_indices = np.argpartition(Prob_scores_n_test, -5)[-5:]
                top5_indices.sort()
                 
                mtr_scores[count+n_test, top5_indices] = Prob_scores_n_test[top5_indices]
                
            count += batch_X.shape[0]
        
        pred_time = time.time() - start_time
        mtr_scores = mtr_scores.tocsr()
    else:
        start_time = time.time()
        mtr_scores = model.predict_scores(X_feat).numpy()
        pred_time = time.time() - start_time
        
    all_metrics = [100*elem for elem in metrics_all.eval(mtr_scores)]
    return all_metrics[0], all_metrics[1], all_metrics[2], all_metrics[3], pred_time
    
    
def get_flags():
    flags = tf.compat.v1.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('max_batch_size', 2000, 'Maximum batch size; the final batch size')
    flags.DEFINE_integer('min_batch_size', 100, 'Minimum batch size; initial batch size')
    flags.DEFINE_integer('step_batch_size', 20, 'Step for increasing batch size after \"freq_batch_size\" epochs')
    flags.DEFINE_integer('freq_batch_size', 10, 'Number of epochs required to increase batch_size by step_batch_size')
    flags.DEFINE_integer('num_epochs', 100, 'Number of epochs')
    flags.DEFINE_integer('display_freq', 5, 'Display loss function value and metrics every FLAGS.display_freq epochs')
    flags.DEFINE_integer('num_inducings', 400, 'Number of inducing points M')
    flags.DEFINE_integer('num_factors', 30, 'Number of factors P')
    flags.DEFINE_float('l_r', 0.005, 'Learning rate for Adam optimizer')
    flags.DEFINE_string('dataset', 'eurlex', 'Dataset name')
    flags.DEFINE_string('kernel', 'se_plus_linear', 'Chosen kernel - accepted values [se, linear, se_plus_linear]')
    flags.DEFINE_bool('print_metrics', True, 'Print metrics throughout optimization (Boolean)')
    
    if FLAGS.max_batch_size < FLAGS.min_batch_size:
        print('ValueError: max_batch_size must be greater than or equal to min_batch_size')
        sys.exit(1)
    
    if FLAGS.kernel not in ['se', 'linear', 'se_plus_linear']:
        print('ValueError: kernel must have one of the following values: se, linear, se_plus_linear')
        sys.exit(1)
        
    return FLAGS