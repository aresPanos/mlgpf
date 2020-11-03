import numpy as np
import time
import tensorflow as tf
from scipy.sparse import csr_matrix
from utilities import *
from mlgpf_model import *
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from xclib.evaluation import xc_metrics
import os

np.random.rand(1234567)
tf.random.set_seed(1234567)

# Get flags from command line
FLAGS = get_flags()

if FLAGS.kernel == 'se':
    use_se, use_linear, use_se_plus_linear = True, False, False
elif FLAGS.kernel == 'linear':
    use_se, use_linear, use_se_plus_linear = False, True, False
else:
    use_se, use_linear, use_se_plus_linear = False, False, True

datafile_name = '../logs/mlgpf_output_' + FLAGS.dataset + '_kernel=' + FLAGS.kernel + '.txt'

# Read dataset
print('Reading data...')
X_train, Y_train, X_test, Y_test = read_data(FLAGS.dataset)
Ntrain, Ntest, num_dim, num_labels = X_train.shape[0], X_test.shape[0], X_train.shape[1], Y_train.shape[1]
wts = xc_metrics.compute_inv_propesity(Y_train, A=0.55, B=1.5)
print('Done!')

# Initialize inducing points Z via k-means clustering
print('Initialize inducing inputs via k-means...')
kmeans = MiniBatchKMeans(n_clusters=FLAGS.num_inducings, max_iter=1, batch_size=1000, compute_labels=False, tol=1e-3, n_init=1)
kmeans.fit(X_train)
Z_0 = normalize(kmeans.cluster_centers_)
del kmeans
print('Done!')

print('Results are being written in file ', datafile_name)

X_train = tf.sparse.SparseTensor(np.column_stack(X_train.nonzero()), X_train.data, dense_shape=X_train.shape)
Y_train = tf.sparse.SparseTensor(np.column_stack(Y_train.nonzero()), Y_train.data.astype(bool), dense_shape=Y_train.shape)
X_test = tf.sparse.SparseTensor(np.column_stack(X_test.nonzero()), X_test.data, dense_shape=X_test.shape)

shuffle_size = 50000 if Ntrain > 50000 else Ntrain
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).repeat().shuffle(buffer_size=shuffle_size)
batches_iter = iter(train_dataset.batch(FLAGS.min_batch_size))
iters_per_epoch = Ntrain // FLAGS.min_batch_size
del X_train, Y_train

model = MLGPF_model(num_points=Ntrain,
                        num_labels=num_labels,
                        num_dim=num_dim,
                        num_factors=FLAGS.num_factors,
                        num_inducings=FLAGS.num_inducings,
                        Z_init=Z_0,
                        use_se=use_se,
                        use_linear=use_linear,
                        use_se_plus_linear=use_se_plus_linear)


adam_opt = tf.optimizers.Adam(learning_rate=FLAGS.l_r)

@tf.function(autograph=False)
def optimization_step(batches_iter_in):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        objective = model.neg_elbo(next(batches_iter_in))
        grads = tape.gradient(objective, model.trainable_variables)
    adam_opt.apply_gradients(zip(grads, model.trainable_variables))
    return objective

with open(datafile_name, 'w') as fl:
    print('\n*************  MLGPF model  ********************', file=fl)  
    print('Dataset: ', FLAGS.dataset.upper(), file=fl)
    print('Kernel: ', FLAGS.kernel.upper(), file=fl)
    print('Number of labels: ', num_labels, file=fl)
    print('Number of training points: ', Ntrain, file=fl)
    print('Number of testing points: ', Ntest, file=fl)
    print('Number of inducing inputs M: ', FLAGS.num_inducings, file=fl)
    print('Number of factors P: ', FLAGS.num_factors, file=fl)
    print('Number of input dimensions: ', num_dim, file=fl)
    print('Number of epochs: ', FLAGS.num_epochs, file=fl)
    print('***********************************************************\n\n', file=fl)


print('Training model...')
current_batch_size = FLAGS.min_batch_size
train_time = 0
iters = 0
elbo_history = []
for epoch in range(FLAGS.num_epochs):      
    for _ in range(iters_per_epoch):
        start_time = time.time()
        loss_value = optimization_step(batches_iter)
        train_time += time.time() - start_time
        elbo_history.append(-loss_value.numpy())
        
        if ((iters+1) % FLAGS.display_freq) == 0 or iters == 0:
            with open(datafile_name, 'a') as fl:
                print('\nEpoch: {}/{}  Iteration: {}  ELBO: {:.4f}' .format(epoch+1, FLAGS.num_epochs, iters+1, elbo_history[-1]), file=fl)
                if FLAGS.print_metrics:
                    P_k, nDCG_k, PSP_k, PSnDCG_k, pred_time = compute_metrics(model, X_test, Y_test, wts, batch_size=1000)
                    print('P@1 = {:6.2f}   PSP@1 = {:6.2f}   nDCG@1 = {:6.2f}   PSnDCG@1 = {:6.2f}' .format(P_k[0], PSP_k[0], nDCG_k[0], PSnDCG_k[0]), file=fl)
                    print('P@3 = {:6.2f}   PSP@3 = {:6.2f}   nDCG@3 = {:6.2f}   PSnDCG@3 = {:6.2f}' .format(P_k[2], PSP_k[2], nDCG_k[2], PSnDCG_k[2]), file=fl)
                    print('P@5 = {:6.2f}   PSP@5 = {:6.2f}   nDCG@5 = {:6.2f}   PSnDCG@5 = {:6.2f}' .format(P_k[4], PSP_k[4], nDCG_k[4], PSnDCG_k[4]), file=fl)
        iters += 1
    
    if ((epoch+1) % FLAGS.freq_batch_size) == 0 and (current_batch_size < FLAGS.max_batch_size):
        current_batch_size += FLAGS.step_batch_size
        batches_iter = iter(train_dataset.batch(current_batch_size))
        iters_per_epoch = Ntrain // current_batch_size
        

print('Done!')
with open(datafile_name, 'a') as fl:
    P_k, nDCG_k, PSP_k, PSnDCG_k, pred_time = compute_metrics(model, X_test, Y_test, wts, batch_size=1000)
    print('\n\n###########\nTraining time: {:.4f} hours' .format(train_time / 3600), file=fl)
    print('Prediction time: {:.3f} seconds'.format(pred_time), file=fl)

# Save model's parameters
filename_model_dir = '../models/' + FLAGS.dataset
try:
    if not os.path.exists(filename_model_dir):
       os.makedirs(filename_model_dir)
except OSError:
    print ("Creation of the directory %s failed" % filename_model_dir)
print('Saving model in directory ', filename_model_dir)
                          
if use_se_plus_linear:
    dict_mlgpf = {'q_mu': model.q_mu.numpy(), 'q_sqrt': model.q_sqrt.numpy(), 
              'Phi': model.Phi.numpy(), 'bias_vec': model.bias_vec.numpy(), 
              'Z_unorm': model.Z_unorm.numpy(), 'lengthscales': model.lengthscales.numpy(), 
              'se_var': model.se_var.numpy(), 
              'linear_var': model.linear_var.numpy(),
              'elbo_history': elbo_history}  
else:
    dict_mlgpf = {'q_mu': model.q_mu.numpy(), 'q_sqrt': model.q_sqrt.numpy(), 
              'Phi': model.Phi.numpy(), 'bias_vec': model.bias_vec.numpy(), 
              'Z_unorm': model.Z_unorm.numpy(), 'lengthscales': model.lengthscales.numpy(), 
              'elbo_history': elbo_history}
                           
np.save(filename_model_dir + '/' + FLAGS.dataset + '_kernel=' + FLAGS.kernel + '_mlgpf.npy', dict_mlgpf)
print('Done!')