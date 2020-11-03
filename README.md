# Large Scale Multi-Label Learning using Gaussian Processes #

This repository includes the code used in our paper [Large Scale Multi-Label Learning using Gaussian Processes](https://www.springer.com/journal/10994) to reproduce results. 

## Requirements ##
numpy - version 
tensorFlow==2.1.0  
tensorFlow_probability==0.9.0  
gpflow==2.0.0
silence-tensorflow==1.1.1 (optional)
xc_metrics (dowoloaded from [here](https://github.com/kunaldahiya/pyxclib))

## Flags ##
* max_batch_size: Maximum batch size; the final batch size (integer - default=2000)
* min_batch_size: Minimum batch size; initial batch size (integer - default=100)
* num_epochs: Number of epochs (integer - default=100)
* display_freq: Display loss function value and metrics every FLAGS.display_freq epochs (integer - default=5)
* num_inducings: Number of inducing points M (integer - default=400)
* num_factors: Number of factors P (integer - default=30)
* l_r: Learning rate for Adam optimizer (float - default=0.005)
* d_dfgp: Number of output dimensions for DFGP's DNN (integer - default=4)
* dataset: Dataset name (string - available names=[bibtex,delicious,eurlex] - default=eurlex)
* kernel: Chosen kernel - accepted values [se, linear, se_plus_linear] (string - default=se_plus_linear)
* print_metrics: Print metrics throughout optimization (boolean - default=True)

## Source code ##

The following files can be found in the **src** directory :  

- *mlgpf_model.py*: implementation of all the MLGPF model
- *utilities.py*: various utility functions
- *main_script.py*: code for replicating the results of MLGPF model over several real-world datasets

The **data** folder contains saved real-world datasets obtained from the [Extreme Multi-label Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). The **logs** and **models** folders are used to save results and models' parameters, respectively.

## Examples ##
Some representive examples to run experiments using MLGPF model.

```
# Train MLGPF model over the Eurlex dataset.
# Set the number of epochs equal to 250. 
# Set the number of inducing inputs and factors equal to 400 and 300, respectively
# Use as kernel a linear combination of squared exponential and linear kernel

python src/main_script.py --dataset=eurlex --num_epochs=250 --num_inducings=400 --num_factors=300 --kernel=se_plus_linear
```


