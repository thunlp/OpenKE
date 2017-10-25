# OpenKE
An Open-source Framework for Knowledge Embedding.

More information is available on our website 
[http://openke.thunlp.org/](http://openke.thunlp.org/)

## Overview

This is an Efficient implementation based on TensorFlow for knowledge representation learning (KRL). We use C++ to implement some underlying operations such as data preprocessing and negative sampling. For each specific model, it is implemented by TensorFlow with Python interfaces so that there is a convenient platform to run models on GPUs. OpenKE composes 3 repositories:

OpenKE: the main project based on TensorFlow, which provides the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/TensorFlow-TransX"> TensorFlow-TransX</a>: light and simple version of OpenKE based on TensorFlow, including TransE, TransH, TransR and TransD. 

<a href="https://github.com/thunlp/Fast-TransX"> Fast-TransX</a>: efficient lightweight C++ inferences for TransE and its extended models utilizing the framework of OpenKE, including TransH, TransR, TransD, TranSparse and PTransE. 

## Installation

1. Install TensorFlow

2. Clone the OpenKE repository:

	$ git clone https://github.com/thunlp/OpenKE
	
	$ cd OpenKE

3. Compile C++ files
	
	$ bash make.sh

## Data

Datasets are required in the following format, containing at least three files for training:

triple2id.txt: training file, the first line is the number of triples for training. Then the follow lines are all in the format (e1, e2, rel).

entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.


## Quickstart

To compute a knowledge graph embedding, first import datasets and set configure parameters for training, then train models and export results. For instance, we write a example.py to train TransE:


	import config
	import models
	import tensorflow as tf
	
	con = config.Config()
	
	con.set_in_path("benchmarks/FB15K/")
	con.set_out_path("benchmarks/FB15K/")
	con.set_work_threads(1)
	
	con.set_export_files("res/model.vec")
	con.set_export_steps(10)
	
	con.set_train_times(500)
	con.set_nbatches(100)
	con.set_alpha(0.5)
	con.set_bern(0)
	con.set_dimension(200)
	con.set_margin(1)
	con.set_ent_neg_rate(1)
	con.set_rel_neg_rate(0)
	con.set_optimizer("SGD")
	
	con.init()
	con.set_model(models.TransE)
	con.run()
	

### Step 1: Import datasets

	con.set_in_path("benchmarks/FB15K/")
	con.set_out_path("benchmarks/FB15K/")
	
We import knowledge graphs from benchmarks/FB15K/ folder. The data consists of three essential files mentioned before:

*	triple2id.txt
*	entity2id.txt
*	relation2id.txt

Validation and test files are required and used to evaluate the training results, However, they are not indispensable for training.

	con.set_work_threads(1)

We can allocate several threads to sample positive and negative cases.


### Step 2: Set configure parameters for training.

	con.set_train_times(500)
	con.set_nbatches(100)
	con.set_alpha(0.5)
	con.set_dimension(200)
	con.set_margin(1)

We set essential parameters, including the data traversing rounds, learning rate, batch size, and dimensions of entity and relation embeddings.

	con.set_bern(0)
	con.set_ent_neg_rate(1)
	con.set_rel_neg_rate(0)

For negative sampling, we can corrupt entities and relations to construct negative triples. set\_bern(0) will use the traditional sampling method, and set\_bern(1) will use the method in (Wang et al. 2014) denoted as "bern".
	
	con.set_optimizer("SGD")
	
We can select a proper gradient descent optimization algorithm to train models.

### Step 3: Train models

	con.init()
	con.set_model(models.TransE)
	con.run()
positive
We set the knowledge graph embedding model and start the training process.
	
### Step 4: Export results

	con.set_export_files("res/model.vec")

	con.set_export_steps(10)

The results will be automatically exported to the given files every few rounds.

## Interfaces

### Config
	
	class Config(object):
			
		#To set the learning rate
		def set_alpha(alpha = 0.001)
		
		#To set the degree of the regularization on the parameters
		def set_lmbda(lmbda = 0.0)
		
		#To set the gradient descent optimization algorithm (SGD, Adagrad, Adadelta, Adam)
		def set_optimizer(optimizer = "SGD")
		
		#To set the data traversing rounds
		def set_train_times(self, times)
		
		#To split the training triples into several batches, nbatches is the number of batches
		def set_nbatches(nbatches = 100)
		
		#To set the margin for the loss function
		def set_margin(margin = 1.0)
		
		#To set the dimensions of the entities and relations at the same time
		def set_dimension(dim)
		
		#To set the dimensions of the entities
		def set_ent_dimension(self, dim)
		
		#To set the dimensions of the relations
		def set_rel_dimension(self, dim)
		
		#To set the input folder		
		def set_in_path(path = "./")
		
		#To set the output folder
		def set_out_path(path = "./")
		
		#To allocate threads for each batch sampling
		def set_work_threads(threads = 1)
		
		#To set negative sampling algorithms, unif (bern = 0) or bern (bern = 1)
		def set_bern(bern = 1)
		
		#For each positive triple, we construct rate negative triples by corrupt the entity
		def set_ent_neg_rate(rate = 1)
		
		#For each positive triple, we construct rate negative triples by corrupt the relation
		def set_rel_neg_rate(rate = 0)
		
		#To sample a batch of training triples, including positive and negative ones.
		def sampling()
		
		#To set the import files, all parameters can be restored from the import files
		def set_import_files(path = None)
		
		#To set the export files
		def set_export_files(path = None)
		
		#To export results every several rounds
		def set_export_steps(steps = 1)
		
		#To set the knowledge embedding model
		set_model(model)
		
		#The framework will print loss values during training if flag = 1
		def set_log_on(flag = 1)
	
	
	
	
	


### Model

	class Model(object)
	
		# return config which saves the training parameters.
		get_config(self)
		
		# in_batch = True, return [positive_head, positive_tail, positive_relation]
		# The shape of positive_head is [batch_size, 1]
		# in_batch = False, return [positive_head, positive_tail, positive_relation]
		# The shape of positive_head is [batch_size]
		get_positive_instance(in_batch = True)
		
		# in_batch = True, return [negative_head, negative_tail, negative_relation]
		# The shape of positive_head is [batch_size, negative_ent_rate + negative_rel_rate]
		# in_batch = False, return [negative_head, negative_tail, negative_relation]
		# The shape of positive_head is [(negative_ent_rate + negative_rel_rate) * batch_size]		
		get_negative_instance(in_batch = True)

		# in_batch = True, return all training instances with the shape [batch_size, (1 + negative_ent_rate + negative_rel_rate)]
		# in_batch = False, return all training instances with the shape [(negative_ent_rate + negative_rel_rate + 1) * batch_size]
		def get_all_instance(in_batch = False)

		# in_batch = True, return all training labels with the shape [batch_size, (1 + negative_ent_rate + negative_rel_rate)]
		# in_batch = False, return all training labels with the shape [(negative_ent_rate + negative_rel_rate + 1) * batch_size]
		# The positive triples are labeled as 1, and the negative triples are labeled as -1
		def get_all_labels(in_batch = False)
		
		# To define containers for training triples
		def input_def()
		
		# To define embedding parameters for knowledge embedding models
		def embedding_def()

		# To define loss functions for knowledge embedding models
		def loss_def()
		
		def __init__(config)

	#The implementation for TransE
	class TransE(Model)

	#The implementation for TransH	
	class TransH(Model)

	#The implementation for TransR
	class TransR(Model)

	#The implementation for TransD
	class TransD(Model)
	
	#The implementation for RESCAL
	class RESCAL(Model)
	
	#The implementation for DistMult
	class DistMult(Model)
	
	#The implementation for HolE
	class HolE(Model)					
	
	#The implementation for ComplEx
	class ComplEx(Model)
	
		
	
	
	
