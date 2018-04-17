# OpenKE-PyTorch
An Open-source Framework for Knowledge Embedding implemented with PyTorch.

More information is available on our website 
[http://openke.thunlp.org/](http://openke.thunlp.org/)

## Overview

This is an Efficient implementation based on PyTorch for knowledge representation learning (KRL). We use C++ to implement some underlying operations such as data preprocessing and negative sampling. For each specific model, it is implemented by PyTorch with Python interfaces so that there is a convenient platform to run models on GPUs. OpenKE composes 4 repositories:

OpenKE-PyTorch: the project based on PyTorch, which provides the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/OpenKE"> OpenKE</a>: OpenKE implemented with TensorFlow, also providing the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/TensorFlow-TransX"> TensorFlow-TransX</a>: light and simple version of OpenKE based on TensorFlow, including TransE, TransH, TransR and TransD. 

<a href="https://github.com/thunlp/Fast-TransX"> Fast-TransX</a>: efficient lightweight C++ inferences for TransE and its extended models utilizing the framework of OpenKE, including TransH, TransR, TransD, TranSparse and PTransE. 

## Installation

1. Install PyTorch

2. Clone the OpenKE-PyTorch branch:

	$ git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE
	
	$ cd OpenKE

3. Compile C++ files
	
	$ bash make.sh

## Data

* For training, datasets contain three files:

  train2id.txt: training file, the first line is the number of triples for training. Then the follow lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2*** .

  entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

  relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

* For testing, datasets contain additional two files (totally five files):

  test2id.txt: testing file, the first line is the number of triples for testing. Then the follow lines are all in the format ***(e1, e2, rel)*** .

  valid2id.txt: validating file, the first line is the number of triples for validating. Then the follow lines are all in the format ***(e1, e2, rel)*** .

## Quickstart

### Training

To compute a knowledge graph embedding, first import datasets and set configure parameters for training, then train models and export results. For instance, we write an example_train_transe.py to train TransE:
	

	import config
	import models
	import json
	import numpy as np


	con = config.Config()
	#Input training files from benchmarks/FB15K/ folder.
	con.set_in_path("./benchmarks/FB15K/")
	con.set_work_threads(4)
	con.set_train_times(500)
	con.set_nbatches(100)
	con.set_alpha(0.001)
	con.set_margin(1.0)
	con.set_bern(0)
	con.set_dimension(50)
	con.set_ent_neg_rate(1)
	con.set_rel_neg_rate(0)
	con.set_opt_method("SGD")

	#Models will be exported via tf.Saver() automatically.
	con.set_export_files("./res/model.vec.tf", 0)
	#Model parameters will be exported to json files automatically.
	con.set_out_files("./res/embedding.vec.json")
	#Initialize experimental settings.
	con.init()
	#Set the knowledge embedding model
	con.set_model(models.TransE)
	#Train the model.
	con.run()	

#### Step 1: Import datasets

	con.set_in_path("benchmarks/FB15K/")
	
We import knowledge graphs from benchmarks/FB15K/ folder. The data consists of three essential files mentioned before:

*	train2id.txt
*	entity2id.txt
*	relation2id.txt

Validation and test files are required and used to evaluate the training results, However, they are not indispensable for training.

	con.set_work_threads(8)

We can allocate several threads to sample positive and negative cases.


#### Step 2: Set configure parameters for training.

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

#### Step 3: Export results

	con.set_export_files("./res/model.vec.pt")

	con.set_out_files("./res/embedding.vec.json")

Model parameters will be exported via torch.save() automatically every few rounds. Also, they will be exported to json files finally. 

#### Step 4: Train models

	con.init()
	con.set_model(models.TransE)
	con.run()
  
We set the knowledge graph embedding model and start the training process.

### Testing

To evaluate the model, first import datasets and set essential configure paramters, then set model parameters and test the model. For instance, we write an example_test_transe.py to test TransE.

There are three approaches to test models:

(1) Set import files and OpenKE-PyTorch will automatically load models via torch.load().

	import config
	import models
	import numpy as np
	import json

	con = config.Config()
	con.set_in_path("./benchmarks/FB15K/")
	con.set_test_flag(True)
	con.set_work_threads(4)
	con.set_dimension(100)
	con.set_import_files("./res/model.vec.pt")
	con.init()
	con.set_model(models.TransE)
	con.test()

(2) Read model parameters from json files and manually load parameters.

	import config
	import models
	import numpy as np
	import json

	con = config.Config()
	con.set_in_path("./benchmarks/FB15K/")
	con.set_test_flag(True)
	con.set_work_threads(4)
	con.set_dimension(100)
	con.init()
	con.set_model(models.TransE)
	f = open("./res/embedding.vec.json", "r")
	content = json.loads(f.read())
	f.close()
	con.set_parameters(content)
	con.test()

(3) Manually load models via torch.load().

	import config
	import models
	import numpy as np
	import json

	con = config.Config()
	con.set_in_path("./benchmarks/FB15K/")
	con.set_test_flag(True)
	con.set_work_threads(4)
	con.set_dimension(100)
	con.init()
	con.set_model(models.TransE)
	con.import_variables("./res/model.vec.pt")
	con.test()

### Getting the embedding matrix

There are four approaches to get the embedding matrix.

(1) Set import files and OpenKE will automatically load models via torch.load().

	import json
	import numpy as py
	import config
	import models
	con = config.Config()
	con.set_in_path("./benchmarks/FB15K/")
	con.set_test_flag(True)
	con.set_work_threads(4)
	con.set_dimension(100)
	con.set_import_files("./res/model.vec.pt")
	con.init()
	con.set_model(models.TransE)
	# Get the embeddings (numpy.array)
	embeddings = con.get_parameters("numpy")
	# Get the embeddings (python list)
	embeddings = con.get_parameters()

(2) Read model parameters from json files and manually load parameters.

	import json
	import numpy as py
	import config
	import models
	con = config.Config()
	con.set_in_path("./benchmarks/FB15K/")
	con.set_test_flag(True)
	con.set_work_threads(4)
	con.set_dimension(100)
	con.init()
	con.set_model(models.TransE)
	f = open("./res/embedding.vec.json", "r")
	embeddings = json.loads(f.read())
	f.close()

(3) Manually load models via torch.load().

	con = config.Config()
	con.set_in_path("./benchmarks/FB15K/")
	con.set_test_flag(True)
	con.set_work_threads(4)
	con.set_dimension(100)
	con.init()
	con.set_model(models.TransE)
	con.import_variables("./res/model.vec.pt")
	# Get the embeddings (numpy.array)
	embeddings = con.get_parameters("numpy")
	# Get the embeddings (python list)
	embeddings = con.get_parameters()
  
(4) Immediately get the embeddings after training the model.

	...
	...
	...
	#Models will be exported via tf.Saver() automatically.
	con.set_export_files("./res/model.vec.pt")
	#Model parameters will be exported to json files automatically.
	con.set_out_files("./res/embedding.vec.json")
	#Initialize experimental settings.
	con.init()
	#Set the knowledge embedding model
	con.set_model(models.TransE)
	#Train the model.
	con.run()
	#Get the embeddings (numpy.array)
	embeddings = con.get_parameters("numpy")
	#Get the embeddings (python list)
	embeddings = con.get_parameters()

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

		#To import dataset from the benchmark folder
		def set_in_path(self, path)
		
		#To export model parameters to json files when training completed
		def set_out_files(self, path)
		
		#To set the import files, all parameters can be restored from the import files
		def set_import_files(self, path)
		
		#To set the export file of model paramters, and export results every few rounds
		def set_export_files(self, path, steps = 0)

		#To export results every few rounds
		def set_export_steps(self, steps)

		#To save model via torch.save()
		def save_pytorch(self)

		#To restore model via torch.load()
		def restore_pytorch(self)

		#To export model paramters, when path is none, equivalent to save_tensorflow()
		def export_variables(self, path = None)

		#To import model paramters, when path is none, equivalent to restore_tensorflow()
		def import_variables(self, path = None)
		
		#To export model paramters to designated path
		def save_parameters(self, path = None)

		#To manually load parameters which are read from json files
		def set_parameters(self, lists)
		
		#To get model paramters, if using mode "numpy", you can get np.array , else you can get python lists
		def get_parameters(self, mode = "numpy")
 
		#To set the knowledge embedding model
		set_model(model)
		
		#The framework will print loss values during training if flag = 1
		def set_log_on(flag = 1)

		#This is essential when testing
		def set_test_flag(self, flag)
	
Note that model paramters can be loaded only when all the configuration paramters are set.

### Model

	class Model(object)
	
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
		
		#To calulate the loss
		def forward(self)

		# To define loss functions for knowledge embedding models
		def loss_func()
		
		# To define the prediction functions for knowledge embedding models
		def predict(self)

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
	
	#The implementation for ComplEx
	class ComplEx(Model)
	
		
	
	
	

