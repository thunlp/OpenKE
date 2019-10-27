# OpenKE-PyTorch

An Open-source Framework for Knowledge Embedding implemented with PyTorch.

More information is available on our website 
[http://openke.thunlp.org/](http://openke.thunlp.org/)

If you use the code, please cite the following [paper](http://aclweb.org/anthology/D18-2024):

```
 @inproceedings{han2018openke,
   title={OpenKE: An Open Toolkit for Knowledge Embedding},
   author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
   booktitle={Proceedings of EMNLP},
   year={2018}
 }
```

## Overview

This is an Efficient implementation based on PyTorch for knowledge representation learning (KRL). We use C++ to implement some underlying operations such as data preprocessing and negative sampling. For each specific model, it is implemented by PyTorch with Python interfaces so that there is a convenient platform to run models on GPUs. OpenKE composes 4 repositories:

OpenKE-PyTorch: the project based on PyTorch, which provides the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/OpenKE/tree/OpenKE-Tensorflow1.0"> OpenKE-Tensorflow1.0</a>: OpenKE implemented with TensorFlow, also providing the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/TensorFlow-TransX"> TensorFlow-TransX</a>: light and simple version of OpenKE based on TensorFlow, including TransE, TransH, TransR and TransD. 

<a href="https://github.com/thunlp/Fast-TransX"> Fast-TransX</a>: efficient lightweight C++ inferences for TransE and its extended models utilizing the framework of OpenKE, including TransH, TransR, TransD, TranSparse and PTransE. 


*** **UPDATE** ***

We are now developing a new version of OpenKE-PyTorch. The project has been completely reconstructed and is faster, more extendable and the codes are easier to read and use now. If you need get to the old version, please refer to branch [OpenKE-PyTorch(old)](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)).

*** **New Features** ***

- RotatE
- More enhancing strategies (e.g., adversarial training)
- More scripts of the typical models for the benchmark datasets.
- More extendable interfaces


## Models

OpenKE (Tensorflow): 

*	RESCAL, HolE
*  DistMult, ComplEx, Analogy
*  TransE, TransH, TransR, TransD

OpenKE (PyTorch): 

*	RESCAL
*  DistMult, ComplEx, Analogy
*  TransE, TransH, TransR, TransD
*  SimplE
*	RotatE

We welcome any issues and requests for model implementation and bug fix.

## Experiments

We have provided the hyper-parameters of some models to achieve the state-of-the-art performace on FB15K237 and WN18RR. These scripts can be founded in the folder "./examples/". Up to now, these models include TransE, TransH, TransR, TransD, DistMult, ComplEx. The results of these models are as follows,

|Model			|	WN18RR	|	FB15K237	| WN18RR (Paper\*)| FB15K237  (Paper\*)|
|:-:		|:-:	|:-:  |:-:  |:-:  |
|TransE	|0.512	|0.476|0.501|0.486|
|TransH	|0.507	|0.490|-|-|
|TransR	|0.519	|0.511|-|-|
|TransD	|0.508	|0.487|-|-|
|DistMult	|0.479	|0.419|0.49|0.419|
|ComplEx	|0.485	|0.426|0.51|0.428|
|ConvE		|0.506	|0.485|0.52|0.501|
|RotatE	|0.549	|0.479|-|0.480|
|RotatE (+adv)	|0.565	|0.522|0.571|0.533|


<strong> We are still trying more hyper-parameters and more training strategies (e.g., adversarial training and label smoothing regularization) for these models. </strong> Hence, this table is still in change. We welcome everyone to help us update this table and hyper-parameters.


## Installation

1. Install PyTorch

2. Clone the OpenKE-PyTorch branch:

	$ git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE
	
	$ cd OpenKE

3. Compile C++ files
	
	$ bash make.sh

## Data

* For training, datasets contain three files:

  train2id.txt: training file, the first line is the number of triples for training. Then the following lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2*** .
  **Note that train2id.txt contains ids from entitiy2id.txt and relation2id.txt instead of the names of the entities and relations. If you use your own datasets, please check the format of your training file. Files in the wrong format may cause segmentation fault.**

  entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

  relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

* For testing, datasets contain additional two files (totally five files):

  test2id.txt: testing file, the first line is the number of triples for testing. Then the following lines are all in the format ***(e1, e2, rel)*** .

  valid2id.txt: validating file, the first line is the number of triples for validating. Then the following lines are all in the format ***(e1, e2, rel)*** .

  type_constrain.txt: type constraining file, the first line is the number of relations. Then the following lines are type constraints for each relation. For example, the relation with id 1200 has 4 types of head entities, which are 3123, 1034, 58 and 5733. The relation with id 1200 has 4 types of tail entities, which are 12123, 4388, 11087 and 11088. You can get this file through **n-n.py** in folder benchmarks/FB15K
