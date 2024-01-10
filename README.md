# OpenKE (sub-project of OpenSKL)

OpenKE is a sub-project of OpenSKL, providing an **Open**-source **K**nowledge **E**mbedding toolkit for knowledge representation learning (KRL), with <a href="https://ojs.aaai.org/index.php/AAAI/article/view/9491/9350"> TransR</a> and  <a href="https://aclanthology.org/D15-1082.pdf">PTransE</a> as key features to handle complex relations and relational paths in large-scale knowledge graphs.

## Overview

OpenKE is an efficient implementation based on PyTorch for knowledge embedding. We use C++ to implement some underlying operations such as data preprocessing and negative sampling. For each specific model, it is implemented by PyTorch with Python interfaces so that there is a convenient platform to run models on GPUs. OpenKE contains 4 repositories:

[OpenKE-PyTorch](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch): the repository based on PyTorch, which provides the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/OpenKE/tree/OpenKE-Tensorflow1.0"> OpenKE-Tensorflow1.0</a>: OpenKE implemented with TensorFlow, also providing the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/TensorFlow-TransX"> TensorFlow-TransX</a>: light and simple version of OpenKE based on TensorFlow, including TransE, TransH, TransR and TransD. 

<a href="https://github.com/thunlp/Fast-TransX"> Fast-TransX</a>: efficient lightweight C++ inferences for TransE and its extended models utilizing the framework of OpenKE, including TransH, TransR, TransD, TranSparse and PTransE. 

More information (especially the embedding databases of popular knowledge graphs obtained by OpenKE and related documents) is available on our website 
[http://openke.thunlp.org/](http://openke.thunlp.org/)

## Models
Besides our proposed TransR and PTransE, we also support the following typical knowledge embedding models:

OpenKE (PyTorch): 

*	 RESCAL
*  DistMult, ComplEx, Analogy
*  TransE, TransH, TransR, TransD
*  SimplE
*	 RotatE

OpenKE (Tensorflow): 

*	 RESCAL, HolE
*  DistMult, ComplEx, Analogy
*  TransE, TransH, TransR, TransD

TensorFlow-TransX (TensorFlow):

*  TransE, TransH, TransR, TransD

Fast-TransX (C++):

*  TransE, TransH, TransR, TransD, TranSparse, PTransE

We welcome any issues and requests for model implementation and bug fix.

## Evaluation

To validate the effectiveness of this toolkit, we employ the link prediction task on large-scale knowledge graphs for evaluation.

### Settings
For each test triplet, the head is removed and replaced by each of the entities from the entity set in turn. The scores of those corrupted triplets are first computed by the models and then sorted by the order. Then, we get the rank of the correct entity. This whole procedure is also repeated by removing those tail entities. We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10, Hits@3, Hits@1). The mean rank (MR) and mean reciprocal rank (MRR) of the test triplets under this setting are also reported.

Because some corrupted triplets may be in the training set and validation set. In this case, those corrupted triplets may be ranked above the test triplet, but this should not be counted as an error because both triplets are true. Hence, we remove those corrupted triplets appearing in the training, validation or test set, which ensures the corrupted triplets are not in the dataset. We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10 (filter), Hits@3(filter), Hits@1(filter)) under this setting. The mean rank (MR (filter)) and mean reciprocal rank (MRR (filter)) of the test triplets under this setting are also reported.

More details of the above-mentioned settings can be found from the papers [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf), [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf).

For those large-scale entity sets, to corrupt all entities with the whole entity set is time-costing. Hence, we also provide the experimental setting named "[type constraint](https://www.dbs.ifi.lmu.de/~krompass/papers/TypeConstrainedRepresentationLearningInKnowledgeGraphs.pdf)" to corrupt entities with some limited entity sets determining by their relations.

### Results

We have provided the hyper-parameters of some models to achieve the state-of-the-art performace (Hits@10 (filter)) on FB15K237 and WN18RR. These scripts can be founded in the folder "./examples/". The results of these models are as follows: the left two columns are the performance implemented by OpenKE, and the right two columns are the performance reported in the original papers. Overall, OpenKE can reproduce the results in the original papers. 

|Model			|	WN18RR	|	FB15K237	| WN18RR (Paper\*)| FB15K237  (Paper\*)|
|:-:		|:-:	|:-:  |:-:  |:-:  |
|TransE (2013)	|0.512	|0.476|0.501|0.486|
|TransH (2014)	|0.507	|0.490|-|-|
|TransR (2015)	|0.519	|0.511|-|-|
|TransD (2015)	|0.508	|0.487|-|-|
|DistMult (2014)	|0.479	|0.419|0.49|0.419|
|ComplEx (2016)	|0.485	|0.426|0.51|0.428|
|ConvE	(2017)	|0.506	|0.485|0.52|0.501|
|RotatE (2019)	|0.549	|0.479|-|0.480|
|RotatE+adv (2019)	|0.565	|0.522|0.571|0.533|

RotatE has the best performance by representing knowledge in complex space. Our proposed TransR has the second best performance, and the real-valued representations learned by TransR can be more easily integrated with other neural network models, e.g. pre-trained language models. Please refer to our another toolkit [Knowledge-Plugin](https://github.com/THUNLP/Knowledge-Plugin) for such integration.

## Usage

### Installation

1. Install [PyTorch](https://pytorch.org/get-started/locally/)

2. Clone the OpenKE-PyTorch branch:
```bash
git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE --depth 1
cd OpenKE
cd openke
```
3. Compile C++ files
```bash
bash make.sh
```
4. Quick Start
```bash
cd ../
cp examples/train_transe_FB15K237.py ./
python train_transe_FB15K237.py
```
### Data Format

* For training, datasets contain three files:

  train2id.txt: training file, the first line is the number of triples for training. Then the following lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2*** .
  **Note that train2id.txt contains ids from entitiy2id.txt and relation2id.txt instead of the names of the entities and relations. If you use your own datasets, please check the format of your training file. Files in the wrong format may cause segmentation fault.**

  entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

  relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

* For testing, datasets contain additional two files (totally five files):

  test2id.txt: testing file, the first line is the number of triples for testing. Then the following lines are all in the format ***(e1, e2, rel)*** .

  valid2id.txt: validating file, the first line is the number of triples for validating. Then the following lines are all in the format ***(e1, e2, rel)*** .

  type_constrain.txt: type constraining file, the first line is the number of relations. Then the following lines are type constraints for each relation. For example, the relation with id 1200 has 4 types of head entities, which are 3123, 1034, 58 and 5733. The relation with id 1200 has 4 types of tail entities, which are 12123, 4388, 11087 and 11088. You can get this file through **n-n.py** in folder benchmarks/FB15K .

## Citation

If you find OpenKE is useful for your research, please consider citing the following papers:

```
 @inproceedings{han2018openke,
   title={OpenKE: An Open Toolkit for Knowledge Embedding},
   author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
   booktitle={Proceedings of EMNLP},
   year={2018}
 }
```

This package is mainly contributed (in chronological order) by [Xu Han](https://github.com/THUCSTHanxu13), [Yankai Lin](https://github.com/Mrlyk423), [Ruobing Xie](http://nlp.csai.tsinghua.edu.cn/~xrb/), [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/), [Xin Lv](https://github.com/davidlvxin), [Shulin Cao](https://github.com/ShulinCao), [Weize Chen](https://github.com/chenweize1998), [Jingqin Yang](https://github.com/yjqqqaq).

******************
## About OpenSKL
OpenSKL project aims to harness the power of both structured knowledge and natural languages via representation learning. All sub-projects of OpenSKL, under the categories of **Algorithm**, **Resource** and **Application**, are as follows.

- **Algorithm**: 
  - [OpenKE](https://www.github.com/thunlp/OpenKE)
    - An effective and efficient toolkit for representing structured knowledge in large-scale knowledge graphs as embeddings, with <a href="https://ojs.aaai.org/index.php/AAAI/article/view/9491/9350"> TransR</a> and  <a href="https://aclanthology.org/D15-1082.pdf">PTransE</a> as key features to handle complex relations and relational paths.
    - This toolkit also includes three repositories:
       - [KB2E](https://www.github.com/thunlp/KB2E)
       - [TensorFlow-Transx](https://www.github.com/thunlp/TensorFlow-Transx)
       - [Fast-TransX](https://www.github.com/thunlp/Fast-TransX)
  - [ERNIE](https://github.com/thunlp/ERNIE)
    - An effective and efficient toolkit for augmenting pre-trained language models with knowledge graph representations.
  - [OpenNE](https://www.github.com/thunlp/OpenNE)
    - An effective and efficient toolkit for representing nodes in large-scale graphs as embeddings, with [TADW](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) as key features to incorporate text attributes of nodes.
  - [OpenNRE](https://www.github.com/thunlp/OpenNRE)
    - An effective and efficient toolkit for implementing neural networks for extracting structured knowledge from text, with [ATT](https://aclanthology.org/P16-1200.pdf) as key features to consider relation-associated text information.
    - This toolkit also includes two repositories:
      - [JointNRE](https://www.github.com/thunlp/JointNRE)
      - [NRE](https://github.com/thunlp/NRE)
- **Resource**:
  - The embeddings of large-scale knowledge graphs pre-trained by OpenKE, covering three typical large-scale knowledge graphs: Wikidata, Freebase, and XLORE. The embeddings are free to use under the [MIT license](https://opensource.org/license/mit/), and please click the following link to submit [download requests](http://139.129.163.161/download/wikidata).
  - OpenKE-Wikidata
    - Wikidata is a free and collaborative database, collecting structured data to provide support for Wikipedia. The original Wikidata contains 20,982,733 entities, 594 relations and 68,904,773 triplets. In particular, Wikidata-5M is the core subgraph of Wikidata, containing  5,040,986 high-frequency entities from Wikidata with their corresponding 927 relations and 24,267,796 triplets.
    - [TransE version](http://139.129.163.161/download/wikidata): Knowledge embeddings of Wikidata pre-trained by OpenKE. 
    - [TransR version](http://139.129.163.161/download/wikidata) of Wikidata-5M: Knowledge embeddings of Wikidata-5M pre-trained by OpenKE.
  - OpenKE-Freebase
    - Freebase was a large collaborative knowledge base consisting of data composed mainly by its community members. It was an online collection of structured data harvested from many sources. Freebase contains 86,054,151 entities, 14,824 relations and 338,586,276 triplets.
    - [TransE version](http://139.129.163.161/download/wikidata): Knowledge embeddings of Freebase pre-trained by OpenKE. 
  - OpenKE-XLORE
    - XLORE is one of the most popular Chinese knowledge graphs developed by THUKEG. XLORE contains 10,572,209 entities, 138,581 relations and 35,954,249 triplets.
    - [TransE version](http://139.129.163.161/download/wikidata): Knowledge embeddings of XLORE pre-trained by OpenKE.
- **Application**:   
    - [Knowledge-Plugin](https://github.com/THUNLP/Knowledge-Plugin)
      - An effective and efficient toolkit of plug-and-play knowledge injection for pre-trained language models. Knowledge-Plugin is general for all kinds of knowledge graph embeddings mentioned above. In the toolkit, we plug the TransR version of Wikidata-5M into BERT as an example of applications. With the TransR embedding, we enhance the knowledge ability of BERT without fine-tuning the original model, e.g., up to 8% improvement on question answering.
