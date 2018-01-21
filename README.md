# OpenKE：一个高效的知识表示学习工具包
OpenKE ([http://openke.thunlp.org/](http://openke.thunlp.org/)) 是一个由清华大学自然语言处理与社会人文计算实验室研制推出的知识表示学习工具包，提供TransE、TransH、TransR、TransD、RESCAL、DistMult、HolE、ComplEx 等算法的统一接口的高效实现，同时为开发新的知识表示模型提供平台。

## 目录
* [项目介绍](#项目介绍)
* [编译安装](#编译安装)
* [数据介绍](#数据介绍)
* [评测结果](#评测结果)
* [快速开始](#快速开始)
* [接口介绍](#接口介绍)
* [作者信息](#作者信息)

## 项目介绍

OpenKE (An Open-source Framework for Knowledge Embedding) 有如下两个特点：

1. 使用 C++ 实现数据预处理和负采样等底层操作，支持多线程并行训练，并采用高效的负采样算法 (offset-based negtive sampling) 。

2. 提供 TensorFlow 和 PyTorch 两种框架训练模型，为部署 GPU 提供方便的平台。

OpenKE 包括以下三个库：

OpenKE: 这是本项目主要的库，快速稳定地实现知识图谱嵌入的经典模型。

<a href="https://github.com/thunlp/TensorFlow-TransX"> TensorFlow-TransX</a>: OpenKE 的轻量级简化版本，基于 Tensorflow 实现 TransE, TransH, TransR 和 TransD。

<a href="https://github.com/thunlp/Fast-TransX"> Fast-TransX</a>: 使用 C++ 快速简便实现 TransE, TransH, TransR, TransD, TranSparse 和 Fast-PTransE, 采用多线程加速训练。

## 编译安装

1. 安装 TensorFlow 或 PyTorch

2. 从 github 下载 OpenKE 项目

	$ git clone https://github.com/thunlp/OpenKE
	
	$ cd OpenKE

3. 编译 C++ 文件
	
	$ bash make.sh

## 数据介绍

数据集至少包含三个文件，格式如下：

triple2id.txt: 训练数据，第一行是三元组总数，接下来每行是描述 (e1, e2, rel) 的三元组。

entity2id.txt: 所有的实体和对应编号，第一行是实体总数，接下来每行是描述 (entity, id) 的二元组。

relation2id.txt: 所有的关系和对应编号，第一行是关系总数，接下来每行是描述 (relation, id) 的二元组。

## 评测结果

FB15K:

| Model | MeanRank(Raw)	| MeanRank(Filter)	| Hit@10(Raw)	| Hit@10(Filter)|
| ----- |:-------------:| :----------------:|:-----------:|:-------------:|
|TransE (n = 100, rounds = 1000)|220.03|70.33|50.22|75.50|
|TransH (n = 100, rounds = 1000)|219.40|69.36|50.42|75.50|
|TransR (n = 50, rounds = 1000)|224.93|79.92|42.00|59.84|
|TransD (n = 100, rounds = 1000)|225.37|74.58|50.31|75.32|
|RESCAL (n = 100, rounds = 1000)|263.72|160.95|37.32|52.68|
|DistMult (n = 100, rounds = 1000)|247.15|136.43|51.86|75.41|
|ComplEx (n = 100, rounds = 1000)|283.51|161.36|51.24|80.51|

## 快速开始

实现知识图谱嵌入时，首先导入数据集并设置超参数，然后训练模型并导出训练结果。以下的 example.py 展示如何实现 TransE:


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
	

### 第一步：导入数据集 

	con.set_in_path("benchmarks/FB15K/")
	con.set_out_path("benchmarks/FB15K/")
	
我们从 benchmarks/FB15K/ 文件夹导入知识图谱，数据集由三个文件组成：

*	triple2id.txt
*	entity2id.txt
*	relation2id.txt

验证集和测试集用来评估训练结果，训练模型时可以不用。

	con.set_work_threads(1)

我们可以分配多线程来采集正负样本。


### 第二步：设置模型的超参数

	con.set_train_times(500)
	con.set_nbatches(100)
	con.set_alpha(0.5)
	con.set_dimension(200)
	con.set_margin(1)

设置模型必要的超参数，包括数据训练回合 (train_times), 学习率 (learning rate), Batch 大小 (batch_size), 实体和关系嵌入的维度 (dimensions)。

	con.set_bern(0)
	con.set_ent_neg_rate(1)
	con.set_rel_neg_rate(0)

对于负采样，我们可以打破实体或关系构造负样本。 set\_bern(0) 将使用传统的负采样方法， set\_bern(1) 将使用 (Wang et al. 2014) 提出的方法 "bern"。
	
	con.set_optimizer("SGD")
	
我们可以选择合适的梯度下降优化算法来训练模型。

### 第三步：训练模型

	con.init()
	con.set_model(models.TransE)
	con.run()

设置模型并开始训练。
	
### 第四步：导出训练结果

	con.set_export_files("res/model.vec")

	con.set_export_steps(10)

每几个回合后，结果会被自动导出到指定的文件。

## 接口介绍

### Config
	
	class Config(object):
			
		#设置学习率
		def set_alpha(alpha = 0.001)
		
		#设置正则化参数
		def set_lmbda(lmbda = 0.0)
		
		#设置梯度下降优化算法 (SGD, Adagrad, Adadelta, Adam)
		def set_optimizer(optimizer = "SGD")
		
		#设置数据训练的次数
		def set_train_times(self, times)
		
		#把训练数据分为几批, nbatches 是批数
		def set_nbatches(nbatches = 100)
		
		#设置损失函数的 margin
		def set_margin(margin = 1.0)
		
		#同时设置实体和关系的维度
		def set_dimension(dim)
		
		#设置实体的维度
		def set_ent_dimension(self, dim)
		
		#设置关系的维度
		def set_rel_dimension(self, dim)
		
		#设置数据导入的文件路径		
		def set_in_path(path = "./")
		
		#设置导出结果的文件路径
		def set_out_path(path = "./")
		
		#为批样本采集设置线程数目
		def set_work_threads(threads = 1)
		
		#设置负样本采集算法, unif (bern = 0) 或 bern (bern = 1)
		def set_bern(bern = 1)
		
		#对于每个正样本，我们打破实体来构造负样本
		def set_ent_neg_rate(rate = 1)
		
		#对于每个正样本，我们打破关系来构造负样本
		def set_rel_neg_rate(rate = 0)
		
		#采集一批训练数据，包括正样本和负样本
		def sampling()
		
		#设置导入文件路径，所有的参数可以从导入文件中恢复
		def set_import_files(path = None)
		
		#设置导出文件路径
		def set_export_files(path = None)
		
		#设置每几个回合导出结果
		def set_export_steps(steps = 1)
		
		#设置待训练的知识图谱嵌入模型
		set_model(model)
		
		#如果 flag = 1, 将会在训练过程中打印 loss 
		def set_log_on(flag = 1)
	
	
	
	
	


### Model

#### TensorFlow 版本
	class Model(object)
	
		# 得到保存训练参数的 config
		get_config(self)
		
		# in_batch = True, 得到 [positive_head, positive_tail, positive_relation]
		# positive_head 的形状是 [batch_size, 1]
		# in_batch = False, 得到 [positive_head, positive_tail, positive_relation]
		# positive_head 的形状是 [batch_size]
		get_positive_instance(in_batch = True)
		
		# in_batch = True, 得到 [negative_head, negative_tail, negative_relation]
		# negtive_head 的形状是 [batch_size, negative_ent_rate + negative_rel_rate]
		# in_batch = False, 得到 [negative_head, negative_tail, negative_relation]
		# negtive_head 的形状是 [(negative_ent_rate + negative_rel_rate) * batch_size]		
		get_negative_instance(in_batch = True)

		# in_batch = True, 得到所有的训练样例，形状是 [batch_size, (1 + negative_ent_rate + negative_rel_rate)]
		# in_batch = False, 得到所有的训练样例，形状是 [(negative_ent_rate + negative_rel_rate + 1) * batch_size]
		def get_all_instance(in_batch = False)

		# in_batch = True, 得到所有的训练标签，形状是 [batch_size, (1 + negative_ent_rate + negative_rel_rate)]
		# in_batch = False, 得到所有的训练标签，形状是 [(negative_ent_rate + negative_rel_rate + 1) * batch_size]
		# 正样本的标签是1，负样本的标签是-1
		def get_all_labels(in_batch = False)
		
		# 定义训练样本的容器
		def input_def()
		
		# 定义训练模型的参数
		def embedding_def()

		# 定义模型的损失函数
		def loss_def()
		
		def __init__(config)

#### PyTorch 版本
	class Model(nn.Module):
		def __init__(self,config):
			super(Model,self).__init__()
			self.config=config
			
		# 得到 [positive_head, positive_tail, positive_relation]
		# positive_head 的形状是 [batch_size]
		def get_postive_instance(self):
			self.postive_h=Variable(torch.from_numpy(self.config.batch_h[0:self.config.batch_size]))
			self.postive_t=Variable(torch.from_numpy(self.config.batch_t[0:self.config.batch_size]))
			self.postive_r=Variable(torch.from_numpy(self.config.batch_r[0:self.config.batch_size]))
			return self.postive_h,self.postive_t,self.postive_r
			
		# 得到 [negative_head, negative_tail, negative_relation]
		# negtive_head 的形状是 [(negative_ent_rate + negative_rel_rate) * batch_size]
		def get_negtive_instance(self):
			self.negtive_h=Variable(torch.from_numpy(self.config.batch_h[self.config.batch_size:self.config.batch_seq_size]))
			self.negtive_t=Variable(torch.from_numpy(self.config.batch_t[self.config.batch_size:self.config.batch_seq_size]))
			self.negtive_r=Variable(torch.from_numpy(self.config.batch_r[self.config.batch_size:self.config.batch_seq_size]))
			return self.negtive_h,self.negtive_t,self.negtive_r
			
		# 得到所有的训练样例，形状是 [(negative_ent_rate + negative_rel_rate + 1) * batch_size]
		def get_all_instance(self):
			self.batch_h=Variable(torch.from_numpy(self.config.batch_h))
			self.batch_t=Variable(torch.from_numpy(self.config.batch_t))
			self.batch_r=Variable(torch.from_numpy(self.config.batch_r))
			return self.batch_h,self.batch_t,self.batch_r
			
		# 得到所有的训练标签，形状是 [(negative_ent_rate + negative_rel_rate + 1) * batch_size]
		# 正样本的标签是1，负样本的标签是-1
		def get_all_labels(self):
			self.batch_y=Variable(torch.from_numpy(self.config.batch_y))
			return self.batch_y
			
		# 定义每次训练执行的计算，会被每个子类重写
		def forward(self):
			pass
			
		# 损失函数
		def loss_func(self):
			pass

#### 模型实现

	#实现 TransE
	class TransE(Model)

	#实现 TransH	
	class TransH(Model)

	#实现 TransR
	class TransR(Model)

	#实现 TransD
	class TransD(Model)
	
	#实现 RESCAL
	class RESCAL(Model)
	
	#实现 DistMult
	class DistMult(Model)
	
	#实现 HolE
	class HolE(Model)					
	
	#实现 ComplEx
	class ComplEx(Model)
	
		
## 作者信息	


*	 [韩 旭](https://github.com/THUCSTHanxu13) 博士生
*	[林衍凯](http://thunlp.org/~lyk/) 博士生
*	[谢若冰](http://thunlp.org/~xrb/) 博士生
*	[曹书林](https://github.com/shelly-github) 访问学生
*	[刘知远](http://thunlp.org/~lzy/) 导师
*	孙茂松 导师
	
	
