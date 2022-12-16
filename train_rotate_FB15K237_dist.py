import openke
from openke.config import Trainer, Tester
from openke.module.model import DistributedRotatE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import torch
import torch.distributed as dist
import argparse
import time
import numpy as np
import os

torch.manual_seed(1234)
np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
args = parser.parse_args()
local_rank = args.local_rank

total_start_time = time.time()
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', )
world_size = dist.get_world_size()

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K237/", 
	batch_size = 2000,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
rotate = DistributedRotatE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 1024,
	margin = 6.0,
	epsilon = 2.0,
	world_size = world_size
)

# define the loss function
model = NegativeSampling(
	model = rotate, 
	loss = SigmoidLoss(adv_temperature = 2),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 6000, alpha = 2e-5, use_gpu = True, opt_method = "adam", save_steps=10000)
trainer.run()
total_end_time = time.time()
with open("./total_cost_time_rotate.txt", "a+") as f:
	f.writelines(["total elapsed time {}".format(total_end_time - total_start_time)])
rotate.save_checkpoint('./checkpoint/rotate.ckpt')

# test the model
rotate.load_checkpoint('./checkpoint/rotate.ckpt')
tester = Tester(model = rotate, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)