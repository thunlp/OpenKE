import openke
from openke.config import Trainer, Tester
from openke.module.model import DistMult
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/WN18RR/", 
	batch_size = 2000,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

# define the model
distmult = DistMult(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 1024,
	margin = 200.0,
	epsilon = 2.0
)

# define the loss function
model = NegativeSampling(
	model = distmult, 
	loss = SigmoidLoss(adv_temperature = 0.5),
	batch_size = train_dataloader.get_batch_size(), 
	l3_regul_rate = 0.000005
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 400, alpha = 0.002, use_gpu = True, opt_method = "adam")
trainer.run()
distmult.save_checkpoint('./checkpoint/distmult.ckpt')

# test the model
distmult.load_checkpoint('./checkpoint/distmult.ckpt')
tester = Tester(model = distmult, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
