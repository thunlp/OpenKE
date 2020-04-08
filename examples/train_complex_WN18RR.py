import openke
from openke.config import Trainer, Tester
from openke.module.model import ComplEx
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/WN18RR/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

# define the model
complEx = ComplEx(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200
)

# define the loss function
model = NegativeSampling(
	model = complEx, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 1.0
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 2000, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
trainer.run()
complEx.save_checkpoint('./checkpoint/complEx.ckpt')

# test the model
complEx.load_checkpoint('./checkpoint/complEx.ckpt')
tester = Tester(model = complEx, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)