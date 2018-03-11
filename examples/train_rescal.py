import config
import models
import json

con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("./benchmarks/FB15K/")
#True: Input test files from the same folder.
con.set_log_on(1)
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)	
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(50)
con.set_margin(1.0)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
#Model parameters will be exported via torch.save() automatically.
con.set_export_files("./res/rescal.pt")
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/rescal.vec.json")
con.init()
con.set_model(models.RESCAL)
con.run()

