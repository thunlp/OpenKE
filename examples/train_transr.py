import config
import models
import numpy as np
import json
#Train TransR based on pretrained TransE results.
#++++++++++++++TransE++++++++++++++++++++

con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("./benchmarks/FB15K/")
#True: Input test files from the same folder.
con.set_log_on(1)
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)	
con.set_alpha(0.001)
con.set_bern(0)
con.set_dimension(100)
con.set_margin(1.0)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")
#Model parameters will be exported via torch.save() automatically.
con.set_export_files("./res/transe.pt")
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/transe.vec.json")
con.init()
con.set_model(models.TransE)
con.run()
parameters = con.get_parameters("numpy")

#++++++++++++++TransR++++++++++++++++++++

conR = config.Config()
#Input training files from benchmarks/FB15K/ folder.
conR.set_in_path("./benchmarks/FB15K/")
#True: Input test files from the same folder.

conR.set_work_threads(4)
conR.set_train_times(1000)
conR.set_nbatches(100)
conR.set_alpha(0.001)
conR.set_bern(0)
conR.set_dimension(100)
conR.set_margin(1)
conR.set_ent_neg_rate(1)
conR.set_rel_neg_rate(0)
conR.set_opt_method("SGD")

#Models will be exported via tf.Saver() automatically.
conR.set_export_files("./res/transr.vec.tf")
#Model parameters will be exported to json files automatically.
conR.set_out_files("./res/transr.vec.json")
#Initialize experimental settings.
conR.init()
#Load pretrained TransE results.
conR.set_model(models.TransR)
parameters["transfer_matrix.weight"] = np.array([(np.identity(100).reshape((100*100))) for i in range(conR.get_rel_total())])
conR.set_parameters(parameters)
#Train the model.
conR.run()
