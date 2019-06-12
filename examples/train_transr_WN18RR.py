import config
import models
import tensorflow as tf
import numpy as np

#Train TransR based on pretrained TransE results.
#++++++++++++++TransE++++++++++++++++++++

con = config.Config()
con.set_in_path("./benchmarks/WN18RR/")
con.set_test_link_prediction(True)
con.set_work_threads(8)
con.set_train_times(100)
con.set_nbatches(100)
con.set_alpha(0.5)
con.set_margin(6.0)
con.set_bern(1)
con.set_dimension(50)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")
# con.set_export_files("./res/model1.vec.tf", 0)
# Model parameters will be exported to json files automatically.
# con.set_out_files("./res/embedding1.vec.json")
# con.set_import_files("./res/model1.vec.tf")
con.init()
con.set_model(models.TransE)
con.run()

parameters = con.get_parameters("numpy")
parameters["transfer_matrix"] = np.array([(np.identity(50).reshape((50*50))) for i in range(conR.get_rel_total())])
#++++++++++++++TransR++++++++++++++++++++

conR = config.Config()
#Input training files from benchmarks/FB15K/ folder.
conR.set_in_path("./benchmarks/WN18RR/")
#True: Input test files from the same folder.
conR.set_test_link_prediction(True)

conR.set_work_threads(8)
conR.set_train_times(200)
conR.set_nbatches(100)
conR.set_alpha(0.1)
conR.set_margin(6.0)
conR.set_bern(1)
conR.set_dimension(50)
conR.set_ent_neg_rate(1)
conR.set_rel_neg_rate(0)
conR.set_opt_method("SGD")

#Models will be exported via tf.Saver() automatically.
# conR.set_export_files("./res/model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
# conR.set_out_files("./res/embedding.vec.json")
#Initialize experimental settings.
conR.init()
#Load pretrained TransE results.
conR.set_model(models.TransR)
conR.set_parameters(parameters)
#Train the model.
conR.run()
#To test models after training needs "set_test_flag(True)".
conR.test()

