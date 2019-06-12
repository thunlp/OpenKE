import config
import models
import tensorflow as tf
import numpy as np

#Train TransR based on pretrained TransE results.
#++++++++++++++TransE++++++++++++++++++++

con = config.Config()
con.set_in_path("./benchmarks/FB15K237/")
con.set_test_link_prediction(True)
con.set_work_threads(8)
con.set_train_times(500)
con.set_nbatches(100)
con.set_alpha(0.5)
con.set_margin(5.0)
con.set_bern(1)
con.set_dimension(200)
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
# con.test()

parameters = con.get_parameters("numpy")

#++++++++++++++TransR++++++++++++++++++++

conR = config.Config()
#Input training files from benchmarks/FB15K/ folder.
conR.set_in_path("./benchmarks/FB15K237/")
#True: Input test files from the same folder.
conR.set_test_link_prediction(True)

conR.set_work_threads(8)
conR.set_train_times(1000)
conR.set_nbatches(100)
conR.set_alpha(1.0)
conR.set_margin(4.0)
conR.set_bern(1)
conR.set_dimension(200)
conR.set_ent_neg_rate(25)
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
parameters["transfer_matrix"] = np.array([(np.identity(200).reshape((200*200))) for i in range(conR.get_rel_total())])
conR.set_parameters(parameters)
#Train the model.
conR.run()
#To test models after training needs "set_test_flag(True)".
conR.test()

