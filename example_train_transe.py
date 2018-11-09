import config
import models
import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
#Input training files from benchmarks/FB15K/ folder.
con = config.Config()
#True: Input test files from the same folder.
con.set_in_path("./benchmarks/FB15K/")
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

#Models will be exported via tf.Saver() automatically.
con.set_export_files("./res/model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.TransE)
#Train the model.
con.run()
#To test models after training needs "set_test_flag(True)".
con.test()
con.predict_head_entity(152, 9, 5)
con.predict_tail_entity(151, 9, 5)
con.predict_relation(151, 152, 5)
con.predict_triple(151, 152, 9)
con.predict_triple(151, 152, 8)
