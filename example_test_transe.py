import config
import models
import tensorflow as tf
import numpy as np
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
# (1) Set import files and OpenKE will automatically load models via tf.Saver().
con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
#con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(8)
con.set_dimension(100)
con.set_import_files("./res/model.vec.tf")
con.init()
con.set_model(models.TransE)
con.test()
con.predict_head_entity(152, 9, 5)
con.predict_tail_entity(151, 9, 5)
con.predict_relation(151, 152, 5)
con.predict_triple(151, 152, 9)
con.predict_triple(151, 152, 8)
#con.show_link_prediction(2,1)
#con.show_triple_classification(2,1,3)
# (2) Read model parameters from json files and manually load parameters.
# con = config.Config()
# con.set_in_path("./benchmarks/FB15K/")
# con.set_test_flag(True)
# con.set_work_threads(4)
# con.set_dimension(50)
# con.init()
# con.set_model(models.TransE)
# f = open("./res/embedding.vec.json", "r")
# content = json.loads(f.read())
# f.close()
# con.set_parameters(content)
# con.test()

# (3) Manually load models via tf.Saver().
# con = config.Config()
# con.set_in_path("./benchmarks/FB15K/")
# con.set_test_flag(True)
# con.set_work_threads(4)
# con.set_dimension(50)
# con.init()
# con.set_model(models.TransE)
# con.import_variables("./res/model.vec.tf")
# con.test()
