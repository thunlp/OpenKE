import config
import models
import tensorflow as tf

con = config.Config()
con.set_in_path("./data/FB15K/")
con.set_out_path("./data/FB15K/")
con.set_work_threads(1)
con.set_export_files("./res/model.vec")
con.set_export_steps(100)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(0.01)
con.set_bern(0)
con.set_dimension(100)
con.set_margin(1)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_optimizer("SGD")
con.init()
con.set_model(models.TransE)
con.run()
