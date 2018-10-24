import os
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader
import os
from ARCH_3D import  Alex3D as model
import numpy as np
from datetime import datetime
from AD_Dataset import  Dataset_Import
from tensorflow.contrib.tensorboard.plugins import projector
import AD_Constants as constant
from ARCH_3D import ops as op_linker
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class Main_run(Dataset_Import):


    def __init__(self):
        super().__init__()
        self.now = datetime.now()
        self.training_epoch = 500
        self.auto_encode_epoch = 500
        self.batch_size = 2
        self.validation_interval = 20
        self.dropout_prob = 0.50
        self.learn_rate_auto = 0.001
        self.learn_rate_pred = 0.00001
        self.learning_rate_convlayer=0.00001
        self.learn_rate_fully_layer=0.0001
        self.rms_decay = 0.9
        self.weight_decay = 0.0005
        self.is_train = True
       # 0.0005, 0.001, 0.00146


        # Data info.
        self.image_size =constant.img_shape_tuple[0]  # image size
        self.img_channel = constant.img_channel  # number of channels (1 for black & white)
        self.label_cnt = len(constant.classify_group)  # number of classes

    def train(self, train_type: str="domain", use_encoder_saver: bool=False, use_train_saver: bool=False):


            g2 = tf.Graph()
            # Merge all summaries together


            with g2.as_default() as g3:


                    inputs, labels, training, dropout_keep_prob, learning_rate, domain_label, flip_grad = \
                        model.input_placeholder(self.image_size, self.img_channel, self.label_cnt)

                    logits = model.vgg16(inputs, training, dropout_keep_prob, self.label_cnt)

                    accuracy = model.accuracy(logits, labels)
                    pred_loss = model.loss(logits, labels)


                    if train_type=="single":
                        total_loss = pred_loss * 0.01
                        fully_variables = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="convolution/" + fc_var)
                                           for fc_var in
                                           constant.not_trained if fc_var != "domain_predictor"]

                    convolve_variables = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="convolution/"+con_var) for con_var in
                                          constant.pre_trained]




                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    # train = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
                    with tf.control_dependencies(update_ops):
                        # train = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
                        train_op_trained = tf.train.MomentumOptimizer(self.learning_rate_convlayer, self.rms_decay).minimize(total_loss)#var_list=convolve_variables

                    train_op_untrained = tf.train.MomentumOptimizer(self.learn_rate_fully_layer, self.rms_decay).minimize(total_loss)# var_list=fully_variables

                    train_op = tf.group(train_op_trained, train_op_untrained)

                    tr_merge_summary = tf.summary.merge_all()




            with tf.Session(graph=g3) as  tr_sess:

                if use_train_saver == False:




                    print(" ", end="\n")
                    print("Initializing Class Training")
                    print(" ", end="\n")

                    for epoch in range(2):

                      start_time_2=time.time()
                      for i in range(2):

                            source_feed=self.next_batch_combined(self.batch_size)

                            data_source = [data for data in source_feed]
                            data_source = np.array(data_source)
                            data_feed=list(data_source[0:, 0])
                            data_label=list(data_source[0:, 1])

                            if train_type=="single":
                                feed_dict_source_batch = {inputs: data_feed, labels: data_label, training: True,
                                                          dropout_keep_prob: self.dropout_prob,
                                                          learning_rate: self.learn_rate_pred,
                                                          flip_grad: 1.0}

                                _, loss_source_batch, acc_source_batch, acc_log, summary_out2 = tr_sess.run(
                                    [train_op, total_loss, accuracy,logits,
                                     tr_merge_summary],
                                    feed_dict=feed_dict_source_batch)
                            #print(data_label)
                            #print(acc_log)
                            print('Epoch %d/%d, batch %d/%d is finished!' % (epoch, self.training_epoch, i, total_batch))

                      tr_writer.add_summary(summary_out2,epoch)



                      end_time_2 = time.time()

                      print(" ",end="\n")
                      print("Epoch " + str(epoch + 1) + " completed : Time usage " + str(int(end_time_2 - start_time_2)) + " seconds")
                      print("Training total_loss: {:.5f}".format(loss_source_batch))
                      print("Training class_accuracy: {0:.3f}".format(acc_source_batch))
                      print(" ", end="\n")

                      if train_type == "domain":
                        print("Training domain_loss: {:.5f}".format(loss_domain))
                        print("Training domain_accuracy: {0:.3f}".format(accuracy_domain))

                      self.trainer_shuffling_state = True
                      self.shu_control_state = True
                      self.set_epoch =epoch
                      self.i=0




                  # validation batch

                if use_train_saver==True:
                    print("Validating Data from Saved Model ",end="\n")
                    train_saver = tf.train.Saver()
                    train_saver.restore(tr_sess,  tf.train.latest_checkpoint('train_session'))





                if train_type == "single":
                    max_iteration= int(len(self.source_validation_data()) /self.batch_size)
                v_accuracy_list=[]
                for steps in range(max_iteration):



                      vsource_feed= self.convert_validation_source_data(self.batch_size)

                      data_vsource = [data for data in vsource_feed]
                      data_vsource = np.array(data_vsource)
                      validation_source_dataset = list(data_vsource[0:, 0])
                      valid_source_label = list(data_vsource[0:, 1])
                      valid_source_d_label = list(data_vsource[0:, 2])





                      acc_source_valid = tr_sess.run(accuracy,
                                                     feed_dict={inputs: validation_source_dataset,
                                                                labels: valid_source_label,
                                                                training: False,
                                                                dropout_keep_prob: 1.0})




                      if train_type== "single":

                          validation_accuracy= tr_sess.run(accuracy,
                                                                        feed_dict={inputs: validation_source_dataset,
                                                                                   labels: valid_source_label,
                                                                                   training: False,
                                                                                   dropout_keep_prob: 1.0,
                                                                                   flip_grad: 0})



                      print(" ", end="\n")
                      print("-------Validation ",steps+1,"----------", end="\n")
                      print("Validation   accuracy: {0:.2f}".format(validation_accuracy))

                      v_accuracy_list.append(round(validation_accuracy,2))

                      if train_type == "domain":
                        print("Validation source  accuracy: {0:.2f}".format(acc_source_valid))
                        print("Validation target  accuracy: {0:.2f}".format(acc_target_valid))
                        print("Validation domain  accuracy: {0:.2f}".format(acc_domain))


                print(" ",end="\n")
                print("Average Validation Accuracy ",round(sum(v_accuracy_list)/len(v_accuracy_list),2))
                      # end of validation batch


def netTrain(net_type:str=None):
    pass

if __name__ == '__main__':

  try:
   run_train = Main_run()
   run_train.train(train_type="single", use_encoder_saver=False, use_train_saver=False)

  except Exception as ex:
    print("Exeception caught ",ex)
    raise



