import tensorflow as tf
import os
import  Alex3D as model
import numpy as np
from datetime import datetime
from AD_Dataset import  Dataset_Import
import AD_Constants as constant
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow
import  ops as op
import  time
import os
from tensorflow.python import debug as tf_debug

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class Main_run(Dataset_Import):


    def __init__(self):
        super().__init__()
        self.now = datetime.now()
        self.training_epoch = 50
        self.auto_encode_epoch = 10
        self.batch_size = 1
        self.validation_interval = 20
        self.dropout_prob = 0.60
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
        self.label_cnt = 3  # number of classes

    def train(self):
        # build graph

        inputs, labels, dropout_keep_prob, learning_rate,domain_label,flip_grad = model.input_placeholder(self.image_size, self.img_channel, self.label_cnt)

        autoencoder_run=model.autoencoder(inputs,self.batch_size)

        # autoencoder loss
        autoencoder_loss = model.loss_autoencoder(inputs, autoencoder_run)



        #tf.train.RMSPropOptimizer(learning_rate,self.rms_decay).minimize(total_loss)



        # train = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        train_autoencode = tf.train.AdamOptimizer(learning_rate).minimize(autoencoder_loss)
        #train = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)


        #for layer_wise learning rate

       #tf.train.RMSPropOptimizer(learning_rate,self.rms_decay).minimize(total_loss)#
        #tf.train.AdamOptimizer(self.learning_rate_convlayer).minimize(
        total_batch = int(len(self.source_target_combined_2()) / self.batch_size)



        saver = tf.train.Saver()
        total_batch_auto=int(len(self.source_target_combined())/self.batch_size)

        merged_summary = tf.summary.merge_all()

        with tf.Session() as encoder_sess:

            # encoder_sess = tf_debug.TensorBoardDebugWrapperSession(
            #     encoder_sess, "http://LAPTOP-P0RH6OCI:6006"
            # )

            sum_writer=tf.summary.FileWriter(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "summaries\\auto"),encoder_sess.graph)
            encoder_sess.run(tf.global_variables_initializer())
            #self.training_epoch

            print("Autoencoder Pretraining ...",end="\n")


            for encoder_epoch in range(10):

              start_time = time.time()
              for i in range(20):

                    input_feed= self.next_batch_combined_encoder(self.batch_size)

                    auto_encode_feed_dict = {inputs:input_feed,learning_rate: self.learn_rate_auto,dropout_keep_prob:self.dropout_prob}
                    _,encoder_loss,summary_out=encoder_sess.run([train_autoencode,autoencoder_loss,merged_summary],feed_dict=auto_encode_feed_dict)


                    #saver.save(encoder_sess, 'np_data/model_iter', global_step=i)
              sum_writer.add_summary(summary_out,global_step=encoder_epoch)
              end_time = time.time()
              print(" ", end="\n")
              print("Epoch " + str(encoder_epoch + 1) + " completed : Time usage " + str(int(end_time - start_time)) + " seconds")
              print("Reconstruction Loss: {:.6f}".format(encoder_loss))
              self.i = 0
              self.auto_shuffling_state = True
              self.shu_control_state=True
              self.set_epoch = encoder_epoch

              # Save the final model
            saver.save(encoder_sess, 'np_data/model_final')
            self.set_epoch=0
            self.auto_shuffling_state=False
            self.shu_control_state=False

        exit(1)
        tf.reset_default_graph()

        g2 = tf.Graph()
        # Merge all summaries together

        with g2.as_default() as g3:


                inputs, labels, dropout_keep_prob, learning_rate, domain_label, flip_grad = model.input_placeholder(
                self.image_size, self.img_channel, self.label_cnt)

                logits = model.inference(inputs, dropout_keep_prob, self.label_cnt)

                accuracy = model.accuracy(logits, labels)
                pred_loss = model.loss(logits, labels)

                domain_logits = model.domain_parameters(flip_grad)
                domain_accuracy = model.domain_accuracy(domain_logits, domain_label)
                domain_loss = model.domain_loss(domain_logits, domain_label)

                total_loss = pred_loss + domain_loss

                convolve_variables = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="convolution/"+con_var) for con_var in
                                      constant.pre_trained]

                fully_variables = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="convolution/"+fc_var) for fc_var in
                                   constant.not_trained]

                # train = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
                train_op_trained = tf.train.RMSPropOptimizer(self.learning_rate_convlayer, self.rms_decay).minimize(total_loss,
                                                                                                                    var_list=convolve_variables)
                train_op_untrained = tf.train.RMSPropOptimizer(self.learn_rate_fully_layer, self.rms_decay).minimize(total_loss,
                                                                                                                     var_list=fully_variables)
                train_op = tf.group(train_op_trained, train_op_untrained)

                tr_merge_summary = tf.summary.merge_all()

        tf.reset_default_graph()
        with tf.Session(graph=g3) as  tr_sess:

            imported_meta = tf.train.import_meta_graph("np_data/model_final.meta")
            tr_writer = tf.summary.FileWriter(
                os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "summaries\\train"),tr_sess.graph)

            tr_sess.run(tf.global_variables_initializer())

            #summ_writer = tf.summary.FileWriter(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "summaries"),tr_sess.graph)
            imported_meta.restore(tr_sess, tf.train.latest_checkpoint('np_data'))
            latest_ckp = tf.train.latest_checkpoint('np_data')

            reader = pywrap_tensorflow.NewCheckpointReader(latest_ckp )
            var_to_shape_map = reader.get_variable_to_shape_map()
            #load autoencoder pretrained weights and biase
            op.load_initial_weights(tr_sess,var_to_shape_map,use_pretrain=True)

            print(" ",end="\n")
            print("Initializing Class Training")


            for epoch in range(10):

              start_time_2=time.time()
              for i in range(4):

                    data_feed, data_label, label_domain = self.next_batch_combined(self.batch_size)

                    feed_dict_source_batch = {inputs: data_feed, labels: data_label,
                                              dropout_keep_prob: self.dropout_prob,
                                              learning_rate: self.learn_rate_pred, domain_label: label_domain, flip_grad: 1.0}
                    _, loss_source_batch, acc_source_batch, accuracy_domain, loss_domain, log_d,acc_log,summary_out2= tr_sess.run(
                        [train_op, total_loss, accuracy, domain_accuracy, domain_loss, domain_logits,logits,tr_merge_summary],
                        feed_dict=feed_dict_source_batch)

              tr_writer.add_summary(summary_out2,epoch)

              end_time_2 = time.time()
              print(" ",end="\n")
              print("Epoch " + str(epoch + 1) + " completed : Time usage " + str(int(end_time_2 - start_time_2)) + " seconds")
              print("Training total_loss: {:.5f}".format(loss_source_batch))
              print("Training class_accuracy: {0:.3f}".format(acc_source_batch))
              print("Training domain_loss: {:.5f}".format(loss_domain))
              print("Training domain_accuracy: {0:.3f}".format(accuracy_domain))
              #print("label ",data_label)
              #print("logit ",acc_log)
              self.trainer_shuffling_state = True
              self.shu_control_state = True
              self.set_epoch =epoch
              self.i=0



            validation_source_dataset, valid_source_label, valid_source_d_label = self.convert_validation_source_data()
            validation_target_dataset, valid_target_label, valid_target_d_label = self.convert_validation_target_data()

            acc_source_valid = tr_sess.run(accuracy,
                                                feed_dict={inputs: validation_source_dataset,
                                                labels: valid_source_label,
                                                dropout_keep_prob: 1.0})

            acc_target_valid = tr_sess.run(accuracy,
                                          feed_dict={inputs: validation_target_dataset,
                                                     labels:valid_target_label,
                                                     dropout_keep_prob: 1.0})

            valid_data_feed = np.vstack([validation_source_dataset, validation_target_dataset])
            valid_data_label = np.vstack([valid_source_label, valid_target_label])
            valid_data_d_label = np.vstack([valid_source_d_label, valid_target_d_label])

            validation_accuracy,acc_domain = tr_sess.run([accuracy,domain_accuracy],
                                            feed_dict={inputs: valid_data_feed,
                                                       labels: valid_data_label,
                                                       dropout_keep_prob: 1.0,domain_label:valid_data_d_label, flip_grad: 0})


            print(" ", end="\n")
            print("-------Validation----------", end="\n")
            print("Validation   accuracy: {0:.2f}".format(validation_accuracy))
            print("Validation source  accuracy: {0:.2f}".format(acc_source_valid))
            print("Validation target  accuracy: {0:.2f}".format(acc_target_valid))
            print("Validation domain  accuracy: {0:.2f}".format(acc_domain))




if __name__ == '__main__':

  #try:
    run_train=Main_run()
    run_train.train()

  #except Exception as ex:
   # print("Exeception caught ",ex)