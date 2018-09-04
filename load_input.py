# loading the data sets (training, validation, test)
# each input set must be: (#images, dim1, dim2, dim3, 1)
# The corresponding label must be: (#images, #classes)
import numpy as np
from AD_Dataset import  Dataset_Import

dataset_feed=Dataset_Import()

def load_train_data(image_size, num_channels, label_cnt,batch_size):

    train_dataset,train_labels =dataset_feed.next_batch(batch_size)

    print('Training set', train_dataset.shape, train_labels.shape)

    #print('Validation set', valid_dataset.shape, valid_labels.shape)
    return train_dataset, train_labels
  #valid_dataset, valid_labels


#def load_test_data(image_size, num_channels, label_cnt):
#    test_dataset = np.arange(32768000).reshape(1000, 32, 32, 32)
#    test_labels = np.concatenate((np.ones(500),2*np.ones(500)))
#    test_dataset, test_labels = reformat(test_dataset, test_labels, image_size, num_channels, label_cnt)
#    print('Test set', test_dataset.shape, test_labels.shape)
#    return test_dataset, test_labels



