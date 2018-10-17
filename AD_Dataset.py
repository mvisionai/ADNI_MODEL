from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import nibabel as nib
from dltk.io.preprocessing import whitening,normalise_zero_one
from  sklearn import  preprocessing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import  os
import matplotlib.pyplot as plt
import numpy as np
import AD_Constants as constants
import requests
from numpy import random
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img,array_to_img



class Dataset_Import(object):

    def __init__(self):
        self.main_directory=constants.main_image_directory
        self.yaml_stream=constants.yaml_values()
        self.training_part=constants.training_frac

        self.i = 0
        self.valid_source=0
        self.valid_target=0
        self.auto_shuffling_state=False
        self.trainer_shuffling_state = False
        self.train_ad_fnames = None
        self.train_mci_fnames = None
        self.train_nc_fnames = None
        self.source=constants.source
        self.target=constants.target
        self.img_shape_tuple=constants.img_shape_tuple
        self.img_channel=constants.img_channel
        self.shu_control_state=False

        self.image_group=constants.chosen_epi_format
        self.strict_match=constants.strict_match
        self.pic_index =constants.pic_index
        self.train_dir =constants.train_dir
        self.validation_dir =constants.validation_dir
        self.set_epoch=0
        self.set_checker=0

        # Directory with our training AD dataset
        self.train_ad_dir =constants.train_ad_dir

        # Directory with our training MCI dataset
        self.train_mci_dir = constants.train_mci_dir

        # Directory with our training NC dataset
        self.train_nc_dir =constants.train_nc_dir

        # Directory with our validation AD dataset
        self.validation_ad_dir = constants.validation_ad_dir

        # Directory with our validation MCI dataset
        self.validation_mci_dir =constants.validation_mci_dir

        # Directory with our validation NC dataset
        self.validation_nc_dir =constants.validation_nc_dir

        self.nrows =constants.nrows
        self.ncols = constants.ncols

    def statistics(self):

        #read_directory_file

        print('total training AD Data:', len(self.read_directory_file(self.train_ad_dir,"AD",source=self.source)), end="\n")
        print('total training MCI Data:', len(self.read_directory_file(self.train_mci_dir,"MCI",source=self.source)), end="\n")
        print('total training NC Data:', len(self.read_directory_file(self.train_nc_dir,"NC",source=self.source)), end="\n")
        print('total validation AD Data:', len(self.read_directory_file(self.validation_ad_dir,"AD",source=self.target)), end="\n")
        print('total validation MCI Data:', len(self.read_directory_file(self.validation_mci_dir,"MCI",source=self.target)), end="\n")
        print('total validation NC Data:', len(self.read_directory_file(self.validation_nc_dir,"NC",source=self.target)), end="\n")


    def  read_directory_file(self,original_dir,dx_group,source=None):

        group_data = []
        try:
         if original_dir is not None:


            for file in os.listdir(original_dir):

              if os.path.isdir(os.path.join(original_dir,file)):
                filepath = os.path.join(original_dir, file)

                for sub_file in os.listdir(filepath):


                  if os.path.isdir(os.path.join(filepath,sub_file)):

                      filepath2 = os.path.join(filepath, sub_file)



                      for  sub_file2 in os.listdir(filepath2):
                            decision_check=None

                            if  self.strict_match :
                                decision_check= sub_file2.strip() in self.image_group

                                #str(self.image_group) == str(sub_file.strip())
                            if  decision_check==True :

                                modal_groupings_source=os.path.join(filepath2 ,sub_file2)

                                files_in_time = len(os.listdir(modal_groupings_source))
                                if files_in_time > 1:
                                    time_file=max(os.listdir(modal_groupings_source))
                                else:
                                    time_file=os.listdir(modal_groupings_source)[0]


                                time_grouping_source=os.path.join(modal_groupings_source ,time_file)
                                files_in_time=len(os.listdir(modal_groupings_source))

                                for image_file in os.listdir(time_grouping_source):


                                  image_grouping_source = os.path.join(time_grouping_source, image_file)

                                  if os.path.isdir(os.path.join(image_grouping_source)):


                                    for image_file in os.listdir(image_grouping_source):
                                        image_grouping_source_file = os.path.join(image_grouping_source, image_file)
                                        label=self.get_nii_group(dx_group)
                                        source_label=self.get_nii_source(source)
                                        group_data.append([image_grouping_source_file,label,source_label])
                                        #print(image_grouping_source_file)


        except OSError as e:
            print('Error: %s' % e)

        return  group_data


    def convert_nii_to_image_data(self,nii_path):
        image_load = nib.load(nii_path, mmap=False)
        img_data = img_to_array(image_load.get_data()[:, :, :, 0])
        determine_shape = np.resize(img_data, self.img_shape_tuple)
        return normalise_zero_one(self.add_gaussian_noise(determine_shape))
        #determine_shape*(1./255)

    def one_hot_encode(self,vec, vals=3):
        '''
            For use to one-hot encode the 3- possible labels
            '''
        n = len(vec)
        out = np.zeros((n, vals))
        out[range(n), vec] = 1
        return out

    def get_nii_group(self,nii_path):
        img_label=nii_path

        if img_label=="AD":
            label=0
        elif img_label=="MCI":
            label=1
        elif img_label=="NC":
            label=2

        return  label

    def get_nii_source(self, source_target):
        source_label = source_target

        if source_label == "1.5T":
            source_label = 0
        elif source_label == "3.0T":
            source_label = 1

        return source_label

    def all_source_data(self):
        all_ad_train = self.read_directory_file(self.train_ad_dir, "AD",source=self.source)
        all_mci_train = self.read_directory_file(self.train_mci_dir, "MCI",source=self.source)
        all_nc_train = self.read_directory_file(self.train_nc_dir, "NC",source=self.source)
        all_source = [img_path for i, img_path in enumerate(all_ad_train + all_mci_train + all_nc_train)]
        all_source = np.array(all_source)

        if self.shu_control_state == False:
          self.set_random_seed(700)
        else:
          self.set_random_seed((self.i+1)*self.set_epoch*40)
        return  self.shuffle(all_source)

    def all_target_data(self):
        all_ad_target = self.read_directory_file(self.validation_ad_dir, "AD", source=self.target)
        all_mci_target = self.read_directory_file(self.validation_mci_dir, "MCI", source=self.target)
        all_nc_target = self.read_directory_file(self.validation_nc_dir, "NC", source=self.target)
        all_target = [img_path for i, img_path in enumerate(all_ad_target + all_mci_target + all_nc_target)]
        all_target = np.array(all_target)
        #all_target = np.take(all_target, np.random.permutation(len(all_target)), axis=0, out=all_target)

        if self.shu_control_state == False:
            self.set_random_seed(1000)
        else:
            self.set_random_seed((self.i + 1) * self.set_epoch *70)
        return self.shuffle(all_target)



    def shuffle(self,data):
        data=np.array(data)
        shuffled = np.take(data, np.random.permutation(len(data)), axis=0, out=data)
        return  shuffled

    def source_training_data(self):
        return self.all_source_data()[0:self.training_number(len(self.all_source_data()))]

    def target_training_data(self):
        return self.all_target_data()[0:self.training_number(len(self.all_target_data()))]

    def source_validation_data(self):
         return self.all_source_data()[self.training_number(len(self.all_source_data()))+1:len(self.all_source_data())-1]


    def target_validation_data(self):
        return self.all_target_data()[self.training_number(len(self.all_target_data())) +1:len(self.all_target_data()) - 1]

    def training_number(self,size):
        parts=int(round(size*self.training_part))
        return parts

    def save_training_data(self):

        trained_data=[]

        for img_path in self.all_source_data():
            data= self.convert_nii_to_image_data(img_path)
            label=self.get_nii_group(img_path)

            trained_data.append([data,label])
        np.save('np_data/trainData.npy', trained_data)
        print("Training Data Saved")


    def save_validation_data(self):

        validation_data = []

        for img_path in self.all_source_data():
            data = self.convert_nii_to_image_data(img_path)
            label = self.get_nii_group(img_path)

            validation_data.append([data, label])
        np.save('np_data/validateData.npy', validation_data)
        print("Validation Data Saved")


    def show_image(self):
        # Set up matplotlib fig, and size it to fit 4x4 pics
        self.statistics()
        fig = plt.gcf()
        fig.set_size_inches(self.ncols *8, self.nrows * 8)

        self.pic_index += 8
        #[self.pic_index - 8:self.pic_index]
        next_ad_pix = [fname
                        for fname in self.read_directory_file(self.validation_ad_dir,"AD")]
        next_mci_pix = [fname
                        for fname in self.read_directory_file(self.validation_mci_dir,"MCI")]
        next_nc_pix = [fname
                        for fname in self.read_directory_file(self.validation_nc_dir,"NC")]

        for i, img_path in enumerate(next_nc_pix+next_ad_pix  + next_mci_pix):
            # Set up subplot; subplot indices start at 1
            #sp = plt.subplot(self.nrows, self.ncols, i + 1)
            #sp.axis('Off')  # Don't show axes (or gridlines)


            image_load = nib.load(img_path[0],
                                  mmap=False)
            loads = img_to_array(image_load.get_data()[:, :, :, 0])
            data=np.resize(loads,(270,270,270))
            print(data.shape)
            break

            #print(data_normalise.shape)
            #plt.imshow(data_normalise[:, :, 210], cmap='gray')
            #img = mpimg.imread(img_path)
            #plt.imshow(img)

        #plt.show()

    def next_batch_source_path(self, batch_size):
        return self.source_training_data()[self.i:self.i + batch_size]


    def next_batch_target_path(self, batch_size):
        return self.target_training_data()[self.i:self.i + batch_size]


    def next_batch_source(self, batch_size):
        # Note that the  dimension in the reshape call is set by an assumed batch size set
        batch_data= self.source_training_data()[self.i:self.i + batch_size]
        batch_data=self.shuffle(batch_data)
        label=self.all_source_labels(batch_data)
        domain_label = self.encode_domain_labels(batch_data)
        #print("from ",self.i," to ",self.i + batch_size)
        self.i = (self.i + batch_size) % len(self.all_source_data())

        if len(self.img_shape_tuple) == 2:
            data_set = np.resize(self.convert_batch_to_img_data(batch_data),
                                 (batch_size, self.img_shape_tuple[0], self.img_shape_tuple[1], self.img_channel))
        elif len(self.img_shape_tuple) == 3:
            data_set = np.resize(self.convert_batch_to_img_data(batch_data), (
            batch_size, self.img_shape_tuple[0], self.img_shape_tuple[1], self.img_shape_tuple[2], self.img_channel))
        #print('Training set',data_set.shape, label.shape)
        return data_set,label,domain_label


    def batch_process_data(self, batch_size,feed_data):
        # Note that the  dimension in the reshape call is set by an assumed batch size set
        batch_data=feed_data
        label=self.all_source_labels(batch_data)
        domain_label = self.encode_domain_labels(batch_data)
        #print("from ",self.i," to ",self.i + batch_size)
        if len(self.img_shape_tuple) == 2:
            data_set = np.resize(self.convert_batch_to_img_data(batch_data),
                                 (batch_size, self.img_shape_tuple[0], self.img_shape_tuple[1], self.img_channel))
        elif len(self.img_shape_tuple) == 3:
            data_set = np.resize(self.convert_batch_to_img_data(batch_data), (
            batch_size, self.img_shape_tuple[0], self.img_shape_tuple[1], self.img_shape_tuple[2], self.img_channel))
        #print('Training set',data_set.shape, label.shape)
        return data_set,label,domain_label

    def source_target_combined(self):
        if self.auto_shuffling_state == False:
          self.set_random_seed(0)
        else:
          self.set_random_seed((self.i+1)*self.set_epoch*10)

        return  self.shuffle(np.vstack((self.all_target_data(),self.all_source_data())))

    def source_data_feed(self):
        if self.auto_shuffling_state == False:
            self.set_random_seed(0)
        else:
            self.set_random_seed((self.i + 1) * self.set_epoch * 10)
        return self.shuffle(self.all_source_data())

    def source_target_combined_2(self):
        if self.trainer_shuffling_state == False:
          self.set_random_seed(0)
        else:
            np.random.seed(None)
            self.set_random_seed((self.set_epoch+1)*100)

        return  self.shuffle(np.vstack(( self.target_training_data(),self.source_training_data())))

    def source_data(self):
        if self.trainer_shuffling_state == False:
          self.set_random_seed(0)
        else:
            np.random.seed(None)
            self.set_random_seed((self.set_epoch+1)*100)

        return  self.shuffle(self.source_training_data())

    def training_source_target(self):
        return self.source_target_combined()[0:self.training_number(len(self.all_source_data()))]


    def validate_source_target(self):
        return self.source_target_combined()[self.training_number(len(self.source_target_combined())) + 1:len(self.source_target_combined()) - 1]



    def next_batch_combined_encoder(self,batch_size,training_type="domain"):
        # Note that the  dimension in the reshape call is set by an assumed batch size set

        if training_type=="domain":
          batch_data =self.source_target_combined()[self.i:self.i + batch_size]
          self.i = (self.i + batch_size) % len(self.source_target_combined())
        elif training_type=="single":
            batch_data = self.source_data_feed()[self.i:self.i + batch_size]
            self.i = (self.i + batch_size) % len( self.source_data_feed())

        if len(self.img_shape_tuple) == 2:

           for c in range(batch_size):

             yield np.resize(self.convert_batch_to_img_data(batch_data[c]),
                                 (self.img_shape_tuple[0], self.img_shape_tuple[1], self.img_channel))
        elif len(self.img_shape_tuple) == 3:

           for c in range(batch_size):
             yield np.resize(self.convert_batch_to_img_data(batch_data[c]), (self.img_shape_tuple[0], self.img_shape_tuple[1], self.img_shape_tuple[2], self.img_channel))


    def next_batch_combined(self,batch_size):
        # Note that the  dimension in the reshape call is set by an assumed batch size set
        batch_data =self.source_target_combined_2()[self.i:self.i + batch_size]
        self.set_random_seed(None)
        batch_data = self.shuffle(batch_data)


        # print("from ",self.i," to ",self.i + batch_size)
        self.i = (self.i + batch_size) % len(self.source_target_combined_2())

        if len(self.img_shape_tuple) == 2:
            for c in range(batch_size):
                 yield np.resize(self.convert_batch_to_img_data(batch_data[c]),
                                 (self.img_shape_tuple[0], self.img_shape_tuple[1], self.img_channel)),self.all_source_labels(batch_data[c]),self.encode_domain_labels(batch_data[c])
        elif len(self.img_shape_tuple) == 3:
            for c in range(batch_size):
                 yield np.resize(self.convert_batch_to_img_data(batch_data[c]),
                                (self.img_shape_tuple[0], self.img_shape_tuple[1], self.img_shape_tuple[2], self.img_channel)),self.all_source_labels(batch_data[c]),self.encode_domain_labels(batch_data[c])



    def next_batch_target(self, batch_size):
        # Note that the  dimension in the reshape call is set by an assumed batch size set
        batch_data = self.target_training_data()[self.i:self.i + batch_size]
        batch_data = self.shuffle(batch_data)
        label = self.all_source_labels(batch_data)
        domain_label=self.encode_domain_labels(batch_data)

        # print("from ",self.i," to ",self.i + batch_size)
        self.i = (self.i + batch_size) % len(self.all_source_data())

        if len(self.img_shape_tuple) == 2:
         data_set = np.resize(self.convert_batch_to_img_data(batch_data), (batch_size,self.img_shape_tuple[0],self.img_shape_tuple[1],self.img_channel))
        elif len(self.img_shape_tuple) ==3:
         data_set = np.resize(self.convert_batch_to_img_data(batch_data), (batch_size,self.img_shape_tuple[0],self.img_shape_tuple[1],self.img_shape_tuple[2],self.img_channel))
        # print('Training set',data_set.shape, label.shape)
        return data_set, label,domain_label


    def all_source_labels(self, batch_data):
        data=batch_data
        col = 1
        encoded_label = self.one_hot_encode(np.hstack([int(data[col])]),3)
        return encoded_label[0]

    def encode_domain_labels(self,batch_data):
        data = batch_data
        col = 2
        encoded_label = self.one_hot_encode(np.hstack([int(data[col])]),2)
        return encoded_label[0]

    def all_encoded_labels(self, batch_data):
        data= batch_data
        col = 1
        encoded_label = self.one_hot_encode(np.hstack([ int((data[i][col])) for i in range(len(data))]),3)
        #self.one_hot_encode(np.hstack([(data[i][col]) for i in range(len(data))]), 3)
        return encoded_label


    def convert_batch_to_img_data(self, batch_data):
        data = batch_data
        col = 0
        datas =[self.convert_nii_to_image_data(data[col])]

        return datas

    def convert_validation_source_data(self,batch_size):

        batch_data =self.source_validation_data()[self.valid_source:self.valid_source + batch_size]
        data=self.shuffle(batch_data)

        self.valid_source = (self.valid_source + batch_size) % len(self.source_validation_data())

        if len(self.img_shape_tuple) == 2:
          for c in range(batch_size):
            yield np.resize(self.convert_batch_to_img_data(data[c]),(self.img_shape_tuple[0],self.img_shape_tuple[1],self.img_channel)), self.all_source_labels(
                data[c]), self.encode_domain_labels(data[c])

        elif len(self.img_shape_tuple) == 3:
          for c in range(batch_size):
             yield np.resize(self.convert_batch_to_img_data(data[c]), (self.img_shape_tuple[0],self.img_shape_tuple[1],self.img_shape_tuple[2],self.img_channel)), self.all_source_labels(
                data[c]), self.encode_domain_labels(data[c])

        #datas = [self.convert_nii_to_image_data(data[i][col]) for i in range(len(data))]


    def convert_validation_target_data(self,batch_size):

        batch_data = self.target_validation_data()[self.valid_target:self.valid_target + batch_size]
        data = self.shuffle(batch_data)
        #datas = [self.convert_nii_to_image_data(data[i][col]) for i in range(len(data))]

        self.valid_target = (self.valid_target + batch_size) % len(self.target_validation_data())

        if  len(self.img_shape_tuple)==2:
            for c in range(batch_size):
               yield np.resize(self.convert_batch_to_img_data(data[c]), (len(data),self.img_shape_tuple[0],self.img_shape_tuple[1],self.img_channel)), self.all_source_labels(data[c]),self.encode_domain_labels(data[c])

        elif len( self.img_shape_tuple)==3:
            for c in range(batch_size):
               yield  np.resize(self.convert_batch_to_img_data(data[c]),(self.img_shape_tuple[0],self.img_shape_tuple[1],self.img_shape_tuple[2],self.img_channel)), self.all_source_labels(
                  data[c]), self.encode_domain_labels(data[c])




    def convert_source_target_data(self):
        data = self.validate_source_target()
        data = self.shuffle(data)
        # datas = [self.convert_nii_to_image_data(data[i][col]) for i in range(len(data))]
        if len(self.img_shape_tuple) == 2:
            return np.resize(self.convert_batch_to_img_data(data), (
            len(data), self.img_shape_tuple[0], self.img_shape_tuple[1], self.img_channel)), self.all_source_labels(
                data), self.encode_domain_labels(data)

        elif len(self.img_shape_tuple) == 3:
            return np.resize(self.convert_batch_to_img_data(data), (
            len(data), self.img_shape_tuple[0], self.img_shape_tuple[1], self.img_shape_tuple[2],
            self.img_channel)), self.all_source_labels(
                data), self.encode_domain_labels(data)


    def load_dataset(self):
        test_image=nib.load(self.all_source_data()[0][0],mmap=False)
        test_mask=nib.load(self.all_target_data()[0][0],mmap=False)
        data=test_image.get_data()
        data2=test_mask.get_data()



        #print(data2.shape)
        first_vol = data[:, :, :, 0]
        #first_vol = np.resize(first_vol, (110,121000))
        print("Original shape ",first_vol.shape)
        first_vol2 = data2[:, :, :, 0]

        print("Original data ",first_vol)

        augment=self.flip(first_vol)
        print("Augmented shape ", augment.shape)
        print("Augmented data ", augment)

        #plt.imshow(first_vol[:, :,210], cmap='gray')
        #plt.imshow(first_vol2[:, :,110], cmap='gray')

        #plt.show()
        #plt.imsave("png/test_1.jpg",first_vol[:, :,210],cmap="gray")
        #plt.imsave("png/test_2.jpg",first_vol2[:, :,110],cmap="gray")

    def sample_yaml(self):
        pass

    def add_gaussian_noise(self,image, sigma=0.05):
        """
        Add Gaussian noise to an image
        Args:
            image (np.ndarray): image to add noise to
            sigma (float): stddev of the Gaussian distribution to generate noise
                from
        Returns:
            np.ndarray: same as image but with added offset to each channel
        """

        image += np.random.normal(0, sigma, image.shape)
        return image
    def set_random_seed(self,seed_v):
        np.random.seed(seed_v)

    def flip(self,imagelist, axis=1):
        """Randomly flip spatial dimensions
        Args:
            imagelist (np.ndarray or list or tuple): image(s) to be flipped
            axis (int): axis along which to flip the images
        Returns:
            np.ndarray or list or tuple: same as imagelist but randomly flipped
                along axis
        """

        # Check if a single image or a list of images has been passed
        was_singular = False
        if isinstance(imagelist, np.ndarray):
            imagelist = [imagelist]
            was_singular = True

        # With a probility of 0.5 flip the image(s) across `axis`
        do_flip = np.random.random(1)
        if do_flip > 0.5:
            for i in range(len(imagelist)):
                imagelist[i] = np.flip(imagelist[i], axis=axis)
        if was_singular:
            return imagelist[0]
        return imagelist

    def url_requests(self):
        r = requests.get('https://frightanic.com/goodies_content/docker-names.php')
        print(r.text.rstrip())

#if __name__=="__main__"    :

   #try:
        #dataset_feed=Dataset_Import()
       # print(dataset_feed.target_validation_data())
        #dataset_feed.load_dataset()
        #dataset_feed.sample_yaml()
        #dataset_feed.url_requests()
        #lens=int(len(dataset_feed.source_validation_data())/5)
        #lent=int(len(dataset_feed.target_validation_data())/5)
        #maxc=max(lens,lent)




        #validation_target_dataset, valid_target_label, valid_target_d_label = self.convert_validation_target_data()
        #dataset_feed.statistics()
        #dataset_feed.show_image()
        #dataset_feed.statistics()
        #dataset_feed.cnn_layer()
        #dataset_feed.save_training_data()
        #dataset_feed.all_training_data()
        #print(dataset_feed. convert_validation_source_data()[1])
        #print("te", dataset_feed.source_target_combined_2()[:, 0][:2])
        #print("tr",dataset_feed.source_target_combined()[:,0][:2])


        #print(dataset_feed.next_batch_source(5))
        # i=0
        # for data in dataset_feed.all_target_data():
        #     print(data[0])
        #     data=dataset_feed.convert_nii_to_image_data(data[0])
        #     i=i+1
        #     print(i)




  # except Exception as ex:
      # print(ex)



