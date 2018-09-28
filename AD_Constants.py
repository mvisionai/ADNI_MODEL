import yaml
import  os
main_image_directory = "M:\LabWork\ADNI_STATISTICS\ADNI"
chosen_epi_format=["MP-RAGE","MPRAGE"]
strict_match=True
yaml_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),"init_yaml.yml")

pre_trained=["conv1layer","conv2layer","conv3layer","conv4layer","conv5layer"]
not_trained=["fc1layer","fc2layer","fc3layer","domain_predictor"]

# Parameters for our graph; we'll output images in a 4x4 configuration
# os.kill(os.getpid(), signal.pthread_kill())
nrows = 24
ncols = 4



#(20,20,20)
img_shape_tuple=(96,96,96)
train_ad_fnames = None
img_channel=1
train_mci_fnames = None
train_nc_fnames = None
source="1.5T"
target="3.0T"
training_frac=80/100

# Index for iterating over images
pic_index = 0
train_dir = os.path.join(main_image_directory, 'train')
validation_dir = os.path.join(main_image_directory, 'validation')

# Directory with our training AD dataset
train_ad_dir = os.path.join(train_dir, 'AD')

# Directory with our training MCI dataset
train_mci_dir = os.path.join(train_dir, 'MCI')

# Directory with our training NC dataset
train_nc_dir = os.path.join(train_dir, 'NC')

# Directory with our validation AD dataset
validation_ad_dir = os.path.join(validation_dir, 'AD')

# Directory with our validation MCI dataset
validation_mci_dir = os.path.join(validation_dir, 'MCI')

# Directory with our validation NC dataset
validation_nc_dir = os.path.join(validation_dir, 'NC')


def yaml_values():

    with open(yaml_file, 'r') as stream:
        try:
            init_args = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        finally:
            stream.close()

    return init_args