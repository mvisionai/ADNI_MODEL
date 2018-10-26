import numpy as np
import matplotlib.pyplot as plt
from dipy.data import fetch_tissue_data, read_tissue_data
import  nibabel as nib
from dipy.segment.tissue import TissueClassifierHMRF
import time

nclass = 3
beta = 0.1

def plane_image(t1):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    img_ax = np.rot90(t1[..., 89])
    imgplot = plt.imshow(img_ax, cmap="gray")
    a.axis('off')
    a.set_title('Axial')
    a = fig.add_subplot(1, 2, 2)
    img_cor = np.rot90(t1[:, 128, :])
    imgplot = plt.imshow(img_cor, cmap="gray")
    a.axis('off')
    a.set_title('Coronal')
    plt.savefig('t1_image.png', bbox_inches='tight', pad_inches=0)
    plt.show()

def segment_image(t_1):

    t0 = time.time()

    hmrf = TissueClassifierHMRF()
    initial_segmentation, final_segmentation, PVE = hmrf.classify(t_1, nclass, beta)

    t1 = time.time()
    total_time = t1 - t0
    print('Total time:' + str(total_time))


    img_ax = np.rot90(PVE[..., 89, 0])
    plt.imsave('csf.png', img_ax, cmap='gray')



    img_cor = np.rot90(PVE[:, :, 89, 1])
    plt.imsave('gray.png',img_cor, cmap='gray')


    img_cor = np.rot90(PVE[:, :, 89, 2])
    plt.imsave('white.png', img_cor, cmap='gray')

    plt.show()



t1_img = nib.load("AI.nii")
t1 = t1_img.get_data()
t1=t1[:,:,:,0]
print('t1.shape (%d, %d, %d)' % t1.shape)


segment_image(t1)
