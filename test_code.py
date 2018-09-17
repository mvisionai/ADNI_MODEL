import  nibabel as nib
import numpy as np

img = nib.load("AI.nii")

print(img.header)