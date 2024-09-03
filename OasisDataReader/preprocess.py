from helpers import *
import SimpleITK as sitk

path = '/home/huyenpk/Projects/AlzheimerDiagnosisAssist/Data/OAS1_0042_MR1/FSL_SEG/OAS1_0042_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.img';
visualize_3D_array_slices_ANTS(path, orientation='LPS')
