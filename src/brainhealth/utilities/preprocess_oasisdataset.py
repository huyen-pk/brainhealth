import os
import shutil
import numpy as np
from brainhealth.utilities import visualizer

def classify_patients(OASIS_dataset_crosssec: str) -> tuple[list[str], list[str]]:
    demented = []
    nondemented = []
    for disc in os.listdir(OASIS_dataset_crosssec):
        disc_path = os.path.join(OASIS_dataset_crosssec, disc)
        for sub_dir in os.listdir(disc_path):
            sub_dir_path = os.path.join(disc_path, sub_dir)
            for patient_session in os.listdir(sub_dir_path):
                patient_path = os.path.join(sub_dir_path, patient_session)
                patient_info = os.path.join(patient_path, f'{patient_session}.txt')

                # Read patient info to classify into demented and nondemented
                with open(patient_info, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'SESSION ID' in line:
                            session_id = line.split()[-1]
                            if session_id.endswith('MR2'):
                                break
                        if 'CDR' in line:
                            CDR = line.split()
                            if len(CDR) > 1 and float(CDR[1]) > 0:
                                demented.append(patient_path)
                            else:
                                nondemented.append(patient_path)
                            break
    return demented, nondemented


def copy_files_to_classified_dir(paths: list[str], destination_dir: str):
    for path in paths:
        t88_path = os.path.join(path, 'PROCESSED', 'MPRAGE', 'T88_111')
        for file in os.listdir(t88_path):
            if (file.endswith('.hdr') or file.endswith('.img')) and \
               not os.path.exists(os.path.join(destination_dir, file)):
                shutil.copy(os.path.join(t88_path, file), destination_dir)

def extract_images(input_path: str, output_dir_path: str):
    _, _, files = next(os.walk(input_path))
    for file in files:
        if file.endswith('.img') and file.find('masked') == -1:
            print(os.path.join(input_path, file))
            visualizer.extract_images_from_3D_array_slices_ANTS(
                os.path.join(input_path, file),
                output_dir_path,
                'SAL') # extract axial slices
            visualizer.extract_images_from_3D_array_slices_ANTS(
                os.path.join(input_path, file),
                output_dir_path,
                'LPI') # extract sagittal slices
            visualizer.extract_images_from_3D_array_slices_ANTS(
                os.path.join(input_path, file),
                output_dir_path,
                'ASL') # extract coronal slices
            

import nibabel as nib
from nilearn import plotting
from nilearn.datasets import fetch_atlas_aal
import numpy as np
import matplotlib.pyplot as plt

def exploreData():
    img = nib.load('/home/huyenpk/Projects/AlzheimerDiagnosisAssist/Data/OAS1_0042_MR1/FSL_SEG/OAS1_0042_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.img')
    data = img.get_fdata()

    # Print some basic info about the image
    print("Image shape:", data.shape)
    print("Affine matrix:\n", img.affine)
    plt.imshow(data[:, 23, 15, 0], cmap='gray')
    plt.show()
    num_rows = 15
    num_cols = 15
    
    # Get the number of slices from the third element of the array
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through first dimension and display all slices (saggital)
    for i in range(min(data.shape[0], num_rows * num_cols)):
        axes[i].imshow(data[i, :, :], cmap='gray')
        axes[i].axis('off')  # Hide axis

    # Loop through second dimension and display all slices (coronal)
    for i in range(min(data.shape[1], num_rows * num_cols)):
        axes[i].imshow(data[:, i, :], cmap='gray')
        axes[i].axis('off')  # Hide axis

    # Loop through third dimension and display all slices (axial)
    for i in range(min(data.shape[2], num_rows * num_cols)):
        axes[i].imshow(data[:, :, i], cmap='gray')
        axes[i].axis('off')  # Hide axis

    # Turn off unused subplots
    for j in range(num_rows * num_cols, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def alignmentAAL(img):
    # Fetch and load the AAL atlas
    atlas = fetch_atlas_aal()
    atlas_img = nib.load(atlas['maps'])
    labels = atlas['labels']

    # Extract hippocampal regions
    hippocampus_mask = (atlas_img.get_fdata() == labels.index('Hippocampus_L')) | \
                       (atlas_img.get_fdata() == labels.index('Hippocampus_R'))

    hippocampus_img = nib.Nifti1Image(hippocampus_mask.astype(np.int32), img.affine)
    plotting.plot_roi(hippocampus_img, bg_img=img, title="Hippocampal Regions from AAL Atlas", display_mode='ortho')
    plotting.show()

    # Display the image using nilearn's plot_anat function
    plotting.plot_anat(img, title="Visualization of .img File")

    # Show the plot
    plt.show()