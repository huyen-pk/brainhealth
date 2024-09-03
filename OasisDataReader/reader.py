import nibabel as nib
from nilearn import plotting
from nilearn.datasets import fetch_atlas_aal
import numpy as np
import matplotlib.pyplot as plt


# # Path to the .hdr file
# header_file_path = '/home/huyenpk/Projects/AlzheimerDiagnosisAssist/Data/OAS1_0042_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0042_MR1_mpr_n4_anon_sbj_111.hdr'

# # Load the header file
# header = nib.load(header_file_path)

# # Access header information
# header_data = header.header

# print(type(header))
# print(type(header_data))
# # Print the header information
# print(header_data)

# origin         : [8224 8224 8224 8224 8192]

# Load your image
img = nib.load('/home/huyenpk/Projects/AlzheimerDiagnosisAssist/Data/OAS1_0042_MR1/FSL_SEG/OAS1_0042_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.img')


def exploreData(img):
    data = img.get_fdata()

    # Print some basic info about the image
    print("Image shape:", data.shape)
    print("Affine matrix:\n", img.affine)
    plt.imshow(data[:, 23, 15, 0], cmap='gray')
    plt.show()
    # num_rows = 15
    # num_cols = 15
    
    # # Get the number of slices from the third element of the array
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # # Flatten the axes array for easy iteration
    # axes = axes.flatten()

    # Loop through first dimension and display all slices (saggital)
    # for i in range(min(data.shape[0], num_rows * num_cols)):
    #     axes[i].imshow(data[i, :, :], cmap='gray')
    #     axes[i].axis('off')  # Hide axis

    # # Loop through second dimension and display all slices (coronal)
    # for i in range(min(data.shape[1], num_rows * num_cols)):
    #     axes[i].imshow(data[:, i, :], cmap='gray')
    #     axes[i].axis('off')  # Hide axis

    # # Loop through third dimension and display all slices (axial)
    # for i in range(min(data.shape[2], num_rows * num_cols)):
    #     axes[i].imshow(data[:, :, i], cmap='gray')
    #     axes[i].axis('off')  # Hide axis

    # # Turn off unused subplots
    # for j in range(num_rows * num_cols, len(axes)):
    #     axes[j].axis('off')

    # plt.tight_layout()
    # plt.show()


def displayAllSlices(img):
    # Get the image data
    data = img.get_fdata()

    # Print some basic info about the image
    print("Image shape:", data.shape)
    print("Affine matrix:\n", img.affine)

    num_rows = 13
    num_cols = 13
    # Get the number of slices from the third element of the array
    num_slices= data.shape[2]

    # Check if the number of slices exceeds the grid size
    num_slices_to_display = min(num_slices, num_rows * num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through slices and plot them
    for i in range(num_slices_to_display):
        axes[i].imshow(data[:, :, i], cmap='gray')
        axes[i].axis('off')  # Hide axis

    # Turn off unused subplots
    for j in range(num_slices_to_display, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def displayOneSlice(img, slice_index):
    # Get the image data
    data = img.get_fdata()
    print("Image shape:", data.shape)
    print("Image affine:", img.affine)
    # Display a the slices of the image
    plt.imshow(data[:, :, slice_index, 0], cmap='gray')
    plt.title(f'Slice {slice_index}')
    plt.axis('off')
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

if __name__ == "__main__":
    exploreData(img)