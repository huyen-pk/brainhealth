import matplotlib.pyplot as plt
import os
from ipywidgets import interact
import numpy as np
import math
import ants
import SimpleITK as sitk
from PIL import Image
import cv2

def visualize_3D_array_slices_ANTS(path: str, orientation: str = 'IAL'):
  """
  Given a 3D array with shape (Z,Y,X) This function will plot all
  the 2D arrays with shape (Y,X) inside the 3D array, in the Z direction. 
  The purpose of this function to visual inspect the 2D arrays in the image. 

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  img_raw = ants.image_read(path)
  img_ants = ants.reorient_image2(img_raw, orientation)
  SLICE=(0, img_ants.shape[0]-1) # z-axis of the array represents the number of slices (depth of the volume)
  print(f'number of slices {SLICE}')

  print(f'original orientation of image = {ants.get_orientation(img_raw)}, shape {img_ants.shape}')
  print(f'''image reoriented to {orientation} = 
        {'true' if ants.get_orientation(img_ants) == orientation else '' }, 
         shape {img_ants.shape}''')
  
  num_rows = math.ceil(np.sqrt(SLICE[1]))
  num_cols = num_rows
  num_slices_to_display = min(SLICE[1], num_rows * num_cols)
  _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
  print(f'num_rows {num_rows}, num_cols {num_cols}, num_slices_to_display {num_slices_to_display}')
  # Flatten the axes array for easy iteration
  axes = axes.flatten()
  arr = img_ants.numpy()
  # Loop through slices and plot them
  for i in range(num_slices_to_display):
      axes[i].imshow(arr[i, :, :], cmap='gray')
      axes[i].axis('off')  # Hide axis
  # Turn off unused subplots
  for j in range(num_slices_to_display, len(axes)):
      axes[j].axis('off')
  plt.tight_layout()
  plt.show()

def extract_images_from_3D_array_slices_ANTS(input_path: str, output_dir: str, orientation: str = 'IAL'):
  """
  Given a 3D array with shape (Z,Y,X) This function will extract all
  the 2D arrays with shape (Y,X) inside the 3D array, in the Z direction. 

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  img_raw = ants.image_read(input_path)
  img_ants = ants.reorient_image2(img_raw, orientation)
  SLICE=(0, img_ants.shape[0]-1) # z-axis of the array represents the number of slices (depth of the volume)
  arr = img_ants.numpy()
  # Normalize the image data to the range [0, 255]
  arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
  # Loop through slices and write them to disk
  for i in range(SLICE[1]):
    pil_img = Image.fromarray(arr[i, :, :])
    output_path = f'{os.path.join(output_dir, os.path.basename(input_path))}_{orientation}_{i}.jpg'
    pil_img.save(output_path)

def show_sitk_img_info(img: sitk.Image):
  """
  Given a sitk.Image instance prints the information about the MRI image contained.

  Args:
    img : instance of the sitk.Image to check out
  """
  pixel_type = img.GetPixelIDTypeAsString()
  origin = img.GetOrigin()
  dimensions = img.GetSize()
  spacing = img.GetSpacing()
  direction = img.GetDirection()

  info = {'Pixel Type' : pixel_type, 'Dimensions': dimensions, 'Spacing': spacing, 'Origin': origin,  'Direction' : direction}
  for k,v in info.items():
    print(f' {k} : {v}')


def rescale_linear(array: np.ndarray, new_min: int, new_max: int):
  """Rescale an array linearly."""
  minimum, maximum = np.min(array), np.max(array)
  m = (new_max - new_min) / (maximum - minimum)
  b = new_min - m * minimum
  return m * array + b

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