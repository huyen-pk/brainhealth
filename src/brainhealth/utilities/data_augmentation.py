import numpy as np
from monai.transforms import Compose, Rand3DElastic, ToTensor

def elastic_transform(image):
    # Define the augmentation pipeline
    transforms = Compose([
        # Apply a random elastic deformation.
        # Adjust parameters such as sigma_range, magnitude_range, and spatial_size as required.
        Rand3DElastic(
            prob=0.3,  # Always apply for demonstration; set to lower value in practice.
            sigma_range=(5, 7),         # Smoothing kernel for the deformation.
            magnitude_range=(0, 1),     # The magnitude (or intensity) of the deformation.
            # spatial_size=image.shape,  # Ensure the transform knows the input spatial dimensions.
        ),
        # Convert the result to a PyTorch tensor.
        ToTensor(),
    ])
    transposed = np.transpose(image, (2, 0, 1))
    augmented_image_tensor = transforms(transposed).numpy()
    transposed_back = np.transpose(augmented_image_tensor, (1, 2, 0))
    return  transposed_back