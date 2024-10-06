import numpy as np
import cv2
import os
from scipy.fftpack import fft2, ifft2

def detect_outliers_fourier_transform(image_folder, threshold=1.5):
    outliers = []
    magnitude_spectrum = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            # Apply Fourier Transform
            f_transform = fft2(image)
            f_transform_shifted = np.fft.fftshift(f_transform)

            # Calculate magnitude spectrum
            magnitude_value = np.log(np.abs(f_transform_shifted) + 1)
            magnitude_spectrum[filename] = magnitude_value

    mean = np.mean(list(magnitude_spectrum.values()))
    std_dev = np.std(list(magnitude_spectrum.values()))

    print("mean: ", mean, "std_dev", std_dev)
    # for filename, magnitude in magnitude_spectrum.items():
    #     # Detect outliers
    #     if magnitude > mean + threshold * std_dev:
    #         outliers.append(filename)

    return outliers

if __name__ == "__main__":
    image_folder = os.path.expanduser('~/Projects/AlzheimerDiagnosisAssist/tests/data/image_outliers')
    outliers = detect_outliers_fourier_transform(image_folder)
    print(f"Detected outliers {len(outliers)}:", outliers)