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