import os
import shutil

from helpers import *
import SimpleITK as sitk
import xml.etree.ElementTree as ET
from torchvision import datasets, transforms
import torch


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
            extract_images_from_3D_array_slices_ANTS(
                os.path.join(input_path, file),
                output_dir_path,
                'SAL') # extract axial slices
            extract_images_from_3D_array_slices_ANTS(
                os.path.join(input_path, file),
                output_dir_path,
                'LPI') # extract sagittal slices
            extract_images_from_3D_array_slices_ANTS(
                os.path.join(input_path, file),
                output_dir_path,
                'ASL') # extract coronal slices

def get_sample_weights(dataset, train_dataset):
    
    # Code taken from:
    #     https://www.maskaravivek.com/post/pytorch-weighted-random-sampler/
    y_train_indices = train_dataset.indices
    y_train = [dataset.targets[i] for i in y_train_indices]
    
    class_sample_counts = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    
    weights = 1. / class_sample_counts
    sample_weights = np.array([weights[t] for t in y_train])
    sample_weights = torch.from_numpy(sample_weights)
    
    return sample_weights

def load_data(dataset_path: str) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    oasis_dataset = datasets.ImageFolder(
        dataset_path,
        transform=transform
    )

    proportions = [(1 - 0.15 - 0.15), 0.15, 0.15]
    lengths = [int(p * len(oasis_dataset)) for p in proportions]
    lengths[-1] = len(oasis_dataset) - sum(lengths[:-1])
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(oasis_dataset, lengths)

    sample_weights = get_sample_weights(oasis_dataset, train_dataset)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights.type('torch.DoubleTensor'), len(sample_weights))
    
    # Creating loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    return oasis_dataset



datasets_path = os.path.expanduser('~/Projects/AlzheimerDiagnosisAssist/Data/')
OASIS_dataset_crosssec = os.path.join(datasets_path, 'OASIS', 'oasis_cross-sectional')
oasis_demented_dir = os.path.join(datasets_path, 'OASIS', 'demented')
oasis_nondemented_dir = os.path.join(datasets_path, 'OASIS', 'nondemented')
os.makedirs(oasis_demented_dir, exist_ok=True)
os.makedirs(oasis_nondemented_dir, exist_ok=True)

# demented, nondemented = classify_patients(OASIS_dataset_crosssec)

# copy_files_to_classified_dir(demented, oasis_demented_dir)
# copy_files_to_classified_dir(nondemented, oasis_nondemented_dir)
# _, _, files = next(os.walk(oasis_demented_dir))
# visualize_3D_array_slices_ANTS(os.path.join(oasis_demented_dir, files[0]), orientation='SAL')

demented_slices = os.path.join(datasets_path, 'OASIS', 'slices', 'demented');
os.makedirs(demented_slices, exist_ok=True)
nondemented_slices = os.path.join(datasets_path, 'OASIS', 'slices', 'nondemented');
os.makedirs(nondemented_slices, exist_ok=True)

extract_images(oasis_demented_dir, demented_slices)
extract_images(oasis_nondemented_dir, nondemented_slices)
