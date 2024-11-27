import pytest
import os
import numpy as np
from keras import preprocessing
from infrastructure.units_of_work import Local_ModelTrainingDataDomain
from unittest.mock import patch, MagicMock

class TestLocalModelTrainingDataDomain:

    @pytest.fixture(scope="function")
    def setup(self):
        model_repository = MagicMock()
        checkpoint_repository = MagicMock()
        dataset_repository = MagicMock()
        return Local_ModelTrainingDataDomain(model_repository, checkpoint_repository, dataset_repository)

    @patch('infrastructure.units_of_work.os.getenv')
    @patch('infrastructure.units_of_work.os.path.exists')
    @patch('infrastructure.units_of_work.os.listdir')
    @patch('infrastructure.units_of_work.pp.image_dataset_from_directory')
    def test_get_dataset_should_return_correct_shape(self, mock_image_dataset_from_directory, mock_listdir, mock_exists, mock_getenv, setup):
        mock_getenv.return_value = '/fake/dataset/repo'
        mock_exists.return_value = True
        mock_listdir.return_value = ['file1', 'file2']
        
        fake_dataset = MagicMock()
        fake_dataset.__iter__.return_value = iter([(np.random.rand(32, 32, 32, 3), np.random.randint(0, 2, 32))])
        mock_image_dataset_from_directory.return_value = fake_dataset

        images, labels = setup.get_dataset(page_size=32, page_index=0)

        assert images.shape == (1, 32, 32, 32, 3)
        assert labels.shape == (1, 32)

    @patch('infrastructure.units_of_work.os.getenv')
    @patch('infrastructure.units_of_work.os.path.exists')
    @patch('infrastructure.units_of_work.os.listdir')
    def test_get_dataset_should_raise_exception_if_repo_not_exists(self, mock_listdir, mock_exists, mock_getenv, setup):
        mock_getenv.return_value = '/fake/dataset/repo'
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            setup.get_dataset(page_size=32, page_index=0)

    @patch('infrastructure.units_of_work.os.getenv')
    @patch('infrastructure.units_of_work.os.path.exists')
    @patch('infrastructure.units_of_work.os.listdir')
    def test_get_dataset_should_raise_exception_if_repo_is_empty(self, mock_listdir, mock_exists, mock_getenv, setup):
        mock_getenv.return_value = '/fake/dataset/repo'
        mock_exists.return_value = True
        mock_listdir.return_value = []

        with pytest.raises(FileNotFoundError):
            setup.get_dataset(page_size=32, page_index=0)