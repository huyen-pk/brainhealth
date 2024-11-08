import pytest
from unittest.mock import patch, MagicMock
from botocore.exceptions import NoCredentialsError
import numpy as np
from infrastructure.storage import S3Storage

@pytest.fixture
def s3downloader():
    return S3Storage('test-bucket')

@patch('brainhealth.utilities.S3downloader.boto3.client')
def test_download_from_s3_success(mock_boto_client, s3downloader):
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator

    mock_page_iterator = MagicMock()
    mock_paginator.paginate.return_value = mock_page_iterator

    mock_page = {
        'Contents': [{'Key': 'test_image.jpg'}],
        'NextContinuationToken': 'token'
    }
    mock_page_iterator.build_full_result.return_value = mock_page

    with patch('brainhealth.utilities.S3downloader.tempfile.TemporaryDirectory') as mock_temp_dir, \
         patch('brainhealth.utilities.S3downloader.os.makedirs'), \
         patch('brainhealth.utilities.S3downloader.os.path.join', return_value='/tmp/test_image.jpg'), \
         patch('brainhealth.utilities.S3downloader.pp.image_dataset_from_directory') as mock_image_dataset:
        
        mock_temp_dir.return_value.__enter__.return_value = '/tmp'
        mock_image_dataset.return_value = [(np.random.rand(32, 32, 32, 3), np.random.randint(0, 2, 32))]

        images, labels = s3downloader.download_from_s3(page_size=32, page_index=0, page_count=1, folder_path='test')

        assert images.shape == (1, 32, 32, 32, 3)
        assert labels.shape == (1, 32)

@patch('brainhealth.utilities.S3downloader.boto3.client')
def test_download_from_s3_no_credentials_should_raise_exception(mock_boto_client, s3downloader):
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator

    mock_page_iterator = MagicMock()
    mock_paginator.paginate.return_value = mock_page_iterator

    mock_page_iterator.build_full_result.side_effect = NoCredentialsError

    with patch('brainhealth.utilities.S3downloader.tempfile.TemporaryDirectory') as mock_temp_dir, \
         patch('brainhealth.utilities.S3downloader.os.makedirs'), \
         patch('brainhealth.utilities.S3downloader.os.path.join', return_value='/tmp/test_image.jpg'), \
         patch('brainhealth.utilities.S3downloader.pp.image_dataset_from_directory') as mock_image_dataset:
        
        mock_temp_dir.return_value.__enter__.return_value = '/tmp'
        mock_image_dataset.return_value = [(np.random.rand(32, 32, 32, 3), np.random.randint(0, 2, 32))]

        with pytest.raises(NoCredentialsError):
            s3downloader.download_from_s3(page_size=32, page_index=0, page_count=1, folder_path='test')