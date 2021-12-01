import filecmp
from os import path
from pathlib import Path
from data.versioning.versioner import WandbDatasetVersioner, VersioningConfig

CURDIR = Path(__file__).parents[2]
dataset_path_read = CURDIR / "tests/test_videos"
dataset_path_write = CURDIR / "tests/test_versioner_output"


def test_loading_file_exists():
    db = WandbDatasetVersioner()
    config = VersioningConfig(type='file', dataset_path=dataset_path_read / '16.avi', dataset_name='test_dataset')
    db.save_dataset(config)
    db = WandbDatasetVersioner()
    config = VersioningConfig(type='file', dataset_path=dataset_path_write, dataset_name='test_dataset')
    db.load_dataset(config)
    assert Path(dataset_path_write / '16.avi').exists()


def test_loading_file_content_comparison():
    db = WandbDatasetVersioner()
    config = VersioningConfig(type='file', dataset_path=dataset_path_read / '15.avi', dataset_name='test_dataset')
    db.save_dataset(config)
    db = WandbDatasetVersioner()
    config = VersioningConfig(type='file', dataset_path=dataset_path_write, dataset_name='test_dataset')
    db.load_dataset(config)
    assert filecmp.cmp(dataset_path_read / '15.avi', dataset_path_write / '15.avi')


def test_loading_folder_exists():
    db = WandbDatasetVersioner()
    config = VersioningConfig(type='folder', dataset_path=dataset_path_read, dataset_name='test_dataset')
    db.save_dataset(config)
    video_path_test = dataset_path_write / 'test'
    db = WandbDatasetVersioner()
    config = VersioningConfig(type='folder', dataset_path=video_path_test, dataset_name='test_dataset')
    db.load_dataset(config)
    assert Path(video_path_test).exists()


def test_loading_folder_content_comparison():
    db = WandbDatasetVersioner()
    config = VersioningConfig(type='folder', dataset_path=dataset_path_read, dataset_name='test_dataset')
    db.save_dataset(config)
    video_path_test = dataset_path_write / 'test'
    db = WandbDatasetVersioner()
    config = VersioningConfig(type='folder', dataset_path=video_path_test, dataset_name='test_dataset')
    db.load_dataset(config)
    assert len(filecmp.dircmp(dataset_path_read, dataset_path_write / 'test/').diff_files) == 0
