from os import path
from pathlib import Path
from data.versioning.versioner import WandbDatasetVersioner, VersioningConfig

CURDIR = Path(__file__).parents[1]
dataset_path = CURDIR / "test_videos"


def test_saving_file():
    video_path = dataset_path / '15.avi'
    db = WandbDatasetVersioner()
    config = VersioningConfig(type='file', dataset_path=video_path, dataset_name='test_dataset')
    db.save_dataset(config)
    video_path_test = dataset_path / '15_test.avi'
    db = WandbDatasetVersioner()
    config = VersioningConfig(type='file', dataset_path=video_path_test, dataset_name='test_dataset')
    db.load_dataset(config)
    assert Path(video_path_test).exists()


def test_loading_folder():
    video_path = dataset_path
    db = WandbDatasetVersioner()
    config = VersioningConfig(type='folder', dataset_path=video_path, dataset_name='test_dataset')
    db.save_dataset(config)
    video_path_test = dataset_path / 'test'
    db = WandbDatasetVersioner()
    config = VersioningConfig(type='folder', dataset_path=video_path_test, dataset_name='test_dataset')
    db.load_dataset(config)
    assert Path(video_path_test).exists()
