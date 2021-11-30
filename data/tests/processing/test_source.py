from pathlib import Path
import numpy as np
import skvideo.io as sk
from data.processing.source import LocalVideoSource, SourceConfig
from utils.range_types import ClosedRange

CURDIR = Path(__file__).parents[1]
dataset_path = CURDIR / "test_videos"


def test_instance():
    config = SourceConfig()
    source = LocalVideoSource(config=config)
    video = source(path=dataset_path / '15.avi')
    assert isinstance(video, np.ndarray)


def test_non_null():
    config = SourceConfig()
    source = LocalVideoSource(config=config)
    video = source(path=dataset_path / '15.avi')
    assert video.shape[0] > 1


def test_fps():
    path = dataset_path / '15.avi'

    fps = 5

    config_with_fps = SourceConfig(fps=fps)
    source = LocalVideoSource(config=config_with_fps)
    video_with_altered_fps = source(path=path)

    config_without_fps = SourceConfig()
    source = LocalVideoSource(config=config_without_fps)
    video_without_altered_fps = source(path=path)

    video_fps = sk.ffprobe(str(path))['video']['@avg_frame_rate']
    frames, seconds = [int(x) for x in video_fps.split("/")]
    video_fps = frames / seconds

    fps_ratio = video_without_altered_fps.shape[0] / video_with_altered_fps.shape[0]
    assert fps_ratio in ClosedRange(fps - 1, fps + 1)
