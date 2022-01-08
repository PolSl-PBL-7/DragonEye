from typing import NamedTuple, Tuple, Sequence, Any, Optional, Iterable, Union
from copy import deepcopy
from pathlib import Path
import itertools
from tensorflow.python.data.ops.dataset_ops import ConcatenateDataset
from tensorflow.python.data.ops.dataset_ops import Dataset

from data.processing.source import Source, LocalVideoSource, LocalTFDataSource, SourceConfig
from data.processing.process import VideoProcessor, ProcessorConfig
from data.processing.sinks import Sink, LocalTFDatasetSink, SinkConfig

import glob


class DataProcessingConfig(NamedTuple):
    source: Source
    source_config: SourceConfig
    processor: Optional[VideoProcessor] = None
    processor_config: Optional[ProcessorConfig] = None
    sink: Optional[Sink] = None
    sink_config: Optional[SinkConfig] = None
    input: Optional[Union[Sequence, Path]] = None
    video_extentions: Iterable[str] = ['mp4', 'avi', 'mov']


class DataProcessing:
    """
    Object that is responsible for running data processing logic,
    based on configuration.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, config: DataProcessingConfig):
        """ Method for loading files from specified folder and processing them into dingle ConcatenateDataset

        Args:
            source (Source): data loader of class source
            source_config (SourceConfig): config tuple for source
            processor (VideoProcessor, optional): processor of class VideoProcessor, performs operations on numpy arrays and return tf dataset
            processor_config (ProcessorConfig, optional): configuration tuple for processor
            sink (Sink, optional): class
            sink_config (SinkConfig, optional): configuration tuple for class responsible for saving
            input (Union[Sequence, Path], optional): path to the input folder
            video_extentions (Iterable[str]): all compatible video extentions

        Returns:
            (ConcatenateDataset): Concatenated dataset of all selected videos
        """
        # Data loading from different sources
        if isinstance(config.source, LocalVideoSource):
            if isinstance(config.input, Path) or isinstance(config.input, str):
                input_path = Path(config.input)
                video_paths = [
                    input_path.glob(f"*.{extention}") for extention in config.video_extentions
                ]
                video_paths = list(itertools.chain.from_iterable(video_paths))
            else:
                video_paths = config.input

        elif isinstance(config.source, LocalTFDataSource) and isinstance(config.input, Path):
            return config.source(
                config.input,
            )

        else:
            return

        # load, preprocess, batch and save to tf dataset
        data = []
        for path in video_paths:
            vid = config.source(path)

            vid = config.processor(vid) if config.processor else Dataset(vid)
            
            data.append(vid)

        # Concatenating single videos to one tf dataset, and saving it with choosen Sink
        if len(data) > 0:
            dataset = data[0]
            for x in data[1:]:
                dataset = dataset.concatenate(x)
            config.sink(
                dataset=dataset) if config.sink and config.sink_config else None
            return dataset

        return
