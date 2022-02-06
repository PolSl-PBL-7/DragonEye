import os
from pathlib import Path
import pipelines
from utils.file_loaders import Json
from utils.logging_utils import initialize_logger
from enum import Enum

from pipelines import data_processing_pipeline
from pipelines import training_pipeline
from pipelines import prediction_pipeline
from pipelines import report_pipeline
from pipelines import analysis_pipeline

import argparse
import logging

current_dir = Path(os.path.dirname(os.path.realpath(__file__)))


class PipelineType(Enum):

    data_processing_pipeline = data_processing_pipeline.NAME
    training_pipeline = training_pipeline.NAME
    prediction_pipeline = prediction_pipeline.NAME
    report_pipeline = report_pipeline.NAME
    analysis_pipeline = analysis_pipeline.NAME


def build_config_dict(path):

    path = Path(path)
    files = path.glob("*.json")

    config_dict = {}
    for file in files:
        config_dict[file.stem] = Json.load(file)

    return config_dict


def get_pipeline_by_type(pipeline_type: PipelineType):
    if pipeline_type == pipeline_type.data_processing_pipeline:
        return data_processing_pipeline.data_processing_pipeline
    if pipeline_type == pipeline_type.training_pipeline:
        return training_pipeline.training_pipeline
    if pipeline_type == pipeline_type.prediction_pipeline:
        return prediction_pipeline.prediction_pipeline
    if pipeline_type == pipeline_type.report_pipeline:
        return report_pipeline.report_pipeline
    if pipeline_type == pipeline_type.analysis_pipeline:
        return analysis_pipeline.analysis_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pipeline_type', type=PipelineType, choices=list(PipelineType))
    parser.add_argument('-c', '--config_folder', type=str, default=None)
    parser.add_argument('-j', '--config_json', type=str, default=None)

    args = parser.parse_args()

    if args.config_folder is not None:
        config_dict = build_config_dict(args.config_folder)
    elif args.config_json is not None:
        config_dict = Json.load(args.config_json)
    else:
        raise Exception("No config folder or config json provided")

    logging_dict = {"pipeline_runner_args": {'pipeline_type': args.pipeline_type, 'config_folder': args.config_folder}}
    logging_dict.update(config_dict)
    initialize_logger(current_dir / '..' / 'logging' / 'log.log', logging_dict)

    pipeline = get_pipeline_by_type(args.pipeline_type)
    logging.log(logging.INFO, "Running pipeline")
    print(config_dict)
    pipeline(**config_dict)
    logging.log(logging.INFO, "Pipeline finished running")
