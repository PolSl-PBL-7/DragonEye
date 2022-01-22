NAME = 'report_pipeline'


def report_pipeline(pipeline_params, video_report_params, prediction_params):
    from pipelines.prediction_pipeline import prediction_pipeline
    from report.report import ReportConfig, VideoReport
    from datetime import datetime

    dataset, predictions, scores = prediction_pipeline(**prediction_params)

    if pipeline_params['add_date_to_path'] == True:
        video_report_params['path'] += f'/{datetime.now().strftime(r"%m-%d-%Y-%H-%M-%S")}'
    video_report_config = ReportConfig(**video_report_params)
    video_report = VideoReport(video_report_config)
    video_report(dataset, predictions, scores)
