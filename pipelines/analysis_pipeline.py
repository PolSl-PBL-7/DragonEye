import gc

from pipelines.prediction_pipeline import prediction_pipeline

NAME = 'analysis_pipeline'

def get_statistics(all_scores):
    median = np.median(all_scores_test)
    first = np.quantile(all_scores_test, 0.25)
    third = np.quantile(all_scores_test, 0.75)
    iqr = third - first
    threshold = third + 1.5*iqr
    return {
            "first_quantile":first,
            "thisr_quantile": third,
            "iqr": iqr,
            "treshold": threshold,
            "median": median
            }


def analysis_pipeline(training_dataset_prediction_params:dict,testing_dataset_prediction_params:dict):
    print(training_dataset_prediction_params)
    print(testing_dataset_prediction_params)    
    train_dataset, train_predictions, train_scores = prediction_pipeline(**training_dataset_prediction_params)
    train_statistics = get_statistics(train_scores)
    del train_dataset, train_predictions, train_scores
    gc.collect()
    test_dataset, test_predictions, test_scores = prediction_pipeline(**testing_dataset_prediction_params)
    test_statistics = get_statistics(test_scores)
    del test_dataset, test_predictions, test_scores
    gc.collect()
    