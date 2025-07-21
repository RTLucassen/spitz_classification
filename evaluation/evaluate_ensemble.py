"""
Evaluate performance of model ensemble based on spreadsheets with predictions
from individual models.
"""

# from functools import partial
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

from evaluation.evaluation_service import (
    evaluate_with_thresholds,
    evaluate_highest_probability,
    evaluate_without_thresholds,
    bootstrapper,
)

# define paths and directories
models_directory = Path(r"C:\Users\rlucasse\OneDrive - UMC Utrecht\Documents\UMCU\projects\spitz_classification\models")
results_directory = models_directory / "001-005__hipt_4k__ensemble"
predictions_paths = [
    models_directory / Path(r"001__hipt_4k__val_fold-1\results_internal_test_spitz_diagnostic_classification\results_internal_test.xlsx"),
    models_directory / Path(r"002__hipt_4k__val_fold-2\results_internal_test_spitz_diagnostic_classification\results_internal_test.xlsx"),
    models_directory / Path(r"003__hipt_4k__val_fold-3\results_internal_test_spitz_diagnostic_classification\results_internal_test.xlsx"),
    models_directory / Path(r"004__hipt_4k__val_fold-4\results_internal_test_spitz_diagnostic_classification\results_internal_test.xlsx"),
    models_directory / Path(r"005__hipt_4k__val_fold-5\results_internal_test_spitz_diagnostic_classification\results_internal_test.xlsx"),
]
origin = 'internal'
subset = 'test'

# define other settings
thresholds = np.arange(0, 1.001, 0.005)
bootstrap_iterations = 10000
bootstrap_stratified = True
confidence_level = 0.95
bins = 10
seed = 1
image_extension = '.eps'

if __name__ == '__main__':

    # check if output folder already exists
    if not results_directory.exists():
        results_directory.mkdir()

    # combine the predictions from the multiple model instances
    combined_predictions = {}
    for predictions_path in predictions_paths:
        df = pd.read_excel(predictions_path)
        for i, row in df.iterrows():
            if row['specimen'] in combined_predictions:
                combined_predictions[row['specimen']]['y_true'].append(eval(row['y_true']))
                combined_predictions[row['specimen']]['y_pred'].append(eval(row['y_pred']))
            else:
                combined_predictions[row['specimen']] = {
                    'y_true': [eval(row['y_true'])], 
                    'y_pred': [eval(row['y_pred'])],
                }

    predictions = df.to_dict(orient='list')  
    for key in ['y_true', 'y_pred', 'y_pred_stdev']:
        predictions[key] = [] 

    # get the average and standard deviation of the predicted probability
    for specimen in predictions['specimen']:
        predictions['y_true'].append([mean(values) for values in list(zip(*combined_predictions[specimen]['y_true']))])  
        predictions['y_pred'].append([mean(values) for values in list(zip(*combined_predictions[specimen]['y_pred']))])  
        predictions['y_pred_stdev'].append([stdev(values) for values in list(zip(*combined_predictions[specimen]['y_pred']))])  

    # perform evaluation for binary classification task
    N_classes = len(predictions['y_true'][0])
    if N_classes == 2:
        # perform threshold-dependent evaluation
        threshold_results = evaluate_with_thresholds(
            y_true=[pred[1] for pred in predictions['y_true']], 
            y_pred=[pred[1] for pred in predictions['y_pred']], 
            thresholds=thresholds,
        )
        # perform threshold-independent evaluation 
        area_results = evaluate_without_thresholds(
            y_true=[pred[1] for pred in predictions['y_true']], 
            y_pred=[pred[1] for pred in predictions['y_pred']],
            bins=bins,                                                    
            roc_figure_path=results_directory / f'ROC_curve_{origin}_{subset}.{image_extension.replace(".", "")}',
            pr_figure_path=results_directory / f'PR_curve_{origin}_{subset}.{image_extension.replace(".", "")}',
            cal_figure_path=results_directory / f'Calibration_{origin}_{subset}.{image_extension.replace(".", "")}',
        )
        # perform bootstrapping for confidence intervals
        if bootstrap_iterations is not None:
            bootstrap_output = {'accuracy': {'threshold':[]}, 'AUROC': {}}
            for threshold in thresholds:
                # define accuracy wrapper
                def accuracy_wrapper(y_true, y_pred):
                    return evaluate_with_thresholds(y_true, y_pred, threshold)['accuracy'][0]
                # perform bootstrapping for accuracy at specific threshold value
                bootstrap_accuracy_output = bootstrapper(
                    y_true=[pred[1] for pred in predictions['y_true']], 
                    y_pred=[pred[1] for pred in predictions['y_pred']], 
                    eval_func=accuracy_wrapper, 
                    iterations=bootstrap_iterations, 
                    stratified=bootstrap_stratified,
                    confidence_level=confidence_level, 
                    seed=seed,
                    show_progress=False,
                )
                bootstrap_output['accuracy']['threshold'].append(threshold)
                for key, value in bootstrap_accuracy_output.items():
                    if key not in bootstrap_output['accuracy']:
                        bootstrap_output['accuracy'][key] = [value]
                    else:
                        bootstrap_output['accuracy'][key].append(value)

            # define AUROC wrapper
            def AUROC_wrapper(y_true, y_pred):
                return evaluate_without_thresholds(y_true, y_pred, bins)['AUROC'][0]
            # perform bootstrapping for AUROC
            bootstrap_AUROC_output = bootstrapper(
                y_true=[pred[1] for pred in predictions['y_true']], 
                y_pred=[pred[1] for pred in predictions['y_pred']], 
                eval_func=AUROC_wrapper, 
                iterations=bootstrap_iterations, 
                stratified=bootstrap_iterations,
                confidence_level=confidence_level,
                seed=seed,
                show_progress=False,
            )
            for key, value in bootstrap_AUROC_output.items():
                if key not in bootstrap_output['AUROC']:
                    bootstrap_output['AUROC'][key] = [value]
                else:
                    bootstrap_output['AUROC'][key].append(value)

    # perform evaluation for multi-class classification task
    elif N_classes > 2:
        # perform evaluation based the highest probability class
        threshold_results = evaluate_highest_probability(
            y_true=predictions['y_true'], 
            y_pred=predictions['y_pred'],
        )
        # perform threshold-independent evaluation per class
        area_results = {}
        for class_index in range(N_classes):
            cl = f'class_{class_index}'
            area_results_per_class = evaluate_without_thresholds(
                y_true=[pred[class_index] for pred in predictions['y_true']], 
                y_pred=[pred[class_index] for pred in predictions['y_pred']],
                bins=bins,                                                    
                roc_figure_path=results_directory / f'ROC_curve_{cl}_{origin}_{subset}.{image_extension.replace(".", "")}',
                pr_figure_path=results_directory / f'PR_curve_{cl}_{origin}_{subset}.{image_extension.replace(".", "")}',
                cal_figure_path=results_directory / f'Calibration_{cl}_{origin}_{subset}.{image_extension.replace(".", "")}',
            )
            for key, value in area_results_per_class.items():
                area_results[f'{cl}_{key}'] = value

        # perform bootstrapping for confidence intervals
        if bootstrap_iterations is not None:
            bootstrap_output = {'accuracy': {}, 'AUROC': {}}
            # define accuracy wrapper
            def accuracy_wrapper(y_true, y_pred):
                return evaluate_highest_probability(y_true, y_pred)['accuracy'][0]
            # perform bootstrapping for accuracy
            bootstrap_accuracy_output = bootstrapper(
                y_true=predictions['y_true'], 
                y_pred=predictions['y_pred'], 
                eval_func=accuracy_wrapper,
                iterations=bootstrap_iterations, 
                stratified=bootstrap_stratified,
                confidence_level=confidence_level, 
                seed=seed,
                show_progress=False,
            )
            for key, value in bootstrap_accuracy_output.items():
                if key not in bootstrap_output['accuracy']:
                    bootstrap_output['accuracy'][key] = [value]
                else:
                    bootstrap_output['accuracy'][key].append(value)

            for class_index in range(N_classes):
                cl = f'class_{class_index}'
                # define AUROC wrapper
                def AUROC_wrapper(y_true, y_pred):
                    return evaluate_without_thresholds(y_true, y_pred, bins)['AUROC'][0]
                # perform bootstrapping for AUROC
                bootstrap_AUROC_output = bootstrapper(
                    y_true=[pred[class_index] for pred in predictions['y_true']], 
                    y_pred=[pred[class_index] for pred in predictions['y_pred']], 
                    eval_func=AUROC_wrapper, 
                    iterations=bootstrap_iterations, 
                    stratified=bootstrap_stratified,
                    confidence_level=confidence_level,
                    seed=seed,
                    show_progress=False,
                )
                for key, value in bootstrap_AUROC_output.items():
                    if key not in bootstrap_output['AUROC']:
                        bootstrap_output['AUROC'][f'{cl}_{key}'] = [value]
                    else:
                        bootstrap_output['AUROC'][f'{cl}_{key}'].append(value)

    # convert dictionaries to dataframes
    predictions_df = pd.DataFrame.from_dict(predictions)
    threshold_results_df = pd.DataFrame.from_dict(threshold_results)
    area_results_df = pd.DataFrame.from_dict(area_results)
    if bootstrap_iterations is not None:
        bootstrap_accuracy_df =  pd.DataFrame.from_dict(bootstrap_output['accuracy'])
        bootstrap_AUROC_df = pd.DataFrame.from_dict(bootstrap_output['AUROC'])

    # save evaluation results in spreadsheet
    with pd.ExcelWriter(results_directory / f'results_{origin}_{subset}.xlsx') as writer:
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        threshold_results_df.to_excel(writer, sheet_name='Results (threshold)', index=False)
        area_results_df.to_excel(writer, sheet_name='Results (area)', index=False)
        if bootstrap_iterations is not None:
            bootstrap_accuracy_df.to_excel(writer, sheet_name='Bootstrap accuracy results', index=False)
            bootstrap_AUROC_df.to_excel(writer, sheet_name='Bootstrap AUROC results', index=False)