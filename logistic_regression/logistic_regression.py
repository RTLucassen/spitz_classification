import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluation.evaluation_service import (
    evaluate_with_thresholds, 
    evaluate_without_thresholds, 
    bootstrapper,
)

bootstrap_iterations = 10000
bootstrap_confidence_level = 0.95
seed = 12345
evaluate_on_test = False

variant = 'internal'

training = ['fold-5', 'fold-1', 'fold-2', 'fold-3']
validation = ['fold-4']
test = ['test']

path = r"data.xlsx"

if __name__ == '__main__':

    df = pd.read_excel(path)

    if variant == 'internal':
        df = df[df['internal_paths'] != '[]']
    elif variant == 'consultation':
        df = df[df['consultation_paths'] != '[]']
    elif variant == 'combined':
        pass
    else:
        raise ValueError(f'Invalid variant name: {variant}')
    
    data_dict = {}
    for name, subset in [('training', training), ('validation', validation), ('test', test)]:
        df_subset = df[df['set'].isin(subset)]

        specimen = []
        X = []
        y = []
        for i, row in df_subset.iterrows():
            feature = []
            feature.append(row['age']/100)
            feature.append(1 if row['sex'] == 'M' else 0)
            feature.append(1 if (row['location_group'] == 'HEAD') or (row['location_group'] == 'NECK') else 0)
            feature.append(1 if (row['location_group'] == 'HAND') or (row['location_group'] == 'HANDPALM') or (row['location_group'] == 'FOOT') or (row['location_group'] == 'FOOTSOLE') else 0)
            feature.append(1 if (row['location_group'] == 'TRUNK') or (row['location_group'] == 'BUTTOCK') else 0)
            feature.append(1 if row['location_group'] == 'UPPER EXTREMITY' else 0)
            feature.append(1 if row['location_group'] == 'LOWER EXTREMITY' else 0)
            specimen.append(row['specimen'])
            X.append(feature)
            y.append(row['label_spitz'])

        data_dict[name] = {'specimen':specimen, 'X': np.array(X), 'y': np.array(y)}

    clf = LogisticRegression(random_state=0, penalty='l2')
    clf.fit(data_dict['training']['X'], data_dict['training']['y'])
    val_prediction = clf.predict_proba(data_dict['validation']['X'])[:, 1]

    # determine best threshold based on the validation set
    val_evaluation = evaluate_with_thresholds(data_dict['validation']['y'], val_prediction, np.linspace(0, 1, 101))

    if False:
        predictions = {
            'specimen': data_dict['validation']['specimen'],
            'y_true': data_dict['validation']['y'],
            'y_pred': val_prediction,
        }
        df_val_prediction = pd.DataFrame.from_dict(predictions)
        df_val_prediction.to_excel('pred.xlsx')

    best_accuracy = max(val_evaluation['accuracy'])
    best_threshold = val_evaluation['threshold'][val_evaluation['accuracy'].index(best_accuracy)]

    val_AUROC = evaluate_without_thresholds(data_dict['validation']['y'], val_prediction, 10)['AUROC'][0]

    # define accuracy wrapper
    def accuracy_wrapper(y_true, y_pred):
        return evaluate_with_thresholds(y_true, y_pred, best_threshold)['accuracy'][0]

    def AUROC_wrapper(y_true, y_pred):
        return evaluate_without_thresholds(y_true, y_pred, 10)['AUROC'][0]

    bootstrap_accuracy_output = bootstrapper(
        y_true=data_dict['validation']['y'], 
        y_pred=val_prediction, 
        eval_func=accuracy_wrapper, 
        iterations=bootstrap_iterations, 
        stratified=False,
        confidence_level=bootstrap_confidence_level,
        seed=seed,
    )
    bootstrap_AUROC_output = bootstrapper(
        y_true=data_dict['validation']['y'], 
        y_pred=val_prediction, 
        eval_func=AUROC_wrapper, 
        iterations=bootstrap_iterations, 
        stratified=False,
        confidence_level=bootstrap_confidence_level,
        seed=seed,
    )
    print(f'Best validation accuracy: {best_accuracy:0.3f} (best threshold: {best_threshold})')
    print(f'Coefficients: {clf.coef_}')
    print(f'Bootstrap validationaccuracy results: {bootstrap_accuracy_output}')
    print(f'Validation AUROC: {val_AUROC:0.3f}')
    print(f'Bootstrap validation AUROC results: {bootstrap_AUROC_output}')

    # evaluate on test set
    if evaluate_on_test:
        test_prediction = clf.predict_proba(data_dict['test']['X'])[:, 1]
        test_accuracy = evaluate_with_thresholds(data_dict['test']['y'], test_prediction, best_threshold)['accuracy'][0]
        test_AUROC = evaluate_without_thresholds(data_dict['test']['y'], test_prediction, 10, 'roc.png', 'pr.png', 'ece.png')['AUROC'][0]
        
        bootstrap_accuracy_output = bootstrapper(
            y_true=data_dict['test']['y'], 
            y_pred=test_prediction, 
            eval_func=accuracy_wrapper, 
            iterations=bootstrap_iterations, 
            stratified=False,
            confidence_level=bootstrap_confidence_level,
            seed=seed,
        )
        bootstrap_AUROC_output = bootstrapper(
            y_true=data_dict['test']['y'], 
            y_pred=test_prediction, 
            eval_func=AUROC_wrapper, 
            iterations=bootstrap_iterations, 
            stratified=False,
            confidence_level=bootstrap_confidence_level,
            seed=seed,
        )
        print(f'Test accuracy: {test_accuracy:0.3f}')
        print(f'Bootstrap test accuracy results: {bootstrap_accuracy_output}')
        print(f'Test AUROC: {test_AUROC:0.3f}')
        print(f'Bootstrap test AUROC results: {bootstrap_AUROC_output}')