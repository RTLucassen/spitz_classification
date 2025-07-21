import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluation.evaluation_service import (
    evaluate_with_thresholds, 
    evaluate_highest_probability,
    evaluate_without_thresholds, 
    bootstrapper,
)

# define evaluation settings
bootstrap_iterations = 10000
bootstrap_confidence_level = 0.95
seed = 12345
evaluate_on_test = True

variant = 'internal'
image_variant_mapping = {
    'training': 'internal',
    'validation': 'internal',
    'test': 'internal',
}
balanced_classes = False
include_clinical_feat = True
include_image_feat = False
include_image_pred = False

folds = ['fold-1', 'fold-2', 'fold-3', 'fold-4', 'fold-5']
test = ['test']

label = 'label_spitz_dignity'

path = r"data.xlsx"
feat_pred_path = r"spitz_classification\models\201-205__uni_224__ensemble\SvsCM_features.json"


if __name__ == '__main__':

    # load clinical information
    df = pd.read_excel(path)
    if variant == 'internal':
        df = df[df['internal_paths'] != '[]']
    elif variant == 'consultation':
        df = df[df['consultation_paths'] != '[]']
    elif variant == 'combined':
        pass
    else:
        raise ValueError(f'Invalid variant name: {variant}')

    # select only the Spitz cases for the Spitz-only tasks
    if label in ['label_spitz_signature', 'label_spitz_dignity']:
        df = df[df['category'] == 'SPITZ']
    if label == 'label_spitz_dignity':
        grouped_label = []
        for i, row in df.iterrows():
            if row[label] == 0:
                grouped_label.append(0)
            elif row[label] in [1,2]:
                grouped_label.append(1)
            elif row[label] in [3,4]:
                grouped_label.append(2)
            else:
                raise ValueError
        df[label] = grouped_label

    # determine the number of classes
    N_classes = len(set(df[label]))

    # load feature vectors and predictions by image model
    if include_image_feat or include_image_pred:
        with open(feat_pred_path, 'r') as f:
            feat_pred_dict = json.loads(f.read())

    classifiers = []
    best_thresholds = []
    for i in range(len(folds)):
        validation = [folds[i]]
        training = [fold for fold in folds if fold != validation[0]]

        data_dict = {}
        for name, subset in [('training', training), ('validation', validation), ('test', test)]:
            df_subset = df[df['set'].isin(subset)]

            X = []
            y = []
            for i, row in df_subset.iterrows():
                feature = []
                if include_clinical_feat:
                    feature.append(row['age']/100)
                    feature.append(1 if row['sex'] == 'M' else 0)
                    feature.append(1 if (row['location_group'] == 'HEAD') or (row['location_group'] == 'NECK') else 0)
                    feature.append(1 if (row['location_group'] == 'HAND') or (row['location_group'] == 'HANDPALM') or (row['location_group'] == 'FOOT') or (row['location_group'] == 'FOOTSOLE') else 0)
                    feature.append(1 if (row['location_group'] == 'TRUNK') or (row['location_group'] == 'BUTTOCK') else 0)
                    feature.append(1 if row['location_group'] == 'UPPER EXTREMITY' else 0)
                    feature.append(1 if row['location_group'] == 'LOWER EXTREMITY' else 0)
                if include_image_feat:
                    try:
                        feat = feat_pred_dict[validation[0]][image_variant_mapping[name]][row['specimen']]['feat_vector']
                    except KeyError as error:
                        print(f'SKIPPED - {error}')
                        continue
                    else:
                        feature.extend(feat)
                if include_image_pred:
                    try:
                        pred = feat_pred_dict[validation[0]][image_variant_mapping[name]][row['specimen']]['y_pred']
                    except KeyError as error:
                        print(f'SKIPPED - {error}')
                        continue
                    else:
                        feature.append(pred[0])     
                
                X.append(feature)
                y.append(row[label])

            data_dict[name] = {'X': np.array(X), 'y': np.array(y)}

        clf = LogisticRegression(random_state=seed, penalty='l2', max_iter=250, 
                                 class_weight='balanced' if balanced_classes else None)
        clf.fit(data_dict['training']['X'], data_dict['training']['y'])

        if N_classes == 2:
            val_prediction = clf.predict_proba(data_dict['validation']['X'])[:, 1]

            # determine best threshold based on the validation set
            val_evaluation = evaluate_with_thresholds(data_dict['validation']['y'], val_prediction, np.linspace(0, 1, 101))

            best_accuracy = max(val_evaluation['accuracy'])
            best_threshold = val_evaluation['threshold'][val_evaluation['accuracy'].index(best_accuracy)]

            # save the best threshold
            best_thresholds.append(best_threshold)
        
        # save the classifier
        classifiers.append(clf)

    # define accuracy and AUROC wrappers
    if N_classes == 2:
        # get the mean best threshold
        mean_best_threshold = sum(best_thresholds)/len(best_thresholds)

        def accuracy_wrapper(y_true, y_pred):
            return evaluate_with_thresholds(y_true, y_pred, mean_best_threshold)['accuracy'][0]
    else:
        def accuracy_wrapper(y_true, y_pred):
            return evaluate_highest_probability(y_true, y_pred, True)['accuracy'][0]

    def AUROC_wrapper(y_true, y_pred):
        return evaluate_without_thresholds(y_true, y_pred, 10)['AUROC'][0]

    # evaluate on test set
    if evaluate_on_test:
        # calculate the ensemble prediction using the predictions for each classifier
        pred_prob = []
        pred_class = []
        if N_classes == 2:
            for clf, threshold in zip(classifiers, best_thresholds):
                prediction = clf.predict_proba(data_dict['test']['X'])
                pred_prob.append(prediction)
                pred_class.append((prediction[:, 1] > threshold).astype(int))
        else:
            for clf in classifiers:
                prediction = clf.predict_proba(data_dict['test']['X'])
                pred_prob.append(prediction)
                pred_class.append(np.argmax(prediction, axis=1))

        mean_pred_prob = np.mean(np.array(pred_prob), axis=0)
        if N_classes == 2:
            ensemble_test_accuracy = evaluate_with_thresholds(data_dict['test']['y'], mean_pred_prob[:, 1], mean_best_threshold)['accuracy'][0]
            ensemble_test_AUROC = evaluate_without_thresholds(data_dict['test']['y'], mean_pred_prob[:, 1], 10)['AUROC'][0]
        
            bootstrap_accuracy_output = bootstrapper(
                y_true=data_dict['test']['y'], 
                y_pred=mean_pred_prob[:, 1], 
                eval_func=accuracy_wrapper, 
                iterations=bootstrap_iterations, 
                stratified=False,
                confidence_level=bootstrap_confidence_level,
                seed=seed,
            ) 
            bootstrap_AUROC_output = bootstrapper(
                y_true=data_dict['test']['y'], 
                y_pred=mean_pred_prob[:, 1], 
                eval_func=AUROC_wrapper, 
                iterations=bootstrap_iterations, 
                stratified=False,
                confidence_level=bootstrap_confidence_level,
                seed=seed,
            )
            print(f'Test accuracy: {ensemble_test_accuracy:0.3f}')
            print(f'Bootstrap test accuracy results: {bootstrap_accuracy_output}')

            print(f'Test AUROC: {ensemble_test_AUROC:0.3f}')
            print(f'Bootstrap test AUROC results: {bootstrap_AUROC_output}')
        
        else:
            ensemble_test_accuracy = evaluate_highest_probability(
                data_dict['test']['y'].astype(int), 
                mean_pred_prob.tolist(), True,
            )['accuracy'][0]

            bootstrap_accuracy_output = bootstrapper(
                y_true=data_dict['test']['y'].astype(int), 
                y_pred=mean_pred_prob.tolist(), 
                eval_func=accuracy_wrapper, 
                iterations=bootstrap_iterations, 
                stratified=False,
                confidence_level=bootstrap_confidence_level,
                seed=seed,
            ) 
            print(f'Test accuracy: {ensemble_test_accuracy:0.3f}')
            print(f'Bootstrap test accuracy results: {bootstrap_accuracy_output}')

            for class_index in range(N_classes):

                one_vs_rest_label = [int(class_index == int(pred)) for pred in data_dict['test']['y']]
                one_vs_rest_pred_prob = [pred[class_index] for pred in mean_pred_prob]

                ensemble_test_AUROC = evaluate_without_thresholds(
                    y_true=one_vs_rest_label, 
                    y_pred=one_vs_rest_pred_prob,
                    bins=10,                                                    
                )['AUROC'][0]
                bootstrap_AUROC_output = bootstrapper(
                    y_true=one_vs_rest_label, 
                    y_pred=one_vs_rest_pred_prob, 
                    eval_func=AUROC_wrapper, 
                    iterations=bootstrap_iterations, 
                    stratified=False,
                    confidence_level=bootstrap_confidence_level,
                    seed=seed,
                )
                print(f'Test AUROC - class {class_index}: {ensemble_test_AUROC:0.3f}')
                print(f'Bootstrap test AUROC results: {bootstrap_AUROC_output}')