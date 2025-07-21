import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats.contingency import crosstab
from scipy.stats import binomtest
import numpy as np
from evaluation.evaluation_service import (
    bootstrapper,
)

def Q1_evaluation(y_trues: list[str], y_preds: list[str], positive: str) -> dict:
    """
    Evaluation of answers for Q1 by pathologists

    Args:
        y_trues:  List with correct labels
        y_preds:  List with labels predicted by pathologists
        positive:  Name of the positive class ('Spitz Tumor' or 'Conv. Melanoma')

    Returns:
        results:  Dictionary with results
    """
    results = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'TPR': None, 'FPR': None, 'ACC': None, 'CORRECT': []}
    for y_true, y_pred in zip(y_trues, y_preds):
        if positive == 'Spitz Tumor':
            if y_true == 'Spitz Tumor' and y_pred == 'Spitz Tumor':
                results['TP'] += 1
                results['CORRECT'].append(1)
            elif y_true == 'Conv. Melanoma' and y_pred == 'Spitz Tumor':
                results['FP'] += 1
                results['CORRECT'].append(0)
            elif y_true == 'Spitz Tumor' and y_pred == 'Conv. Melanoma':
                results['FN'] += 1
                results['CORRECT'].append(0)
            else:
                results['TN'] += 1
                results['CORRECT'].append(1)
        elif positive == 'Conv. Melanoma':
            if y_true == 'Spitz Tumor' and y_pred == 'Spitz Tumor':
                results['TN'] += 1
                results['CORRECT'].append(1)
            elif y_true == 'Conv. Melanoma' and y_pred == 'Spitz Tumor':
                results['FN'] += 1
                results['CORRECT'].append(0)
            elif y_true == 'Spitz Tumor' and y_pred == 'Conv. Melanoma':
                results['FP'] += 1
                results['CORRECT'].append(0)
            else:
                results['TP'] += 1
                results['CORRECT'].append(1)
        else:
            raise ValueError
                 
    results['TPR'] = results['TP']/(results['TP']+results['FN'])
    results['FPR'] = results['FP']/(results['FP']+results['TN'])
    results['ACC'] = ((results['TP']+results['TN'])/(results['TP']+results['TN']+results['FP']+results['FN']))

    return results


# define the positive class
positive = 'Conv. Melanoma'
bootstrap_iterations = 10000
bootstrap_stratified = True
confidence_level = 0.95
bins = 10
seed = 1


# define the path to the spreadsheet with pathologist decisions and load it 
path = r"spitz_classification\sheets\reader_study.xlsx"
df = pd.read_excel(path)

# load the mapping from image names to specimen IDs
mapping_path = r"dataset_reader_study.xlsx"
mapping_df = pd.read_excel(mapping_path)
mapping = {}
for _, row in mapping_df.iterrows():
    mapping[row['new_name'].replace('.ndpi', '').upper().split('_')[-1]] = row['specimen']

# Q1
if True:
    tprs = []
    fprs = []
    correct = {}
    # loop over the pathologist indices
    for i in range(1,5):
        print(f'\nResults Q1 - pathologist {i}')
        # select the answers from one pathologist for Q1
        df_selection = df[(df['Question'] == 'Q1 - Spitz Tumor vs. Conventional Melanoma') & (df['PATH'] == f'Pathologist {i}')]
        # get the predictions, ground truth labels, and the image names
        y_true = df_selection['Ground truth'].tolist()
        y_pred = df_selection['PATH Answer'].tolist()
        image_names = df_selection['Image Name'].tolist()
        # calculate summary statistics
        result = Q1_evaluation(y_true, y_pred, positive)
        tprs.append(result['TPR'])
        fprs.append(result['FPR'])
        correct[f'Q{i}'] = result['CORRECT']
        print(result)

        # perform bootstrapping for confidence intervals
        if bootstrap_iterations is not None:
            # define accuracy wrapper
            def accuracy_wrapper(y_true, y_pred):
                return Q1_evaluation(y_true, y_pred, positive)['ACC']
            # perform bootstrapping for accuracy at specific threshold value
            bootstrap_output = bootstrapper(
                y_true=y_true, 
                y_pred=y_pred, 
                eval_func=accuracy_wrapper, 
                iterations=bootstrap_iterations, 
                stratified=bootstrap_stratified,
                confidence_level=confidence_level, 
                seed=seed,
                show_progress=False,
            )
            print(bootstrap_output)

    # select all results from Q1
    df_selection = df[(df['Question'] == 'Q1 - Spitz Tumor vs. Conventional Melanoma')]

    y_pred_dict = {}
    y_true_dict = {}
    for i, row in df_selection.iterrows():
        if row['ImageID'] not in y_pred_dict:
            y_pred_dict[row['ImageID']] = []
        y_pred_dict[row['ImageID']].append(1 if row['PATH Answer'] == positive else 0)
        y_true_dict[row['ImageID']] = 1 if row['Ground truth'] == positive else 0
    # get the mean prediction
    y_pred_dict = {k:sum(v)/len(v) for k,v in y_pred_dict.items()}
    # order the cases
    y_pred = [y_pred_dict[k] for k in y_pred_dict]
    y_true = [y_true_dict[k] for k in y_pred_dict]
    # perform ROC analysis
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    print(f'\nAveraged predictions of pathologists - AUROC: {auc_score}')

    # perform bootstrapping for confidence intervals
    if bootstrap_iterations is not None:
        # define AUROC wrapper
        def AUROC(y_trues, y_preds):
            fpr, tpr, _ = roc_curve(y_trues, y_preds)
            return auc(fpr, tpr)
        # perform bootstrapping for accuracy at specific threshold value
        bootstrap_output = bootstrapper(
            y_true=y_true, 
            y_pred=y_pred, 
            eval_func=AUROC, 
            iterations=bootstrap_iterations, 
            stratified=bootstrap_stratified,
            confidence_level=confidence_level, 
            seed=seed,
            show_progress=False,
        )
        print(bootstrap_output)

    # load the model ensemble predictions
    model_path = r"spitz_classification\models\201-205__uni_224__eval_path_ensemble\results_internal_test.xlsx"
    df_model = pd.read_excel(model_path)

    # loop over the image names
    y_pred_model = []
    y_true_model = []
    for image_name in image_names:
        df_model_selection = df_model[df_model['specimen'] == mapping[image_name.upper().split('_')[-1]]]
        y_pred_model.append(eval(df_model_selection['y_pred'].iloc[0])[1 if positive == 'Spitz Tumor' else 0])
        y_true_model.append(eval(df_model_selection['y_true'].iloc[0])[1 if positive == 'Spitz Tumor' else 0])
    fpr_model, tpr_model, _ = roc_curve(y_true_model, y_pred_model)

    # determine the correct model predictions
    threshold = 0.482
    y_pred_model_thresholded = [1 if p>threshold else 0 for p in y_pred_model]
    correct['model'] = [1 if i==j else 0 for i, j in zip(y_pred_model_thresholded, y_true_model)]
    
    # statistical analysis using McNemar's test
    for i in range(1,5):
        print(f'\nMcNemar Q1 - pathologist {i}')
        print(len(correct[f'Q{i}']), len(correct[f'model']))
        table = crosstab(correct[f'Q{i}'], correct[f'model'])
        print(table.count)
        print(mcnemar(table.count, exact=True))    
            
    # configure ROC plot
    fig, ax = plt.subplots(figsize=(5,5))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.3)

    # format ticks
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis="both", direction="in", length=5, width=1.3)
    ax.tick_params(which='minor', axis="both", direction="in", 
                    right='on', top='on', length=2.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.200))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_major_locator(MultipleLocator(0.200))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))

    # format axes
    offset = 0.001
    plt.xlim([0-offset, 1+offset])
    plt.ylim([0-offset, 1+offset])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')

    plt.plot(fpr, tpr, label=f'ROC curve - pathologists', lw=2, color="mediumblue")
    plt.plot(fpr_model, tpr_model, label=f'ROC curve - model', lw=2, color="darkblue")
    plt.scatter(fprs, tprs, label=f'pathologists', lw=2, color="red")

    plt.plot([0, 1], [0, 1], ls='--', color='black', lw=0.75)
    plt.legend(loc=4, borderaxespad=1.2, fancybox=False, edgecolor='black', fontsize=10)
    plt.savefig('Q1_results.png')
    plt.close()

# Q2
if True:   
    # load model results
    model_path = r"models\206-210__uni_224__eval_path_ensemble\results_internal_test.xlsx"
    df_model = pd.read_excel(model_path)
    
    for i in range(1,5):
        print(f'\nResults Q2 - pathologist {i}')
        # select the answers from one pathologist for Q2
        df_selection = df[(df['Question'] == 'Q2 - Spitz Subtype') & (df['PATH'] == f'Pathologist {i}')]

        y_true = []
        y_pred = []
        image_names = []
        for _, row in df_selection.iterrows():
            if row['Ground truth'] == 'Unknown':
                continue
            elif row['PATH Answer'] == 'Unknown':
                continue
            else:
                y_true.append(row['Ground truth'])
                y_pred.append(row['PATH Answer'])
                image_names.append(row['Image Name'])

        def accuracy(y_trues, y_preds):
            return sum([int(i == j) for i, j in zip(y_trues, y_preds)])/len(y_trues)
        
        path_correct = [1 if pred == true else 0 for pred, true in zip(y_pred, y_true)]
        acc = accuracy(y_true, y_pred)
        print(f'Accuracy: {acc:0.3f}')
        # perform bootstrapping for confidence intervals
        if bootstrap_iterations is not None:
            # perform bootstrapping for accuracy at specific threshold value
            bootstrap_output = bootstrapper(
                y_true=y_true, 
                y_pred=y_pred, 
                eval_func=accuracy, 
                iterations=bootstrap_iterations, 
                stratified=bootstrap_stratified,
                confidence_level=confidence_level, 
                seed=seed,
                show_progress=False,
            )
            print(bootstrap_output) 
            print('ALK', 'ROS1', 'NTRK', 'Other')
        print(confusion_matrix(y_true, y_pred, labels=['ALK', 'ROS1', 'NTRK', 'Other']))

        y_true_model = []
        y_pred_model = []
        for image_name in image_names:
            df_model_selection = df_model[df_model['specimen'] == mapping[image_name.upper().split('_')[-1]]]
            y_pred_model.append(np.argmax(eval(df_model_selection['y_pred'].iloc[0])))
            y_true_model.append(np.argmax(eval(df_model_selection['y_true'].iloc[0])))
        model_correct = [1 if pred == true else 0 for pred, true in zip(y_pred_model, y_true_model)]

        # statistical analysis using McNemar's test
        print(f'\nMcNemar - pathologist {i}')
        table = crosstab(path_correct, model_correct)
        print(table.count)
        print(mcnemar(table.count, exact=True)) 


if True:
    # select all results from Q2
    df_selection = df[(df['Question'] == 'Q2 - Spitz Subtype')]

    # loop over the targets
    for target in ['ALK', 'ROS1', 'NTRK', 'Other']:
        y_pred_dict = {}
        y_true_dict = {}
        for i in range(1,5):
            df_path = df_selection[df_selection['PATH'] == f'Pathologist {i}']
            for _, row in df_path.iterrows():
                if row['Ground truth'] == 'Unknown':
                    continue
                if row['ImageID'] not in y_pred_dict:
                    y_pred_dict[row['ImageID']] = []
                if row['PATH Answer'] == 'Unknown':
                    y_pred_dict[row['ImageID']].append(None)
                else:
                    y_pred_dict[row['ImageID']].append(1 if row['PATH Answer'] == target else 0)
                if i == 1:
                    y_true_dict[row['ImageID']] = 1 if row['Ground truth'].split(' ')[0] == target else 0   
        
        fprs = []
        tprs = []
        for i in range(4):
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for k in y_pred_dict:
                if y_pred_dict[k][i] == 1 and y_true_dict[k] == 1:
                    TP += 1
                elif y_pred_dict[k][i] == 1 and y_true_dict[k] == 0:
                    FP += 1
                elif y_pred_dict[k][i] == 0 and y_true_dict[k] == 1:
                    FN += 1
                else:
                    TN += 1
            fprs.append(FP/(FP+TN))
            tprs.append(TP/(TP+FN))

        # get the mean prediction
        y_pred_dict = {k: sum([w for w in v if w is not None])/len([w for w in v if w is not None]) for k,v in y_pred_dict.items()}

        # order the cases
        y_pred = [y_pred_dict[k] for k in y_pred_dict]
        y_true = [y_true_dict[k] for k in y_pred_dict]

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = auc(fpr, tpr)
        print(f'\nAveraged predictions of pathologists - {target} - AUROC: {auc_score:0.3f}')

        # perform bootstrapping for confidence intervals
        if bootstrap_iterations is not None:
            # define AUROC wrapper
            def AUROC(y_trues, y_preds):
                fpr, tpr, _ = roc_curve(y_trues, y_preds)
                return auc(fpr, tpr)
            # perform bootstrapping for accuracy at specific threshold value
            bootstrap_output = bootstrapper(
                y_true=y_true, 
                y_pred=y_pred, 
                eval_func=AUROC, 
                iterations=bootstrap_iterations, 
                stratified=bootstrap_stratified,
                confidence_level=confidence_level, 
                seed=seed,
                show_progress=False,
            )
            print(bootstrap_output)

        # configure ROC plot
        fig, ax = plt.subplots(figsize=(5,5))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.3)

        # format ticks
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis="both", direction="in", length=5, width=1.3)
        ax.tick_params(which='minor', axis="both", direction="in", 
                        right='on', top='on', length=2.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.200))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.xaxis.set_major_locator(MultipleLocator(0.200))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))

        # format axes
        offset = 0.001
        plt.xlim([0-offset, 1+offset])
        plt.ylim([0-offset, 1+offset])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')

        plt.plot(fpr, tpr, label=f'ROC curve - pathologists', lw=2, color="mediumblue")
        plt.scatter(fprs, tprs, label=f'pathologists', lw=2, color="red")

        plt.plot([0, 1], [0, 1], ls='--', color='black', lw=0.75)
        plt.legend(loc=4, borderaxespad=1.2, fancybox=False, edgecolor='black', fontsize=10)
        plt.savefig(f'Q2_{target}_results.png')
        plt.close()


if True:
    # load model results
    model_path = r"models\balanced\211-215__uni_224__eval_path_balanced_ensemble\results_internal_test.xlsx"
    df_model = pd.read_excel(model_path)

    # evaluate Q3 prediction
    for i in range(1,5):
        print(f'\nResults Q3 - pathologist {i}')
        # select the answers from one pathologist for Q1
        df_selection = df[(df['Question'] == 'Q3 - Spitz Dignity') & (df['PATH'] == f'Pathologist {i}')]

        y_true = []
        y_pred = []
        image_names = []
        for _, row in df_selection.iterrows():
            if row['Ground truth'] == 'Unknown':
                continue
            elif row['PATH Answer'] == 'Unknown':
                continue
            else:
                y_true.append(row['Ground truth'].split(' ')[0])
                y_pred.append(row['PATH Answer'])
                image_names.append(row['Image Name'])

        def accuracy(y_trues, y_preds):
            return sum([int(i == j) for i, j in zip(y_trues, y_preds)])/len(y_trues)
        
        path_correct = [1 if pred == true else 0 for pred, true in zip(y_pred, y_true)]
        acc = accuracy(y_true, y_pred)
        print(f'Accuracy: {acc:0.3f}')
        # perform bootstrapping for confidence intervals
        if bootstrap_iterations is not None:
            # perform bootstrapping for accuracy at specific threshold value
            bootstrap_output = bootstrapper(
                y_true=y_true, 
                y_pred=y_pred, 
                eval_func=accuracy, 
                iterations=bootstrap_iterations, 
                stratified=bootstrap_stratified,
                confidence_level=confidence_level, 
                seed=seed,
                show_progress=False,
            )
            print(bootstrap_output) 
            print('Benign', 'Intermediate', 'Malignant')
        print(confusion_matrix(y_true, y_pred, labels=['Benign', 'Intermediate', 'Malignant']))

        y_true_model = []
        y_pred_model = []
        for image_name in image_names:
            df_model_selection = df_model[df_model['specimen'] == mapping[image_name.upper().split('_')[-1]]]
            y_pred_model.append(np.argmax(eval(df_model_selection['y_pred'].iloc[0])))
            y_true_model.append(np.argmax(eval(df_model_selection['y_true'].iloc[0])))
        model_correct = [1 if pred == true else 0 for pred, true in zip(y_pred_model, y_true_model)]

        # statistical analysis using McNemar's test
        print(f'\nMcNemar - pathologist {i}')
        table = crosstab(path_correct, model_correct)
        print(table.count)
        print(mcnemar(table.count, exact=True)) 


if True:
    # select all results from Q3
    df_selection = df[(df['Question'] == 'Q3 - Spitz Dignity')]

    # loop over the targets
    for target in ['Benign', 'Intermediate', 'Malignant']:
        y_pred_dict = {}
        y_true_dict = {}
        for i in range(1,5):
            df_path = df_selection[df_selection['PATH'] == f'Pathologist {i}']
            for _, row in df_path.iterrows():
                if row['Ground truth'] == 'Unknown':
                    continue
                if row['ImageID'] not in y_pred_dict:
                    y_pred_dict[row['ImageID']] = []
                if row['PATH Answer'] == 'Unknown':
                    y_pred_dict[row['ImageID']].append(None)
                else:
                    y_pred_dict[row['ImageID']].append(1 if row['PATH Answer'] == target else 0)
                if i == 1:
                    y_true_dict[row['ImageID']] = 1 if row['Ground truth'].split(' ')[0] == target else 0     
            
        fprs = []
        tprs = []
        for i in range(4):
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for k in y_pred_dict:
                if y_pred_dict[k][i] is None:
                    continue
                if y_pred_dict[k][i] == 1 and y_true_dict[k] == 1:
                    TP += 1
                elif y_pred_dict[k][i] == 1 and y_true_dict[k] == 0:
                    FP += 1
                elif y_pred_dict[k][i] == 0 and y_true_dict[k] == 1:
                    FN += 1
                else:
                    TN += 1
            fprs.append(FP/(FP+TN))
            tprs.append(TP/(TP+FN))

        # get the mean prediction
        y_pred_dict = {k: sum([w for w in v if w is not None])/len([w for w in v if w is not None]) for k,v in y_pred_dict.items()}

        # order the cases
        y_pred = [y_pred_dict[k] for k in y_pred_dict]
        y_true = [y_true_dict[k] for k in y_pred_dict]

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = auc(fpr, tpr)
        print(f'\nAveraged predictions of pathologists - {target} - AUROC: {auc_score:0.3f}')

        # perform bootstrapping for confidence intervals
        if bootstrap_iterations is not None:
            # define AUROC wrapper
            def AUROC(y_trues, y_preds):
                fpr, tpr, _ = roc_curve(y_trues, y_preds)
                return auc(fpr, tpr)
            # perform bootstrapping for accuracy at specific threshold value
            bootstrap_output = bootstrapper(
                y_true=y_true, 
                y_pred=y_pred, 
                eval_func=AUROC, 
                iterations=bootstrap_iterations, 
                stratified=bootstrap_stratified,
                confidence_level=confidence_level, 
                seed=seed,
                show_progress=False,
            )
            print(bootstrap_output)

        # configure ROC plot
        fig, ax = plt.subplots(figsize=(5,5))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.3)

        # format ticks
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis="both", direction="in", length=5, width=1.3)
        ax.tick_params(which='minor', axis="both", direction="in", 
                        right='on', top='on', length=2.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.200))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.xaxis.set_major_locator(MultipleLocator(0.200))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))

        # format axes
        offset = 0.001
        plt.xlim([0-offset, 1+offset])
        plt.ylim([0-offset, 1+offset])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')

        plt.plot(fpr, tpr, label=f'ROC curve - pathologists', lw=2, color="mediumblue")
        plt.scatter(fprs, tprs, label=f'pathologists', lw=2, color="red")

        plt.plot([0, 1], [0, 1], ls='--', color='black', lw=0.75)
        plt.legend(loc=4, borderaxespad=1.2, fancybox=False, edgecolor='black', fontsize=10)
        plt.savefig(f'Q3_{target}_results.png')
        plt.close()

if True:
    print('Bionomial test for comparison of model performance to random guessing')
    print('Q1:', binomtest(k=164, n=191, p=0.5))
    print('Q2:',binomtest(k=54, n=99, p=0.25))
    print('Q3:',binomtest(k=50, n=99, p=0.33333))