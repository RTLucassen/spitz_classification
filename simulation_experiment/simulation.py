import random
from functools import partial
from statistics import mean, median
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

IHC_DAYS = 1
ARCHER_DAYS = 10


def baseline(
    y_true: list[int], 
    y_pred: list[float], 
    false_negative: bool, 
    IHCs: list[str] = ['IHC ALK', 'IHC ROS1', 'IHC NTRK'],
) -> list[str]:
    """
    """
    tests = IHCs.copy()
    days = IHC_DAYS
    exams = 2
    
    if mapping(y_true) == 'OTHER' or false_negative:
        tests.append('ARCHER')
        days += ARCHER_DAYS
        exams += 1

    return tests, days, exams


def sequential(
    y_true: list[int], 
    y_pred: list[float], 
    false_negative: bool, 
    IHCs: list[str] = ['IHC NTRK', 'IHC ROS1', 'IHC ALK'],
) -> list[str]:
    """
    """
    tests = []
    days = 0
    exams = 1
    for test in IHCs:
        tests.append(test)
        days += IHC_DAYS
        exams += 1
        if mapping(y_true) in test and not false_negative:
            return tests, days, exams
    
    tests.append('ARCHER')
    days += ARCHER_DAYS
    exams += 1
    
    return tests, days, exams


def AI_based_recommendation(
    y_true: list[int], 
    y_pred: list[float], 
    false_negative: bool, 
    variant: int,
    other_threshold: Optional[float] = None,
) -> list[str]:
    """
    """
    tests = []
    days = 0
    exams = 1

    # get the binary prediction
    pred_order = []
    for prob in sorted(y_pred, reverse=True):
        y_pred_bin = [0,0,0,0]
        y_pred_bin[y_pred.index(prob)] = 1
        pred_order.append(mapping(y_pred_bin))

    # request the corresponding test
    if pred_order[0] == 'OTHER':
        if other_threshold is None:
            tests.append('ARCHER')
            days += ARCHER_DAYS
            exams += 1
            return tests, days, exams
        elif max(y_pred) > other_threshold:
            tests.append('ARCHER')
            days += ARCHER_DAYS
            exams += 1
            return tests, days, exams

    pred_order.remove('OTHER')
    if variant == 1:
        return baseline(y_true, y_pred, false_negative)
    elif variant == 2:
        return sequential(y_true, y_pred, false_negative)
    elif variant == 3:
        return sequential(y_true, y_pred, false_negative, IHCs=[f'IHC {i}' for i in pred_order])
    else:
        raise NotImplementedError


def perfect_recommendation(
    y_true: list[int], 
    y_pred: list[float], 
    false_negative: bool, 
    variant: int,
) -> list[str]:
    """
    """
    tests = []
    days = 0
    exams = 1

    # NOTE (order of classes after the correct prediction in first place does not matter)
    pred_order = [mapping(y_true)]
    for i in range(len(y_true)): 
        y_pred_bin = [0,0,0,0]
        y_pred_bin[i] = 1
        if mapping(y_pred_bin) not in pred_order:
            pred_order.append(mapping(y_pred_bin))

    # request the corresponding test
    if pred_order[0] == 'OTHER':
        tests.append('ARCHER')
        days += ARCHER_DAYS
        exams += 1
        return tests, days, exams

    pred_order.remove('OTHER')
    if variant == 1:
        return baseline(y_true, y_pred, false_negative)
    elif variant == 2:
        return sequential(y_true, y_pred, false_negative)
    elif variant == 3:
        return sequential(y_true, y_pred, false_negative, IHCs=[f'IHC {i}' for i in pred_order])
    else:
        raise NotImplementedError


def mapping(y_true):
    if y_true[0] == 1:
        return 'ALK'
    elif y_true[1] == 1:
        return 'ROS1'
    elif y_true[2] == 1:
        return 'NTRK'
    elif y_true[3] == 1:
        return 'OTHER'
    else:
        raise ValueError


# define the path and sheet name
path = r"results_internal_test.xlsx"
sheet_name = 'Predictions'

# load spreadsheets and add the consultation status, 
df = pd.read_excel(path, sheet_name=sheet_name)
cases = [(eval(row['y_true']), eval(row['y_pred'])) for _, row in df.iterrows()]

# define case study settings
false_negative_IHC = {
    'ALK': 1-0.945,
    'ROS1': 1-0.552,
    'NTRK': 1-0.745,
}
cost_per_test = {
    'IHC ALK': 100,
    'IHC ROS1': 100,
    'IHC NTRK': 100,
    'ARCHER': 1000,
}
N_cases = 100
sample_with_replacement = True
iterations = 10000
seed = 1

# other settings
confidence_level = 0.95

# define methods
methods = {
    'baseline': baseline, 
    'sequential': sequential,
    'AI-based recommendation thresholded (baseline)': partial(AI_based_recommendation, other_threshold=0.5, variant=1),
    'AI-based recommendation thresholded (sequential freq)': partial(AI_based_recommendation, other_threshold=0.5, variant=2),
    'AI-based recommendation thresholded (sequential order)': partial(AI_based_recommendation, other_threshold=0.5, variant=3),
    'Perfect AI-based recommendation thresholded (baseline)': partial(perfect_recommendation, variant=1),
    'Perfect AI-based recommendation thresholded (sequential freq)': partial(perfect_recommendation, variant=2),
    'Perfect AI-based recommendation thresholded (sequential order)': partial(perfect_recommendation, variant=3),
}

if __name__ == '__main__':

    if not sample_with_replacement and N_cases > len(cases):
        raise ValueError('Sampling will result in duplicates')

    # set seed
    random.seed(seed)
    np.random.seed(seed)    

    comparison = {}
    for _ in tqdm(range(iterations)):
        # sample indices
        if sample_with_replacement:
            indices = [random.randint(0, len(cases)-1) for _ in range(N_cases)]
        else:
            indices = list(range(len(cases)))
            indices = random.shuffle(indices)[:N_cases]

        # select cases
        selected_cases = [cases[i] for i in indices]
        # generate probabilities for false negative IHCs
        false_negative_values = [random.random() for _ in indices]

        for name, method in methods.items():
            if name not in comparison:
                comparison[name] = {'cost': [], 'time': [], 'examinations': []}

            cost = []
            time = []
            examinations = []
            for i, (y_true, y_pred) in enumerate(selected_cases):
                # determine whether the case is a false negative based on IHC
                subtype = mapping(y_true)
                if subtype == 'OTHER':
                    false_negative = None
                else:
                    false_negative = false_negative_values[i] < false_negative_IHC[subtype]

                # calculate the cost for the diagnostic tests
                tests, days, exams = method(y_true, y_pred, false_negative)
                cost.append(sum([cost_per_test[test] for test in tests]))
                time.append(days)
                examinations.append(exams)
            # store the cost
            comparison[name]['cost'].append(sum(cost))
            comparison[name]['time'].append(mean(time))
            comparison[name]['examinations'].append(mean(examinations))
        
    for name in methods:
        for metric in ['cost', 'time', 'examinations']:
            avg = np.mean(comparison[name][metric])
            lower = np.quantile(comparison[name][metric], (1-confidence_level)/2)
            upper = np.quantile(comparison[name][metric], 1-((1-confidence_level)/2))
            if metric == 'cost':
                print(f'{name}:  {avg:0.0f} (95% CI, {lower:0.0f}-{upper:0.0f}) euros')
            elif metric == 'time':
                print(f'{name}:  {avg:0.2f} (95% CI, {lower:0.2f}-{upper:0.2f}) days')
            elif metric == 'examinations':
                print(f'{name}:  {avg:0.2f} (95% CI, {lower:0.2f}-{upper:0.2f}) times')
        print('')