# @Author  : zhangyouyuan (zyyzhang@fuzhi.ai)
# @Desc    :
import numpy as np
import sklearn.metrics as sklearn_metric
import torch

from utils.const import KEEP_DIGITS_NUM, EPSILON
KEEP_DIGITS_NUM = 4
EPSILON = 1e-9
class ClassifyMetric(object):
    def __init__(self, metric_type='AUC', optimize_type='max', classes_map=None, **kwargs):
        self.metric_type = metric_type,
        self.optimize_type = optimize_type,

        self.classes_map = classes_map
        self.classes_map_reverse = {v: k for k, v in self.classes_map.items()}

    def compute_metric(self, predicts, targets, *args, **kwargs):
        if isinstance(predicts, torch.Tensor):
            predicts = predicts.detach().cpu().numpy()
        elif isinstance(predicts, (list, tuple)):
            predicts = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in predicts]

        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        elif isinstance(targets, (list, tuple)):
            targets = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in targets]

        self.gathered_outputs = predicts
        self.gathered_targets = targets

        predicts = np.array(predicts)
        targets = np.array(targets)

        metric = dict()
        num_classes = predicts.shape[-1]
        metric['AUC'] = get_auc(predicts, targets, average='macro', num_classes=num_classes)
        metric['eval_auc'] = metric['AUC']
        metric['eval_auc_macro'] = metric['AUC']
        metric['eval_auc_micro'] = get_auc(predicts, targets, average='micro', num_classes=num_classes)
        metric['fpr'], metric['tpr'], metric['split_point'] = get_fpr_tpr(predicts, targets, num_classes=num_classes)
        metric['precision'], metric['recall'], metric['eval_pr'] = get_precision_recall(predicts, targets,
                                                                                        num_classes=num_classes)

        inputs_argmax = np.argmax(predicts, axis=-1)
        if num_classes == 2:
            inputs_argmax += (inputs_argmax == np.argmin(predicts, axis=-1))  # 与auc计算一致，>=0.5 归为正类
        predicts = inputs_argmax

        metric['Precision'] = get_precision(predicts, targets)
        metric['Recall'] = get_recall(predicts, targets)
        metric['ACC'] = get_acc(predicts, targets)
        metric['MacroF1'] = get_macro_f1(predicts, targets)
        metric['MicroF1'] = get_micro_f1(predicts, targets)
        metric['WeightF1'] = get_weight_f1(predicts, targets)
        metric['F1'] = get_f1(predicts, targets)
        metric['LogLoss'] = get_log_loss(predicts, targets, num_classes)

        metric['eval_acc_score'] = metric['ACC']
        metric['eval_pi_score'] = metric['Precision']
        metric['eval_f1_score'] = metric['F1']
        metric['eval_recall_score'] = metric['Recall']
        metric['eval_avg_pi_score'] = metric['eval_pr']

        metric['confusion_matrix'], metric_confusion = get_confusion_matrix(
            predicts, targets, self.classes_map_reverse, metric_type=['npv', 'tnr'])
        metric['npv'] = metric_confusion['npv']
        metric['tnr'] = metric_confusion['tnr']

        return metric


def get_confusion_matrix(inputs, targets, classes_map_reverse, metric_type=None):
    sample_num = len(inputs)
    labels = list(set(targets))
    labels.sort()
    confusion_matrix = {
        'matrix': None,
        'labels': [classes_map_reverse[x] for x in labels]
    }

    # 行和列分别表示真实值和预测值
    matrix = sklearn_metric.confusion_matrix(targets, inputs, labels=labels)

    # get metric
    all_metric = {x: [] for x in metric_type}
    if metric_type:
        for class_id in range(len(labels)):
            tp = matrix[class_id, class_id]
            fp = np.sum(matrix[:, class_id]) - tp
            fn = np.sum(matrix[class_id, :]) - tp
            tn = np.sum(matrix) - tp - fp - fn

            if 'npv' in metric_type:
                npv = tn / (tn + fn + EPSILON)
                all_metric['npv'].append(npv)

            if 'tnr' in metric_type:
                tnr = tn / (fp + tn + EPSILON)
                all_metric['tnr'].append(tnr)

        all_metric = {k: round(float(np.mean(v)), KEEP_DIGITS_NUM) for k, v in all_metric.items()}

    # format matrix
    matrix = [[round(y, KEEP_DIGITS_NUM) for y in x] for x in matrix / sample_num]
    confusion_matrix['matrix'] = matrix
    return (confusion_matrix, all_metric) if metric_type else confusion_matrix


def get_precision(inputs, targets):
    Precision = sklearn_metric.precision_score(targets, inputs, average='macro', zero_division=0)
    return round(Precision, KEEP_DIGITS_NUM)


def get_recall(inputs, targets):
    Recall = sklearn_metric.recall_score(targets, inputs, average='macro', zero_division=0)
    return round(Recall, KEEP_DIGITS_NUM)


def get_acc(inputs, targets):
    ACC = sklearn_metric.accuracy_score(targets, inputs)
    return round(ACC, KEEP_DIGITS_NUM)


def get_auc(inputs, targets, num_classes, average='macro'):
    sample_num = len(targets)

    if num_classes == 2:  # deal binary classify
        inputs_trans = inputs[:, 1]  # 将正样本的label标记为 1
        targets_trans = targets
    else:
        if average == 'micro':
            inputs_trans = inputs.ravel()
            targets_trans = np.zeros((sample_num, num_classes))
            targets_trans[range(sample_num), targets] = 1
            targets_trans = targets_trans.ravel()
        else:
            inputs_trans = inputs
            targets_trans = targets

            # deal the extract content - num_classes in test less than train
            sort_set_targets = list(set(targets_trans))
            if len(sort_set_targets) < num_classes:
                sort_set_targets.sort()
                inputs_trans = inputs_trans[:, np.array(sort_set_targets)]
                inputs_trans = torch.softmax(torch.from_numpy(inputs_trans), dim=-1).numpy()
    try:
        if average == 'micro':
            auc = sklearn_metric.roc_auc_score(targets_trans, inputs_trans, average='micro')
        else:
            auc = sklearn_metric.roc_auc_score(targets_trans, inputs_trans, average='macro', multi_class='ovr')
    except Exception as exp:
        auc = 0.5
    return round(auc, KEEP_DIGITS_NUM)


def get_fpr_tpr(inputs, targets, num_classes):
    curve_sample_num = 100
    sample_num = len(targets)
    if num_classes != 2:  # deal binary classify
        inputs_trans = inputs.ravel()
        targets_trans = np.zeros((sample_num, num_classes))
        targets_trans[range(sample_num), targets] = 1
        targets_trans = targets_trans.ravel()
    else:
        inputs_trans = inputs[:, 1]  # 将正样本的label标记为 1
        targets_trans = targets

    try:
        fpr, tpr, thresholds = sklearn_metric.roc_curve(targets_trans, inputs_trans)
        split_point = round(thresholds[np.argmax(tpr - fpr)], KEEP_DIGITS_NUM)

        if len(fpr) <= curve_sample_num:
            step = 1
        else:
            step = int(len(fpr) / curve_sample_num)
        sample_idx = list(range(0, len(fpr), step))
        fpr_sample = fpr[sample_idx]
        tpr_sample = tpr[sample_idx]

        fpr = [round(float(v), KEEP_DIGITS_NUM) for v in list(fpr_sample)]
        tpr = [round(float(v), KEEP_DIGITS_NUM) for v in list(tpr_sample)]
    except Exception as exp:
        fpr, tpr, split_point = [], [], 0.0

    return fpr, tpr, split_point


def get_precision_recall(inputs, targets, num_classes):
    curve_sample_num = 100
    sample_num = len(targets)
    if num_classes != 2:  # deal binary classify
        inputs_trans = inputs.ravel()
        targets_trans = np.zeros((sample_num, num_classes))
        targets_trans[range(sample_num), targets] = 1
        targets_trans = targets_trans.ravel()
    else:
        inputs_trans = inputs[:, 1]  # 将正样本的label标记为 1
        targets_trans = targets

    try:
        pr_value = round(sklearn_metric.average_precision_score(targets_trans, inputs_trans), KEEP_DIGITS_NUM)
        precision, recall, thresholds = sklearn_metric.precision_recall_curve(targets_trans, inputs_trans)

        if len(precision) <= curve_sample_num:
            step = 1
        else:
            step = int(len(precision) / curve_sample_num)
        sample_idx = list(range(0, len(precision), step))
        precision_sample = precision[sample_idx]
        recall_sample = recall[sample_idx]

        precision = [round(float(v), KEEP_DIGITS_NUM) for v in list(precision_sample)]
        recall = [round(float(v), KEEP_DIGITS_NUM) for v in list(recall_sample)]
    except Exception as exp:
        precision, recall, pr_value = [], [], 0.0

    return precision, recall, pr_value


def get_macro_f1(inputs, targets):
    MacroF1 = sklearn_metric.f1_score(targets, inputs, average='macro')
    return round(MacroF1, KEEP_DIGITS_NUM)


def get_micro_f1(inputs, targets):
    MicroF1 = sklearn_metric.f1_score(targets, inputs, average='micro')
    return round(MicroF1, KEEP_DIGITS_NUM)


def get_weight_f1(inputs, targets):
    WeightF1 = sklearn_metric.f1_score(targets, inputs, average='weighted')
    return round(WeightF1, KEEP_DIGITS_NUM)


def get_f1(inputs, targets):
    # F1 = sklearn_metric.f1_score(targets, inputs, average='samples')
    F1 = get_macro_f1(inputs, targets)
    return round(F1, KEEP_DIGITS_NUM)


def get_log_loss(inputs, targets, num_classes):
    targets_onehot = np.eye(num_classes)[targets]
    inputs_onehot = np.eye(num_classes)[inputs]
    LogLoss = sklearn_metric.log_loss(targets_onehot, inputs_onehot)
    return round(LogLoss, KEEP_DIGITS_NUM)
