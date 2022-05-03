#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   :

from typing import Dict
import uuid
import os


def gen_gp_train_output(model_name, trial_name, metric: Dict, model_path,
                        metric_type="AUC", time_use={}):
    """
    :desc generate train metric output
    :param model_name: model name
    :param trial_name: client_train input parameter
    :param metric: train output metric, dict
    :param model_path: model save dir
    :param metric_type: main metric of GP task
    :param time_use:
    :return:
    """

    assert isinstance(metric, Dict)
    assert metric_type in metric

    output_tpl = {
        "recom_model": model_name,                         # 必填，模型名称 = {model_name}
        "gened_model_name": f"{model_name}_{trial_name}",  # 必填，产出模型名称 = {model_name}_{trial_name}，对应保存的模型文件：ResNet50_ClassifyHead_autotables-2503-train-1.pth
        "metric_type": metric_type,                        # 必填，主评价指标key
        "metric_value": metric.get(metric_type),           # 必填，主评价指标value
        "used_models": model_name,                         # 必填，使用的模型名称
        "model_path": model_path,                          # 必填，产出模型存放的路径 示例="/app/tianji/runs/models/1c4bc928-d0c9-4189-a097-aade874896b6/"
        "model_params": {},                                # 选填，模型超参
        "eval_result": {
            model_name: {                                  # {recom_model}为对应模型名
                "eval": metric                             # eval下为指标的k-v结果
            }
        },  # 必填，评估指标
        "predict_prob": {},                                # 选填，验证集数据
        "train_consuming": {                               # 选填，不同阶段耗时
            model_name: time_use
        }
    }
    return output_tpl


def gen_gp_test_output(model_name, metric, model_path, time_use, predict_file_name):
    """
    :desc generate test metric output
    :param model_name: model name
    :param metric: test output metric, dict
    :param model_path: model save path
    :param time_use:
    :param predict_file_name:
    :return:
    """
    assert isinstance(metric, Dict)

    output_tpl = {
        "model_path": model_path,  # 必填，容器内预测结果存放路径 示例="/app/tianji/runs/models/1c4bc928-d0c9-4189-a097-aade874896b6/"
        "predict_result_filename": predict_file_name,  # 选填，离线预测结果放在model_path路径下，
                                                       # 离线预测最终本地路径={model_path}/{predict_result_filename}
        "preview_data": {},              # 选填，数据集预览结果
        "test_result": {
            model_name: {                # {recom_model}为对应模型名
                "test": metric           # test下为指标的k-v结果
            }
        },                               # 必填，离线验证指标
        "predict_prob": {},              # 选填，预测概率等，分类场景使用
        "test_consuming": {              # 选填，不同阶段耗时
            model_name: time_use
        }
    }
    return output_tpl
