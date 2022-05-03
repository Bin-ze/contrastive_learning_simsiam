### 任务介绍

  - 对比学习sota模型SimSiam的训练,无监督任务，为了验证其效果，将训练好的模型refine到下游分类任务并输出指标。
  - 参考repo:
            https://github.com/lightly-ai/lightly

- ### 数据集格式

    - 训练和测试
        - 原始数据集：
        - 处理好可以训练的数据集：
        - 用于调试的子集：
        - 自定义数据集规则：验证时使用分类数据集格式数据，进行对比学习训练时使用无标注的图像数据：
            ```
                |- dataset_name
                    |- train
                        |- images/                   #自定义数据集的训练数据,无监督图像数据，不需要任何额外的标注
                             |-img1.jpg
                             |-img2.jpg
                                 ...
                        |- annotations/
                    |- valid
                        |- images/                   #使用下游任务图像分类对算法进行验证，故测试验证集为分类数据
                            |- class1
                                |-img1.jpg
                                |-img2.jpg
                                ...
                            |- class2  
                                 |-img1.jpg
                                 |-img2.jpg
                                 ...
                            ...
                        |- annotations/
                    |- test                            #自定义数据集的测试数据
                        |- images/                  
                            |- class1
                                |-img1.jpg
                                |-img2.jpg
                                ...
                            |- class2  
                                 |-img1.jpg
                                 |-img2.jpg
                                 ...
                            ...
                        |- annotations/
             ```
           
            **help：分类数据集的annotations文件夹为空，类别直接从images的文件名中读取**。
            
                
   - infer 样本
        - 图片，分辨率不限。

### 训练

- 输入接口
                
```
        python
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_path", type=str, default="/data/flower_classifier/", help='')
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--trial_name", type=str, default="autotables-2503-train-1")
        parser.add_argument("--annotation_data", type=str, default="{'advance_settings':{}}")
```
-  输出目录结构
```
        python
        # 训练阶段结果产出的文件结构
        /app/tianji/
        ├───── client_train.py
        ├───── runs
        ├─────── models
        ├────────── {uuid}  # {uuid}自己生成，文件夹下存放产出模型文件等
        ├─────── metric
        ├────────── trial.txt  # 文件内的model_path为上述uuid的绝对路径
```
- metric 文件格式

```
        {
            "recom_model": "SimSiam",
            "gened_model_name": "SimSiam_autotables-2503-train-1",
            "metric_type": "acc",
            "metric_value": 0.6044,
            "used_models": "SimSiam",
            "model_path": "/app/tianji/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc",
            "model_params": {},
            "eval_result": {
                "SimSiam": {
                    "eval": {
                        "acc": 0.6044
                    }
                }
            },
            "predict_prob": {},
            "train_consuming": {
                "SimSiam": {
                    "total": 510.9792,
                    "train_time": 394.2867,
                    "valid_time": 116.6924
                }
            }
        }
```

- 指标结果

    输出格式为表格：
    
    | 数据集  | 指标          | 备注          |
    |:-----|:------------|:------------       |
    |      | `acc` | 分类准确率指标 |

### 离线验证

- 输入接口

     ```python
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_path", type=str, default="/data/flower_classifier/", help='')
        parser.add_argument("--model_path", type=str, default=r"/app/tianji/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc", help='')
        parser.add_argument("--used_model", type=str, default="SimSiam_autotables-2503-train-1", help='')
        parser.add_argument("--device", type=str, default='cuda')
       
     ```

- 输出目录结构

     ```python
        # 训练阶段结果产出的文件结构
        /app/tianji/
        ├───── client_test.py
        ├───── runs
        ├─────── models
        ├────────── {uuid}  # {uuid}自己生成，无需和训练的uuid一致
        ├───────────── evaluation_1649384314.json  # evaluation_{timestamp}.json 为{uuid}文件夹下的离线结果文件
        ├─────── metric
        ├────────── trial.txt  # 文件内的model_path为上述uuid的绝对路径
     ```

- metric文件格式

     ```
             {
            "model_path": "/app/tianji/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc",
            "predict_result_filename": "evaluation_1650953895.json",
            "preview_data": {},
            "test_result": {
                "SimSiam": {
                    "test": {
                        "acc": 0.6044
                    }
                }
            },
            "predict_prob": {},
            "test_consuming": {
                "SimSiam": {
                    "test": 2.2919
                }
            }
        }
        
    ```

### 推理

服务请求路由：/autotable/predict
服务端口: 8080

- 服务启动：

    ```
    
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, default=r"/app/tianji/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc")
        parser.add_argument("--used_model", type=str, default="SimSiam_autotables-2503-train-1", help='')
        parser.add_argument("--device", type=str, default='cuda')
        parser.add_argument("--service_port", type=int, default=8080)
        
    ```

- 请求示例：

     ```python
        post_data = {
            'content': image,'need_visual_data': True, 'need_visual_result': True
                    }
     ```

- 返回示例：

     ```python
     {
        'code': 200, 
        'msg': 'ok', 
        'data': [{
                'infer_data': ['daisy', '0.5575'], 
                'visual_data':
                            {'output_type': 'text', 
                            'annotation_type': None, 
                            'data': 'daisy_0.5575'},
                'visual_result': 'daisy_0.5575'
            
                }]
     }

     ```

- 推理示例
      
    输入：
    
    ![输入](10140303196_b88d3d6cec.jpg)
    
    输出：'daisy_0.5575'

 
<!--- 镜像及记录更新-->

<!--| 镜像名                                                      | 镜像id         | 日期       |-->
<!--|:---------------------------------------------------------|:-------------|:---------|-->
<!--| harbor.deepwisdomai.com/deepwisdom/gp_fine-grained-classification_pim:v1.0  |  9e055d78f419  | 20220415 |-->
