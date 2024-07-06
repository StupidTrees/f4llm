# Federated Learning for Large Language Model
Training LLMs with Deepspeed in Federated Learning

## Installation

### 目录组织  
我们建议按照如下的文件结构来使用`F4LLM`:    

    data:  存放数据集，注意该数据集需要已经经过 iid/niid 划分的联邦数据集;
    pretrained:  在nlp文件夹里存储从huggingface中下载的预训练模型，如`Llama-2-7b-hf` for `llama2-base`, `Llama-2-7b-chat-hf` for `llama2-chat`;
    code:  在该文件夹下面使用`fedllm`;
    output: 存储模型输出的记录等。

目录组织如下：
```grapha  
├── userhome  
│   └── data  
|   |   ├── fedni  
|   |   └── fedmedsi
│   ├── pretrained  
│   │   ├── nlp  
│   │   └── cv  
│   ├── output  
│   └── code  
│       └── fedllm  
```  
  
运行路径生成：  
```bash  
mkdir userhome  
cd userhome
mkdir data  
mkdir code  
mkdir pretrained  
cd pretrained  
mkdir nlp  
cd ..  
cd code  
```  
 
### 环境安装  
我们的主要环境: Python 3.10.12, torch2.1.0, cuda 12.2
```bash  
git clone https://github.com/iezhuozhuo/fedllm.git 
cd fedllm
pip install -r resquirements.txt  
```

## Usage
1.训练sh:
```bash
bash ./scripts/run_medsi.sh {your_file_path}/userhome {task_name} {algorithm} {used_gpu_id} {model_name}
```
例如:
```bash
bash ./scripts/run_medsi.sh {your_file_path}/userhome medsi fedavg 0,1,2,3 llama2-base
```

2.测试sh:
```bash
bash ./scripts/eval_medsi.sh {your_file_path}/userhome {task_name} {algorithm} {used_gpu_id} {model_name} {your_checkpoint_model_path_or_file}
```
例如:
```bash
bash ./scripts/eval_medsi.sh {your_file_path}/userhome medsi fedavg 0,1,2,3 llama2-base {your_checkpoint_model_path_or_file}
```

3.转换成待OpenAI GPT测试的sh:
```bash
bash ./scripts/gpt_eval.sh {your_model_predict_path_file} {task_name} {reference_engine}
```
例如:
```bash
bash ./scripts/gpt_eval.sh {your_model_predict_path_file} medsi gpt3.5
```

4.上面整体的pipeline:
```bash
bash ./scripts/pipeline.sh /userhome fedmedsi medsi fedavg 0,1,2,3 llama2-base
```
