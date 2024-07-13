#!/bin/bash
# shellcheck disable=SC2068
# read parameters
idx=0
for i in $@
do
  args[${idx}]=$i
  let "idx=${idx}+1"
done

# split parameters
run_dirs=${args[0]}
project_name=${args[1]}
model_type=${args[2]}
algorithm=${args[3]}
task_name=${args[4]}
port=${args[5]}
device=${args[6]}


if [ "$model_type" = "llama2-base" ]; then
    model_name_or_path=/userhome/pretrain/nlp/Llama-2-7b-hf/
elif [ "$model_type" = "tinyllama" ]; then
    model_name_or_path=/userhome/pretrain/nlp/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T/
elif [ "$model_type" = "qwen" ]; then
    model_name_or_path=/userhome/pretrain/nlp/Qwen-1_8B/
elif [ "$model_type" = "baichuan2-base" ]; then
    model_name_or_path=/userhome/pretrain/nlp/Baichuan2-7B-Base/
else
    echo "Unknown model_type"
    model_name_or_path=""
fi

# example: bash ./scripts/run_all.sh /userhome fed4llm llama2-base fedavg ultrafeedback_binarized 10001 0
deepspeed --include localhost:3 --master_port 10001 main.py \
--do_train \
--raw_dataset_path /userhome/data/${project_name}/${task_name}_data.pkl \
--partition_dataset_path /userhome/data/${project_name}/${task_name}_partition.pkl \
--model_name_or_path ${model_name_or_path} \
--model_type ${model_type} \
--output_dir ${run_dirs}/output/${project_name}/ \
--task_name ${task_name} \
--fl_algorithm ${algorithm} \
--config_path yamls/${task_name}_${algorithm}.yaml \
--data_name ${task_name} \
--role server \
--num_sub 2 \
--server_ip 127.0.0.1 \
--server_port 15001 &

deepspeed --include localhost:0 --master_port 10002 main.py \
--do_train \
--raw_dataset_path /userhome/data/${project_name}/${task_name}_data.pkl \
--partition_dataset_path /userhome/data/${project_name}/${task_name}_partition.pkl \
--model_name_or_path ${model_name_or_path} \
--model_type ${model_type} \
--output_dir ${run_dirs}/output/${project_name}/ \
--task_name ${task_name} \
--fl_algorithm ${algorithm} \
--config_path yamls/${task_name}_${algorithm}.yaml \
--data_name ${task_name} \
--role client \
--client_name 0 \
--client_ip 127.0.0.1 \
--client_port 15002 &

deepspeed --include localhost:1 --master_port 10003 main.py \
--do_train \
--raw_dataset_path /userhome/data/${project_name}/${task_name}_data.pkl \
--partition_dataset_path /userhome/data/${project_name}/${task_name}_partition.pkl \
--model_name_or_path ${model_name_or_path} \
--model_type ${model_type} \
--output_dir ${run_dirs}/output/${project_name}/ \
--task_name ${task_name} \
--fl_algorithm ${algorithm} \
--config_path yamls/${task_name}_${algorithm}.yaml \
--data_name ${task_name} \
--role client \
--client_name 1 \
--client_ip 127.0.0.1 \
--client_port 15003