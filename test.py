#
import sys
import time

from setuptools.glob import glob

from utils.general import run_process

#
# def hello():
#     for i in range(5):
#         time.sleep(10)
#         print("hello")
#
#
# hello()
# print("sub done")


import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class DirectoryHandler(FileSystemEventHandler):
    def __init__(self, eval_count, base_opts, device, port):
        self.base_opts = base_opts
        self.eval_count = eval_count
        self.evaluated_models = 0

    def on_created(self, event):
        if event.is_directory:
            print(f"检测到新文件夹: {event.src_path}")
            self.evaluate_directory(event.src_path)
            self.evaluated_models += 1
            if self.evaluated_models >= self.eval_count:
                print("达到评测次数，停止监控")
                observer.stop()

    def evaluate_directory(self, directory_path):
        print(f"评测模型: {directory_path} 开始")
        time.sleep(300)
        print(f"评测模型: {directory_path} 完成")
        # run pipeline

        # options = [
        #     "--raw_dataset_path", f"{args.run_dirs}/data/{args.project_name}/{args.task_name}_data.pkl",
        #     "--partition_dataset_path", f"{args.run_dirs}/data/{args.project_name}/{args.task_name}_partition.pkl",
        #     "--model_name_or_path", f"{model_path}",
        #     "--output_dir", f"{args.run_dirs}/output/{args.project_name}/",
        #     "--model_type", f"{args.model_type}",
        #     "--task_name", f"{args.task_name}",
        #     "--fl_algorithm", f"{args.fl_algorithm}",
        #     "--config_path", f"./yamls/{args.task_name}_{args.fl_algorithm}.yaml",
        #     "--model_type", f"{args.model_type}",
        #     # "--do_grid", f"{args.do_grid}",
        # ]
        # ops = " ".join(options)

        # train_ckp_dir = f"{args.run_dirs}/output/{args.project_name}/{args.task_name}/" \
        #                 f"{args.fl_algorithm.lower()}/{times}_{args.model_type}_lora/"
        # pred_cmds = f"deepspeed --include localhost:{args.device} --master_port {args.port} " \
        #             f"main.py --do_eval --times {times} --checkpoint_file {train_ckp_dir} "
        # pred_cmds += ops


if __name__ == "__main__":
    path = "/userhome/output/f4llm/ultrafeedback_binarized/fedavg/20240724165202_tinyllama_lora/"
    eval_count = 5  # 评测的模型文件夹数量
    print(f"eval {path}")
    event_handler = DirectoryHandler(eval_count=eval_count)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        while observer.is_alive():
            observer.join(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()