import abc
import torch
import time
import logging
import json
import datetime
from dataclasses import asdict
from quantizer import Quantizer
from qa_dataset import QADataset
from models import CausalLM, Tokenizer
from functools import cached_property, cache
from evaluator import Evaluator, EvaluationResult
from multiprocessing import queues, Queue, Lock, Process
from accelerate import init_empty_weights, infer_auto_device_map
from config import version, cache_file, hf_cache_dir, device_configs
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import os


class Experiment(abc.ABC):
    def __init__(self, model_name: str, dataset_name: str, dtype: torch.dtype, question_count: int, parallel: bool, verbose: bool):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dtype = dtype
        self.question_count = question_count
        self.verbose = verbose
        self.parallel = parallel and len(self.quantizer_list) > 1 and len(device_configs) > 1

    @cached_property
    def tokenizer(self) -> Tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=hf_cache_dir)
        tokenizer.pad_token_id = 0
        return tokenizer

    @cache
    def get_model(self, worker_id: int) -> CausalLM:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name, cache_dir=hf_cache_dir))
        _, max_memory = device_configs[worker_id]
        model.tie_weights()
        device_map = infer_auto_device_map(model, max_memory=max_memory, dtype=self.dtype, no_split_module_classes=model._no_split_modules)
        if any(x == "cpu" or x == "disk" for x in device_map.values()):
            print("Warning: CPU offloading enabled!")
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=device_map, torch_dtype=self.dtype, cache_dir=hf_cache_dir).eval()
        return model

    @cached_property
    def datasets(self) -> QADataset:
        return QADataset(self.dataset_name, self.tokenizer, self.question_count)

    @cached_property
    def quantizer_list(self) -> list[tuple[Quantizer, Quantizer]]:
        return []

    @abc.abstractmethod
    def process_result(self, results: list[EvaluationResult]):
        pass

    def _run_single_evaluation(self, worker_id: int, task_queue: Queue, file_lock: Lock):
        idx, key_quantizer, value_quantizer = task_queue.get(timeout=1)
        print(f"Running evaluation #{idx+1} on worker #{worker_id+1}...")
        device, _ = device_configs[worker_id]
        key_quantizer.set_dtype_and_device(self.dtype, device)
        value_quantizer.set_dtype_and_device(self.dtype, device)
        evaluator = Evaluator(device, version, self.model_name, self.datasets, key_quantizer, value_quantizer)
        with file_lock:
            result = evaluator.get_cached_result(cache_file)
        if result is None:
            model = self.get_model(worker_id)
            logging.info(f"[Worker {worker_id+1}] Evaluating #{idx+1} (Not cached)...")
            start_time_eval = time.time() # Record evaluation start time
            result = evaluator.evaluate(model, use_tqdm=True)
            end_time_eval = time.time() # Record evaluation end time
            eval_duration = end_time_eval - start_time_eval
            logging.info(f"[Worker {worker_id+1}] Evaluation #{idx+1} finished in {eval_duration:.2f} seconds.")
            # Check for timeout
            if eval_duration > 300: # 300 seconds = 5 minutes
                 logging.error(f"[Worker {worker_id+1}] Evaluation #{idx+1} EXCEEDED TIMEOUT ({eval_duration:.2f}s > 300s)! Check configuration or system.")
                 # NOTE: This does not terminate or retry the task automatically.

            # MODIFICATION START: Ensure cache directory exists before writing
            if cache_file:
                try:
                    cache_dir = os.path.dirname(cache_file)
                    if cache_dir: # Avoid trying to create empty dir if cache_file is just a filename
                        os.makedirs(cache_dir, exist_ok=True)
                except OSError as e:
                    print(f"Warning: Could not create cache directory '{cache_dir}': {e}")
            # MODIFICATION END
            with file_lock:
                evaluator.cache_result(cache_file, result)
        if self.verbose:
            print(f"  Params: {evaluator.params}")
            print(f"  Results: {asdict(result)}")
            print("======================================")

        # --- Save individual result immediately --- 
        raw_dir = "experiments/raw"
        # Ensure raw directory exists (might be redundant if created elsewhere, but safe)
        # No need for lock here, makedirs handles existing dirs fine.
        os.makedirs(raw_dir, exist_ok=True) 

        raw_data = {
            "key_quantizer_params": key_quantizer.params,
            "value_quantizer_params": value_quantizer.params,
            "evaluation_result": asdict(result)
        }
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Include worker_id and idx for uniqueness
        raw_filename = os.path.join(raw_dir, f"result_w{worker_id+1}_idx{idx}_{timestamp_str}.json")

        try:
            # Use the file_lock to ensure safe writing from multiple processes
            with file_lock:
                 with open(raw_filename, 'w') as f:
                     json.dump(raw_data, f, indent=4)
            logging.info(f"[Worker {worker_id+1}] Saved result #{idx+1} to {raw_filename}")
        except IOError as e:
             logging.error(f"[Worker {worker_id+1}] Error saving result #{idx+1} to {raw_filename}: {e}")
        # --- End saving individual result ---

        end_time_task = time.time() # Record task end time
        task_duration = end_time_task - start_time_task

    def run(self):
        file_lock = Lock()
        task_queue = Queue()
        for idx, (key_quantizer, value_quantizer) in enumerate(self.quantizer_list):
            task_queue.put((idx, key_quantizer, value_quantizer))
        def worker(worker_id: int):
            while True:
                try:
                    self._run_single_evaluation(worker_id, task_queue, file_lock)
                except queues.Empty:
                    break

        if self.parallel:
            _, _ = self.datasets.questions, self.tokenizer
            process_list: list[Process] = []
            for worker_id in range(len(device_configs)):
                process = Process(target=worker, args=(worker_id,))
                process_list.append(process)
                process.start()
            for process in process_list:
                process.join()
        else:
            worker(0)
            
        # Remove the final result collection loop, as results are saved individually by workers
        # results: list[EvaluationResult] = []
        # for key_quantizer, value_quantizer in self.quantizer_list:
        #     evaluator = Evaluator("cpu", version, self.model_name, self.datasets, key_quantizer, value_quantizer)
        #     result = evaluator.get_cached_result(cache_file)
        #     assert result is not None
        #     results.append(result)
        
        # Call process_result (which might need adaptation in subclasses)
        # Pass an empty list or None, as results are no longer collected here.
        self.process_result([]) # Or self.process_result(None) depending on subclass adaptation
