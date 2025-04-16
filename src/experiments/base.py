import abc
import torch
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
        # --- MODIFICATION START: Build max_memory map for ALL configured GPUs ---
        full_max_memory = {}
        # Slightly reduce available memory per GPU to leave headroom for activations/OS etc.
        # Adjust "23GB" if needed based on observation.
        available_gb_per_gpu = "23GB"
        for i in range(len(device_configs)):
            full_max_memory[i] = available_gb_per_gpu
        # Keep large CPU memory for potential offloading
        full_max_memory["cpu"] = "400GB" # Or get from config if it varies
        print(f"Worker {worker_id}: Inferring device map with max_memory: {full_max_memory}")
        # --- MODIFICATION END ---

        with init_empty_weights():
            # Load config first to avoid downloading full model if only config needed
            config = AutoConfig.from_pretrained(self.model_name, cache_dir=hf_cache_dir, trust_remote_code=True) # Added trust_remote_code
            model = AutoModelForCausalLM.from_config(config)

        # Use the full memory map for inference
        # No need to use device_configs[worker_id] here anymore for max_memory
        model.tie_weights()
        device_map = infer_auto_device_map(
            model,
            max_memory=full_max_memory, # Use the map with all GPUs
            dtype=self.dtype,
            no_split_module_classes=model._no_split_modules
        )
        # --- Optional: Force CPU offload if map is empty or only has CPU ---
        if not device_map or all(v in ["cpu", "disk"] for v in device_map.values()):
             print(f"Warning: infer_auto_device_map resulted in CPU/Disk-only map: {device_map}. Forcing CPU offload map.")
             # Fallback to a map that explicitly offloads to CPU if inference failed
             device_map = infer_auto_device_map(
                  model, max_memory={k:v for k,v in full_max_memory.items() if k == 'cpu'}, # Only allow CPU memory
                  dtype=self.dtype, no_split_module_classes=model._no_split_modules, verbose=True
             )

        print(f"Worker {worker_id}: Using inferred device_map: {device_map}")

        if any(x == "cpu" or x == "disk" for x in device_map.values()):
            print(f"Worker {worker_id}: Warning: CPU/Disk offloading enabled in device map.")
        # Load the actual pretrained model with the calculated map
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map, # Use the calculated map
            torch_dtype=self.dtype,
            cache_dir=hf_cache_dir,
            trust_remote_code=True, # Added trust_remote_code
            # attn_implementation="eager" # Keep if added previously
        ).eval()
        # It's generally okay for all workers to load the model using the same multi-GPU map.
        # Accelerate handles the underlying distribution. The worker's primary device
        # (from device_configs[worker_id][0]) is used elsewhere for placing inputs/outputs.
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
            result = evaluator.evaluate(model, use_tqdm=True)
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
        results: list[EvaluationResult] = []
        for key_quantizer, value_quantizer in self.quantizer_list:
            evaluator = Evaluator("cpu", version, self.model_name, self.datasets, key_quantizer, value_quantizer)
            result = evaluator.get_cached_result(cache_file)
            assert result is not None
            results.append(result)
        self.process_result(results)
