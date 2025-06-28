import os
import time
import psutil
import json
import warnings
from typing import Dict, Union, List
from dotenv import load_dotenv
import torch
import openai
from openai import OpenAI
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM

# Suppress deprecation warnings from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning)


# Load environment variables
load_dotenv()


class LLMInferenceCalculator:
    def __init__(self):
        # Initialize API clients
        self._init_api_clients()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.models = {}
        self.tokenizers = {}

        # Default metrics for fallback
        self.default_metrics = {
            "tinyllama": {
                "base_latency_ms": 150,
                "tokens_per_second": 8,
                "memory_gb": 2,
            },
            "openai": {
                "base_latency_ms": 750,
                "tokens_per_second": 25,
                "memory_gb": 0,  # Cloud-based
            },
            "gemini": {
                "base_latency_ms": 800,
                "tokens_per_second": 22,
                "memory_gb": 0,  # Cloud-based
            },
        }

        # Initialize local models
        self._init_local_models()

    def _init_api_clients(self):
        """Initialize API clients for OpenAI and Gemini."""
        # Initialize OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")

        if not openai_key:
            print("Warning: OPENAI_API_KEY environment variable is not set")
            self.openai_client = None
        else:
            try:
                self.openai_client = OpenAI(api_key=openai_key)
                print("Successfully initialized OpenAI client")
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
                self.openai_client = None

        # Initialize Gemini
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            print("Warning: GOOGLE_API_KEY environment variable is not set")
            self.gemini_client = None
        else:
            try:
                genai.configure(api_key=gemini_key)
                self.gemini_client = genai.GenerativeModel("gemini-1.5-flash")
                print("Successfully initialized Gemini client")
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini client: {e}")
                self.gemini_client = None

    def _init_local_models(self):
        """Initialize local models and tokenizers."""
        model_paths = {"tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}

        for model_name, model_path in model_paths.items():
            try:
                print(f"Loading {model_name} model and tokenizer...")
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
                self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=(
                        torch.float16 if self.device == "cuda" else torch.float32
                    ),
                    device_map="auto",
                )
                print(f"Successfully loaded {model_name}")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                self.models[model_name] = None
                self.tokenizers[model_name] = None

    def _benchmark_local_model(
        self, model: str, sample_text: str, num_runs: int
    ) -> Dict[str, float]:
        """Benchmark local models (TinyLlama, Mistral)."""
        if not self.models.get(model) or not self.tokenizers.get(model):
            print(f"Warning: Model {model} not initialized, using default metrics")
            return {
                "avg_latency_ms": self.default_metrics[model]["base_latency_ms"],
                "avg_memory_usage_mb": self.default_metrics[model]["memory_gb"]
                * 1024,  # Convert GB to MB
                "avg_tokens_per_second": self.default_metrics[model][
                    "tokens_per_second"
                ],
                "std_latency_ms": 0,
                "std_tokens_per_second": 0,
            }

        try:
            tokenizer = self.tokenizers[model]
            model_instance = self.models[model]

            # Track metrics
            latencies = []
            memory_usages = []
            tokens_per_second = []

            input_ids = tokenizer.encode(sample_text, return_tensors="pt").to(
                self.device
            )
            input_token_count = len(input_ids[0])

            # Warm-up run
            _ = model_instance.generate(input_ids, max_new_tokens=50)

            for _ in range(num_runs):
                # Memory before
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                # Time the inference
                start_time = time.time()
                output_ids = model_instance.generate(input_ids, max_new_tokens=50)
                end_time = time.time()

                # Calculate metrics
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                output_token_count = len(output_ids[0]) - input_token_count
                inference_time = end_time - start_time

                latencies.append(inference_time * 1000)  # Convert to ms
                memory_usages.append(memory_after - memory_before)
                tokens_per_second.append(output_token_count / inference_time)

            return {
                "avg_latency_ms": sum(latencies) / len(latencies),
                "avg_memory_usage_mb": sum(memory_usages) / len(memory_usages),
                "avg_tokens_per_second": sum(tokens_per_second)
                / len(tokens_per_second),
                "std_latency_ms": self._calculate_std(latencies),
                "std_tokens_per_second": self._calculate_std(tokens_per_second),
            }
        except Exception as e:
            print(f"Warning: Error benchmarking {model}: {e}")
            return {
                "avg_latency_ms": self.default_metrics[model]["base_latency_ms"],
                "avg_memory_usage_mb": self.default_metrics[model]["memory_gb"]
                * 1024,  # Convert GB to MB
                "avg_tokens_per_second": self.default_metrics[model][
                    "tokens_per_second"
                ],
                "std_latency_ms": 0,
                "std_tokens_per_second": 0,
            }

    def _benchmark_openai(self, sample_text: str, num_runs: int) -> Dict[str, float]:

        default_metrics = {
            "avg_latency_ms": self.default_metrics["openai"]["base_latency_ms"],
            "avg_memory_usage_mb": 0.0,
            "avg_tokens_per_second": self.default_metrics["openai"]["tokens_per_second"],
            "std_latency_ms": 0.0,
            "std_tokens_per_second": 0.0,
        }

        if not self.openai_client:
            print("Warning: OpenAI client not initialized")
            return default_metrics

        latencies = []
        tokens_per_second = []

        for _ in range(num_runs):
            try:
                start_time = time.time()
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": sample_text}],
                    max_tokens=100
                )
                end_time = time.time()

                output_tokens = len(response.choices[0].message.content.split())
                inference_time = end_time - start_time

                latencies.append(inference_time * 1000)  # in ms
                tokens_per_second.append(output_tokens / inference_time)
            except Exception as e:
                print(f"Warning: OpenAI API call failed: {e}")
                return default_metrics

        return {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "avg_memory_usage_mb": 0.0,
            "avg_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
            "std_latency_ms": self._calculate_std(latencies),
            "std_tokens_per_second": self._calculate_std(tokens_per_second),
        }


    def _benchmark_gemini(self, sample_text: str, num_runs: int) -> Dict[str, float]:
        """Benchmark Gemini API."""
        default_metrics = {
            "avg_latency_ms": self.default_metrics["gemini"]["base_latency_ms"],
            "avg_memory_usage_mb": 0.0,
            "avg_tokens_per_second": self.default_metrics["gemini"][
                "tokens_per_second"
            ],
            "std_latency_ms": 0.0,
            "std_tokens_per_second": 0.0,
        }

        if not self.gemini_client:
            print("Warning: Gemini client not initialized, using default metrics")
            return default_metrics

        latencies = []
        tokens_per_second = []

        for _ in range(num_runs):
            try:
                start_time = time.time()
                response = self.gemini_client.generate_content(sample_text)
                end_time = time.time()

                inference_time = end_time - start_time
                # Estimate tokens as Gemini doesn't provide token count
                output_tokens = (
                    len(response.text.split()) * 1.3
                )  # Rough estimate: words * 1.3

                latencies.append(inference_time * 1000)  # ms
                tokens_per_second.append(output_tokens / inference_time)
            except Exception as e:
                print(f"Warning: Gemini API call failed: {e}")
                return default_metrics

        if not latencies:  # If no successful runs
            return default_metrics

        return {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "avg_memory_usage_mb": 0.0,
            "avg_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
            "std_latency_ms": self._calculate_std(latencies),
            "std_tokens_per_second": self._calculate_std(tokens_per_second),
        }

    def benchmark_model(
        self,
        model: str,
        sample_text: str = "This is a sample text for benchmarking the model's performance.",
        num_runs: int = 5,
    ) -> Dict[str, float]:
        """Perform real-time benchmarking of the model."""
        print(f"\nBenchmarking {model}...")

        if model == "openai":
            return self._benchmark_openai(sample_text, num_runs)
        elif model == "gemini":
            return self._benchmark_gemini(sample_text, num_runs)
        else:
            return self._benchmark_local_model(model, sample_text, num_runs)

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return (squared_diff_sum / len(values)) ** 0.5

    def calculate_metrics(
        self,
        model: str,
        tokens: int,
        batch_size: int,
        hardware_type: str,
        deployment_mode: str,
        benchmark_results: Dict[str, float] = None,
    ) -> Dict[str, Union[float, Dict[str, str]]]:
        """Calculate all inference metrics using benchmark results if available."""
        if model not in self.default_metrics:
            raise ValueError(f"Unsupported model: {model}")

        if not benchmark_results:
            # Use default values if no benchmark results available
            return self._calculate_metrics_from_defaults(
                model, tokens, batch_size, hardware_type
            )

        # Calculate metrics using benchmark results
        tokens_per_second = benchmark_results["avg_tokens_per_second"]
        base_latency = benchmark_results["avg_latency_ms"]

        # Calculate total latency including base latency and token processing time
        token_processing_time = (tokens / tokens_per_second) * 1000  # Convert to ms
        latency = base_latency + token_processing_time

        memory = benchmark_results["avg_memory_usage_mb"] / 1024  # Convert to GB

        # Adjust for batch size
        if batch_size > 1:
            latency = (latency * batch_size) * 0.8  # 20% efficiency gain
            memory *= batch_size

        # Adjust for GPU if applicable
        if hardware_type == "gpu" and deployment_mode == "local":
            latency *= 0.3  # GPU provides roughly 70% speedup
            memory *= 1.2  # GPU needs slightly more memory

        return {
            "latency_ms": round(latency, 2),
            "memory_usage_gb": round(memory, 2),
            "tokens_per_second": round(tokens_per_second, 2),
            "std_latency_ms": round(benchmark_results["std_latency_ms"], 2),
            "hardware_requirements": self._get_hardware_requirements(model, memory),
        }

    def _calculate_metrics_from_defaults(
        self, model: str, tokens: int, batch_size: int, hardware_type: str
    ) -> Dict[str, Union[float, Dict[str, str]]]:
        """Calculate metrics using default values when benchmark results are not available."""
        if model not in self.default_metrics:
            raise ValueError(f"Unsupported model: {model}")

        metrics = self.default_metrics[model]

        # Calculate total latency including base latency and token processing time
        token_processing_time = (
            tokens / metrics["tokens_per_second"]
        ) * 1000  # Convert to ms
        latency = metrics["base_latency_ms"] + token_processing_time

        # Calculate memory
        memory = metrics["memory_gb"]
        if model != "openai":
            memory *= batch_size * 1.2  # Account for batch size

        # Adjust for GPU
        if hardware_type == "gpu" and model != "openai":
            latency *= 0.3  # GPU provides roughly 70% speedup
            memory *= 1.2  # GPU needs slightly more memory

        return {
            "latency_ms": round(latency, 2),
            "memory_usage_gb": round(memory, 2),
            "tokens_per_second": metrics["tokens_per_second"],
            "std_latency_ms": 0,  # No standard deviation for default values
            "hardware_requirements": self._get_hardware_requirements(model, memory),
        }

    def _get_hardware_requirements(
        self, model: str, memory_usage_gb: float
    ) -> Dict[str, str]:
        """Get hardware requirements based on actual memory usage."""
        if model == "openai":
            return {
                "min_ram": "N/A (Cloud-based)",
                "min_cpu": "N/A (Cloud-based)",
                "recommended_gpu": "N/A (Cloud-based)",
            }

        return {
            "min_ram": f"{max(4, int(memory_usage_gb * 1.5))}GB RAM",
            "min_cpu": "4+ CPU cores",
            "recommended_gpu": "4GB+ VRAM (optional)",
        }


def main():
    # Example usage with benchmarking
    calculator = LLMInferenceCalculator()

    # Sample configurations for benchmarking
    sample_text = "Explain the concept of machine learning in simple terms."
    models_to_test = ["tinyllama", "openai", "gemini"]

    for model in models_to_test:
        try:
            # Run benchmarks
            benchmark_results = calculator.benchmark_model(
                model=model,
                sample_text=sample_text,
                num_runs=3,  # Reduced for example purposes
            )
            print(f"\nBenchmark results for {model}:")
            print(json.dumps(benchmark_results, indent=2))

            # Calculate metrics using benchmark results
            config = {
                "model": model,
                "tokens": 1000,
                "batch_size": 1,
                "hardware_type": "cpu",
                "deployment_mode": "local" if model == "tinyllama" else "api",
            }

            metrics = calculator.calculate_metrics(
                **config, benchmark_results=benchmark_results
            )
            print(f"\nCalculated metrics for {model}:")
            print(json.dumps(metrics, indent=2))

        except Exception as e:
            print(f"Error processing {model}: {e}")


if __name__ == "__main__":
    main()
