#!/usr/bin/env python3
"""
Persistent Interactive Inference Server

Keeps the compiled pipeline loaded in memory to avoid recompilation.
Useful for multiple inference runs without restart overhead.

The key insight: torch.compile graphs exist only in memory. Once you exit Python,
they're gone and need to be recompiled. This server keeps the Python process 
running so compiled graphs stay cached.
"""

import argparse
import os
import sys
import json
import time
from typing import List

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
from utils.misc import set_seed

class PersistentInteractivePipeline:
    def __init__(self, config_path: str, cli_args=None):
        """Initialize and compile the pipeline once"""
        print("="*60)
        print("INITIALIZING PERSISTENT INTERACTIVE PIPELINE")
        print("="*60)
        
        # Load config
        self.config = OmegaConf.load(config_path)
        
        # Override quantization config with CLI args if provided
        if cli_args:
            # Check both new and legacy flags
            enable_quant = cli_args.quantize or cli_args.fp8
            disable_quant = cli_args.no_quantize or cli_args.no_fp8
            
            if enable_quant:
                if not hasattr(self.config, 'quantization'):
                    self.config.quantization = {}
                self.config.quantization.enabled = True
                flag_used = "--quantize" if cli_args.quantize else "--fp8"
                print(f"Quantization enabled via {flag_used} flag")
            elif disable_quant:
                if hasattr(self.config, 'quantization'):
                    self.config.quantization.enabled = False
                flag_used = "--no-quantize" if cli_args.no_quantize else "--no-fp8"
                print(f"Quantization disabled via {flag_used} flag")
        
        # Setup device
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.local_rank)
        else:
            self.local_rank = 0
            self.device = torch.device("cuda")
            
        set_seed(self.config.seed + self.local_rank)
        torch.set_grad_enabled(False)
        
        # Initialize pipeline (this will compile text encoder and VAE)
        init_start = time.perf_counter()
        print(f"Loading and compiling models on device {self.device}...")
        
        self.pipeline = InteractiveCausalInferencePipeline(self.config, device=self.device)
        
        # Load weights (same as original script)
        if self.config.generator_ckpt:
            self._load_generator_weights()
            
        if getattr(self.config, "adapter", None):
            self._setup_lora()
            
        # Move to device and dtype first
        print("dtype", self.pipeline.generator.model.dtype)
        self.pipeline = self.pipeline.to(dtype=torch.bfloat16)
        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)
        
        # Apply quantization AFTER moving to device
        if getattr(self.config, "quantization", None) and self.config.quantization.get("enabled", False):
            print(f"[DEBUG] Quantization config found: {self.config.quantization}")
            self._apply_quantization()
        else:
            print(f"[DEBUG] No quantization config or disabled. Has quantization attr: {hasattr(self.config, 'quantization')}")
            if hasattr(self.config, 'quantization'):
                print(f"[DEBUG] Quantization config: {self.config.quantization}")
        
        # Debug: Print model dtypes if quantization was attempted
        if getattr(self.config, "quantization", None) and self.config.quantization.get("enabled", False):
            self._print_model_dtypes()
        
        init_time = (time.perf_counter() - init_start) * 1000
        print(f"Pipeline initialization completed: {init_time:.2f} ms")
        print("="*60)
        print("NOTE: First inference will include compilation time (~30s).")
        print("Subsequent inferences will be much faster (~60s vs ~90s)!")
        if getattr(self.config, "quantization", None) and self.config.quantization.get("enabled", False):
            quant_type = self.config.quantization.get("dtype", "unknown")
            print(f"{quant_type.upper()} quantization enabled - expect additional speedup!")
        print("="*60)
        
    def _load_generator_weights(self):
        """Load generator checkpoint"""
        state_dict = torch.load(self.config.generator_ckpt, map_location="cpu")
        raw_gen_state_dict = state_dict["generator_ema" if self.config.use_ema else "generator"]

        if self.config.use_ema:
            def _clean_key(name: str) -> str:
                return name.replace("_fsdp_wrapped_module.", "")
            cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
            self.pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        else:
            self.pipeline.generator.load_state_dict(raw_gen_state_dict)
            
    def _setup_lora(self):
        """Setup LoRA if configured"""
        from utils.lora_utils import configure_lora_for_model
        import peft
        
        print(f"LoRA enabled with config: {self.config.adapter}")
        self.pipeline.generator.model = configure_lora_for_model(
            self.pipeline.generator.model,
            model_name="generator", 
            lora_config=self.config.adapter,
            is_main_process=(self.local_rank == 0),
        )
        
        lora_ckpt_path = getattr(self.config, "lora_ckpt", None)
        if lora_ckpt_path:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(self.pipeline.generator.model, lora_checkpoint["generator_lora"])
            else:
                peft.set_peft_model_state_dict(self.pipeline.generator.model, lora_checkpoint)
            print("LoRA weights loaded")
            
        self.pipeline.is_lora_enabled = True
        
    def _apply_quantization(self):
        """Apply INT8 quantization to specified models"""
        quant_config = self.config.quantization
        quant_dtype = quant_config.get("dtype", "int8")
        quant_method = quant_config.get("method", "dynamic")
        models_to_quantize = quant_config.get("models", [])
        
        print(f"Applying {quant_dtype} quantization (method: {quant_method}) to models: {models_to_quantize}")
        
        if quant_dtype == "int8":
            self._apply_int8_quantization(models_to_quantize, quant_method)
        elif quant_dtype in ["float8_e4m3fn", "float8_e5m2"]:
            print("Warning: FP8 quantization not implemented in this version")
            print("Use dtype: 'int8' for stable quantization")
            raise RuntimeError("FP8 quantization not supported - use INT8")
        else:
            raise RuntimeError(f"Unsupported quantization dtype: {quant_dtype}")
            
    def _apply_int8_quantization(self, models_to_quantize: list, method: str):
        """Apply INT8 quantization using PyTorch's quantization APIs"""
        import torch.quantization as quant
            
        try:
            # Quantize VAE if requested
            if "vae" in models_to_quantize:
                print("Quantizing VAE to INT8...")
                quant_start = time.perf_counter()
                
                if method == "dynamic":
                    # Dynamic quantization (no calibration needed)
                    self.pipeline.vae = quant.quantize_dynamic(
                        self.pipeline.vae, 
                        {torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear}, 
                        dtype=torch.qint8
                    )
                else:
                    print(f"Static quantization method '{method}' not implemented yet")
                    print("Using dynamic quantization instead")
                    self.pipeline.vae = quant.quantize_dynamic(
                        self.pipeline.vae, 
                        {torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear}, 
                        dtype=torch.qint8
                    )
                
                quant_time = (time.perf_counter() - quant_start) * 1000
                print(f"VAE quantization completed: {quant_time:.2f} ms")
                print(f"  - Quantized Conv2d, Conv3d, and Linear layers to INT8")
                
            # Quantize text encoder if requested  
            if "text_encoder" in models_to_quantize:
                print("Quantizing text encoder to INT8...")
                quant_start = time.perf_counter()
                
                self.pipeline.text_encoder = quant.quantize_dynamic(
                    self.pipeline.text_encoder,
                    {torch.nn.Linear, torch.nn.Embedding},
                    dtype=torch.qint8
                )
                        
                quant_time = (time.perf_counter() - quant_start) * 1000
                print(f"Text encoder quantization completed: {quant_time:.2f} ms")
            
            # Quantize generator if requested
            if "generator" in models_to_quantize:
                print("Quantizing generator to INT8...")
                quant_start = time.perf_counter()
                
                # Note: Be careful with LoRA + quantization
                if hasattr(self.pipeline, 'is_lora_enabled') and self.pipeline.is_lora_enabled:
                    print("Warning: Quantizing generator with LoRA may affect adapter performance")
                
                self.pipeline.generator = quant.quantize_dynamic(
                    self.pipeline.generator,
                    {torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear},
                    dtype=torch.qint8
                )
                        
                quant_time = (time.perf_counter() - quant_start) * 1000
                print(f"Generator quantization completed: {quant_time:.2f} ms")
                
        except Exception as e:
            print(f"INT8 quantization failed: {e}")
            raise RuntimeError(f"Quantization failed: {e}") from e
            
    def _manual_fp8_conversion(self, model: torch.nn.Module, target_dtype: torch.dtype):
        """Manual FP8 conversion ensuring all tensors have consistent dtype"""
        total_params = 0
        quantized_params = 0
        total_buffers = 0 
        quantized_buffers = 0
        
        # Convert all parameters
        for name, param in model.named_parameters():
            total_params += 1
            if param.dtype in (torch.bfloat16, torch.float32, torch.float16):
                param.data = param.data.to(target_dtype)
                quantized_params += 1
                
        # Convert all buffers (including mean/std we registered)
        for name, buffer in model.named_buffers():
            total_buffers += 1
            if buffer.dtype in (torch.bfloat16, torch.float32, torch.float16):
                buffer.data = buffer.data.to(target_dtype)
                quantized_buffers += 1
                
        print(f"  - Quantized {quantized_params}/{total_params} parameters")
        print(f"  - Quantized {quantized_buffers}/{total_buffers} buffers")
        
    def _validate_quantization(self, model_name: str, model: torch.nn.Module, target_dtype: torch.dtype):
        """Validate that quantization was applied correctly"""
        fp8_params = 0
        total_params = 0
        fp8_buffers = 0
        total_buffers = 0
        
        # Check parameters
        for name, param in model.named_parameters():
            total_params += 1
            if param.dtype == target_dtype:
                fp8_params += 1
                
        # Check buffers  
        for name, buffer in model.named_buffers():
            total_buffers += 1
            if buffer.dtype == target_dtype:
                fp8_buffers += 1
                
        print(f"  - Validation: {fp8_params}/{total_params} parameters in FP8")
        print(f"  - Validation: {fp8_buffers}/{total_buffers} buffers in FP8")
        
        if fp8_params == 0 and fp8_buffers == 0:
            print(f"  ⚠️  WARNING: No {model_name} tensors were quantized to FP8!")
        else:
            print(f"  ✅ {model_name} quantization successful")
            
    def _print_model_dtypes(self):
        """Debug function to print model data types and quantization status"""
        print("\n" + "="*60)
        print("MODEL QUANTIZATION STATUS DEBUG")
        print("="*60)
        
        # Check for quantized modules (dynamic quantization)
        quantized_modules = []
        for name, module in self.pipeline.vae.named_modules():
            if hasattr(module, '_is_quantized_layer') or 'quantized' in str(type(module)).lower():
                quantized_modules.append((name, type(module).__name__))
                
        if quantized_modules:
            print(f"✅ Found {len(quantized_modules)} quantized modules:")
            for name, module_type in quantized_modules[:5]:  # Show first 5
                print(f"  {name}: {module_type}")
            if len(quantized_modules) > 5:
                print(f"  ... and {len(quantized_modules) - 5} more")
        else:
            print("❌ No quantized modules detected")
            
        # For dynamic quantization, weights stay in original dtype but modules are wrapped
        print(f"\nVAE model type: {type(self.pipeline.vae).__name__}")
        
        # Show a few parameter dtypes for reference
        print(f"\nSample parameter dtypes (may stay original for dynamic quantization):")
        for name, param in list(self.pipeline.vae.named_parameters())[:3]:  # First 3 only
            print(f"  {name}: {param.dtype}")
            
        print("="*60)
        
    def run_inference(self, prompts_list: List[List[str]], switch_frame_indices: List[int], 
                     output_path: str = None, enable_timing: bool = True):
        """Run inference with pre-compiled models (no compilation overhead after first run)"""
        
        print(f"\nRunning inference with {len(prompts_list)} segments...")
        print(f"Switch points: {switch_frame_indices}")
        
        # Clear VAE cache to prevent tensor size mismatches between runs
        # This is crucial for torch.compile stability across different input sizes
        if hasattr(self.pipeline.vae, 'model') and hasattr(self.pipeline.vae.model, 'clear_cache'):
            print("Clearing VAE cache to ensure tensor size consistency...")
            self.pipeline.vae.model.clear_cache()
        
        # Setup latency tracking if requested
        if enable_timing:
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__)))
                from comprehensive_latency_tracker import ComprehensiveLatencyTracker
                latency_tracker = ComprehensiveLatencyTracker(
                    num_frame_per_block=self.config.num_frame_per_block,
                    device=self.device
                )
            except ImportError:
                latency_tracker = None
        else:
            latency_tracker = None
            
        # Generate noise
        noise = torch.randn([
            self.config.num_samples,
            self.config.num_output_frames, 
            16, 60, 104
        ], device=self.device, dtype=torch.bfloat16)
        
        # Run inference (text encoder and VAE already compiled after first run!)
        inference_start = time.perf_counter()
        video = self.pipeline.inference_with_timing(
            noise=noise,
            text_prompts_list=prompts_list,
            switch_frame_indices=switch_frame_indices,
            return_latents=False,
            latency_tracker=latency_tracker
        )
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Save video if output path provided
        if output_path:
            from torchvision.io import write_video
            from einops import rearrange
            current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0
            write_video(output_path, current_video[0].to(torch.uint8), fps=16)
            print(f"Video saved to: {output_path}")
            
        print(f"Inference completed in {inference_time:.2f} ms")
        return video, latency_tracker
        
    def run_from_data_file(self, data_path: str, output_dir: str = "/tmp/persistent_outputs"):
        """Run inference using the same data format as the original script"""
        from utils.dataset import MultiTextDataset
        from torch.utils.data import DataLoader, SequentialSampler
        from torchvision.io import write_video
        from einops import rearrange
        
        # Parse switch_frame_indices from config
        switch_frame_indices = [int(x) for x in self.config.switch_frame_indices.split(",") if x.strip()]
        
        # Create dataset
        dataset = MultiTextDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset), num_workers=0)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing {len(dataset)} prompts from {data_path}")
        print(f"Switch points: {switch_frame_indices}")
        print(f"Output directory: {output_dir}")
        
        for i, batch_data in enumerate(dataloader):
            idx = batch_data["idx"].item()
            prompts_list = batch_data["prompts_list"]
            
            print(f"\n[Prompt {idx}] Processing...")
            
            # Run inference
            video, _ = self.run_inference(
                prompts_list=prompts_list,
                switch_frame_indices=switch_frame_indices,
                output_path=None,
                enable_timing=True
            )
            
            # Save video
            current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0
            for seed_idx in range(self.config.num_samples):
                output_path = os.path.join(output_dir, f"persistent_rank0_{idx}_{seed_idx}.mp4")
                write_video(output_path, current_video[seed_idx].to(torch.uint8), fps=16)
                print(f"Video {seed_idx} saved to: {output_path}")
                
    def interactive_mode(self):
        """Interactive mode for multiple inference runs"""
        print("\nEntering interactive mode. Type 'help' for commands.")
        
        while True:
            try:
                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue
                    
                if cmd[0] == 'quit' or cmd[0] == 'exit':
                    print("Goodbye!")
                    break
                elif cmd[0] == 'help':
                    print("Commands:")
                    print("  data <data_file> [output_dir] - Run inference from data file")
                    print("  custom <prompt1> <prompt2> ... -- <switch1,switch2> <output_path> - Custom prompts")
                    print("  help - Show this help")
                    print("  quit/exit - Exit")
                elif cmd[0] == 'data':
                    if len(cmd) < 2:
                        print("Usage: data <data_file> [output_dir]")
                        continue
                    data_file = cmd[1]
                    output_dir = cmd[2] if len(cmd) > 2 else "/tmp/persistent_outputs"
                    self.run_from_data_file(data_file, output_dir)
                elif cmd[0] == 'custom':
                    print("Custom prompt mode not implemented yet. Use 'data' command.")
                else:
                    print(f"Unknown command: {cmd[0]}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser("Persistent Interactive Pipeline")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--interactive", action="store_true", 
                       help="Enter interactive mode for multiple runs")
    parser.add_argument("--data_path", type=str,
                       help="Run inference on data file and exit")
    parser.add_argument("--output_dir", type=str, default="/tmp/persistent_outputs",
                       help="Output directory for videos")
    parser.add_argument("--quantize", action="store_true",
                       help="Enable quantization (overrides config)")
    parser.add_argument("--no-quantize", action="store_true",
                       help="Disable quantization (overrides config)")
    # Legacy flags for backward compatibility
    parser.add_argument("--fp8", action="store_true",
                       help="Enable quantization (legacy flag)")
    parser.add_argument("--no-fp8", action="store_true",
                       help="Disable quantization (legacy flag)")
    args = parser.parse_args()
    
    # Initialize persistent pipeline
    persistent_pipeline = PersistentInteractivePipeline(args.config_path, args)
    
    if args.data_path:
        # Run once on data file and exit
        persistent_pipeline.run_from_data_file(args.data_path, args.output_dir)
    elif args.interactive:
        # Interactive mode
        persistent_pipeline.interactive_mode()
    else:
        print("Pipeline ready. Use --interactive for interactive mode or --data_path to run once.")

if __name__ == "__main__":
    main()