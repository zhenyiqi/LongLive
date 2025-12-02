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
        # Use float16 for BitsAndBytes int8 compatibility
        self.pipeline = self.pipeline.to(dtype=torch.float16)
        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)
        
        # Optional: GPU quantization via bitsandbytes
        quant_cfg = getattr(self.config, "quantization", None)
        if quant_cfg and quant_cfg.get("enabled", False) and torch.cuda.is_available():
            try:
                method = str(quant_cfg.get("method", "bnb")).lower()
                dtype = str(quant_cfg.get("dtype", "int8")).lower()
                models = quant_cfg.get("models", ["generator"])
                if method in ["bnb", "bitsandbytes"] or dtype in ["bnb_int8", "bnb_int4", "int4", "int8"]:
                    self._apply_bnb_quantization(models, dtype)
                else:
                    print(f"[Quantization] Skipping unknown method '{method}'.")
            except Exception as e:
                print(f"[Quantization] bitsandbytes quantization skipped due to error: {e}")
        
        init_time = (time.perf_counter() - init_start) * 1000
        print(f"Pipeline initialization completed: {init_time:.2f} ms")
        print("="*60)
        print("NOTE: First inference will include compilation time (~30s).")
        print("Subsequent inferences will be much faster (~60s vs ~90s)!")
        print("="*60)
    
    def _apply_bnb_quantization(self, models_to_quantize: list, quant_dtype: str):
        """
        Apply bitsandbytes quantization (8-bit or 4-bit) to Linear layers of selected models.
        - quant_dtype: 'bnb_int8'|'int8' or 'bnb_int4'|'int4'
        """
        import torch
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception as e:
            raise RuntimeError(f"bitsandbytes is required for GPU quantization: {e}")

        use_int4 = quant_dtype.lower() in ["bnb_int4", "int4"]

        def replace_linear_with_bnb(module: torch.nn.Module) -> torch.nn.Module:
            for name, child in list(module.named_children()):
                # Recurse
                new_child = replace_linear_with_bnb(child)
                if new_child is not child:
                    setattr(module, name, new_child)
                    child = new_child
                # Replace Linear layers
                if isinstance(child, torch.nn.Linear):
                    in_f, out_f = child.in_features, child.out_features
                    bias = child.bias is not None
                    if use_int4:
                        quant_layer = bnb.nn.Linear4bit(
                            in_f,
                            out_f,
                            bias=bias,
                            compute_dtype=torch.bfloat16,
                            quant_type="nf4",
                            compress_statistics=True,
                            quant_storage=torch.uint8,
                        )
                    else:
                        quant_layer = bnb.nn.Linear8bitLt(
                            in_f,
                            out_f,
                            bias=bias,
                            has_fp16_weights=False,
                            threshold=6.0,
                        )
                    # Ensure the quantized layer lives on the same (CUDA) device as the original
                    quant_layer = quant_layer.to(child.weight.device)
                    # Copy weights and ensure proper dtype handling
                    with torch.no_grad():
                        # Convert weights to appropriate dtype for BitsAndBytes
                        if use_int4:
                            # 4-bit works with bfloat16
                            quant_layer.weight.copy_(child.weight.detach())
                            if bias:
                                quant_layer.bias.copy_(child.bias.detach())
                        else:
                            # 8-bit Linear8bitLt requires float16, not bfloat16
                            weight_data = child.weight.detach()
                            if weight_data.dtype == torch.bfloat16:
                                weight_data = weight_data.to(torch.float16)
                            quant_layer.weight.copy_(weight_data)
                            if bias:
                                bias_data = child.bias.detach()
                                if bias_data.dtype == torch.bfloat16:
                                    bias_data = bias_data.to(torch.float16)
                                quant_layer.bias.copy_(bias_data)
                    setattr(module, name, quant_layer)
            return module

        print(f"[BNB] Quantizing with {'4-bit (nf4)' if use_int4 else '8-bit'} Linear layers...")
        if "text_encoder" in models_to_quantize and hasattr(self.pipeline, "text_encoder"):
            self.pipeline.text_encoder = replace_linear_with_bnb(self.pipeline.text_encoder)
            print("[BNB] Text encoder quantized")
        if "generator" in models_to_quantize and hasattr(self.pipeline, "generator"):
            target = getattr(self.pipeline.generator, "model", self.pipeline.generator)
            replace_linear_with_bnb(target)
            print("[BNB] Generator quantized")
        if "vae" in models_to_quantize and hasattr(self.pipeline, "vae"):
            replace_linear_with_bnb(self.pipeline.vae)
            print("[BNB] VAE linear layers quantized (if any)")
        
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