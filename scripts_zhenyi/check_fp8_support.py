#!/usr/bin/env python3
"""
Check if FP8 quantization is supported in the current environment
"""
import torch
import sys

def check_fp8_support():
    print("="*60)
    print("FP8 SUPPORT CHECK")
    print("="*60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Check FP8 support
    has_fp8_e4m3fn = hasattr(torch, 'float8_e4m3fn')
    has_fp8_e5m2 = hasattr(torch, 'float8_e5m2')
    
    print(f"\nFP8 support:")
    print(f"  torch.float8_e4m3fn: {'✅ Available' if has_fp8_e4m3fn else '❌ Not available'}")
    print(f"  torch.float8_e5m2: {'✅ Available' if has_fp8_e5m2 else '❌ Not available'}")
    
    if has_fp8_e4m3fn or has_fp8_e5m2:
        print("\n✅ FP8 quantization is supported!")
        
        # Test creating FP8 tensors
        try:
            if has_fp8_e4m3fn:
                test_tensor = torch.randn(10, 10, dtype=torch.float32)
                fp8_tensor = test_tensor.to(torch.float8_e4m3fn)
                print(f"  Test tensor conversion: ✅ {torch.float32} → {torch.float8_e4m3fn}")
            
            if torch.cuda.is_available() and has_fp8_e4m3fn:
                test_cuda = torch.randn(10, 10, device='cuda', dtype=torch.float32)
                fp8_cuda = test_cuda.to(torch.float8_e4m3fn)
                print(f"  CUDA tensor conversion: ✅ {torch.float32} → {torch.float8_e4m3fn}")
                
        except Exception as e:
            print(f"  ⚠️  Tensor conversion test failed: {e}")
            
    else:
        print("\n❌ FP8 quantization is NOT supported")
        print("Requirements:")
        print("  - PyTorch 2.1.0 or later")
        print("  - CUDA support")
        print("  - Compatible GPU (H100, etc.)")
        
        if torch.__version__ < "2.1.0":
            print(f"  Current version {torch.__version__} is too old")
            
    print("="*60)

if __name__ == "__main__":
    check_fp8_support()