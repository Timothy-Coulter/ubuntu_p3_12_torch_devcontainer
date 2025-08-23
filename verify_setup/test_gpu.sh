#!/usr/bin/env python3
"""
GPU Test and Verification Script for torch-starter DevContainer

This script comprehensively tests the GPU setup and PyTorch CUDA functionality.
Run this after container startup to verify everything is working correctly.

Usage:
    python verify_setup/test_gpu.py
    python verify_setup/test_gpu.py --verbose
    python verify_setup/test_gpu.py --benchmark
"""

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import transformers
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


console = Console()


class GPUTester:
    """Comprehensive GPU and CUDA testing suite."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: Dict[str, bool] = {}
        self.warnings: List[str] = []
        self.info: Dict[str, str] = {}
        
    def log(self, message: str, level: str = "info") -> None:
        """Log messages with appropriate styling."""
        if level == "error":
            console.print(f"❌ {message}", style="red")
        elif level == "warning":
            console.print(f"⚠️  {message}", style="yellow")
            self.warnings.append(message)
        elif level == "success":
            console.print(f"✅ {message}", style="green")
        else:
            if self.verbose:
                console.print(f"ℹ️  {message}", style="blue")
    
    def test_system_info(self) -> bool:
        """Test system information and environment."""
        console.print("\n[bold cyan]System Information[/bold cyan]")
        
        try:
            # Python version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            self.info["python_version"] = python_version
            self.log(f"Python: {python_version}")
            
            # Platform info
            system_info = f"{platform.system()} {platform.release()}"
            self.info["system"] = system_info
            self.log(f"System: {system_info}")
            
            # PyTorch version
            torch_version = torch.__version__
            self.info["torch_version"] = torch_version
            self.log(f"PyTorch: {torch_version}")
            
            # CUDA version from PyTorch
            cuda_version = torch.version.cuda or "Not available"
            self.info["cuda_version"] = cuda_version
            self.log(f"CUDA Version (PyTorch): {cuda_version}")
            
            # cuDNN version
            cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "Not available"
            self.info["cudnn_version"] = str(cudnn_version)
            self.log(f"cuDNN Version: {cudnn_version}")
            
            return True
            
        except Exception as e:
            self.log(f"System info test failed: {e}", "error")
            return False
    
    def test_nvidia_smi(self) -> bool:
        """Test nvidia-smi availability and GPU detection."""
        console.print("\n[bold cyan]NVIDIA Driver Information[/bold cyan]")
        
        try:
            # Run nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = result.stdout.strip().split('\n')
            self.info["gpu_count"] = str(len(gpus))
            
            for i, gpu_info in enumerate(gpus):
                parts = [part.strip() for part in gpu_info.split(',')]
                if len(parts) >= 4:
                    name, memory, compute_cap, driver = parts[:4]
                    self.log(f"GPU {i}: {name}")
                    self.log(f"  Memory: {memory}")
                    self.log(f"  Compute Capability: {compute_cap}")
                    if i == 0:  # Store driver version from first GPU
                        self.info["driver_version"] = driver
                        self.log(f"  Driver Version: {driver}")
            
            self.log("nvidia-smi working correctly", "success")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.log(f"nvidia-smi failed: {e}", "error")
            self.log("This may indicate missing NVIDIA drivers or GPU not available", "warning")
            return False
    
    def test_cuda_availability(self) -> bool:
        """Test PyTorch CUDA availability."""
        console.print("\n[bold cyan]PyTorch CUDA Tests[/bold cyan]")
        
        # Basic CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            self.log("CUDA is available in PyTorch", "success")
        else:
            self.log("CUDA is NOT available in PyTorch", "error")
            return False
        
        # Device count
        device_count = torch.cuda.device_count()
        self.info["torch_device_count"] = str(device_count)
        
        if device_count > 0:
            self.log(f"Found {device_count} CUDA device(s)", "success")
        else:
            self.log("No CUDA devices found", "error")
            return False
        
        # Test each device
        for i in range(device_count):
            try:
                device_props = torch.cuda.get_device_properties(i)
                self.log(f"Device {i}: {device_props.name}")
                self.log(f"  Compute Capability: {device_props.major}.{device_props.minor}")
                self.log(f"  Total Memory: {device_props.total_memory // 1024**3}GB")
                self.log(f"  Multiprocessors: {device_props.multi_processor_count}")
                
                # Test device context
                with torch.cuda.device(i):
                    current_device = torch.cuda.current_device()
                    if current_device == i:
                        self.log(f"Successfully switched to device {i}", "success")
                    else:
                        self.log(f"Failed to switch to device {i}", "error")
                        return False
                        
            except Exception as e:
                self.log(f"Error testing device {i}: {e}", "error")
                return False
        
        return True
    
    def test_tensor_operations(self) -> bool:
        """Test basic tensor operations on GPU."""
        console.print("\n[bold cyan]GPU Tensor Operations[/bold cyan]")
        
        if not torch.cuda.is_available():
            self.log("Skipping tensor tests - CUDA not available", "warning")
            return False
        
        try:
            device = torch.device('cuda:0')
            self.log(f"Testing on device: {device}")
            
            # Basic tensor creation
            x = torch.randn(1000, 1000, device=device)
            self.log(f"Created tensor on GPU: {x.device}")
            
            # Basic arithmetic
            y = torch.randn(1000, 1000, device=device)
            z = x + y
            self.log(f"Addition result shape: {z.shape}, device: {z.device}")
            
            # Matrix multiplication
            result = torch.mm(x, y)
            self.log(f"Matrix multiplication result: {result.shape}")
            
            # Move to CPU and back
            cpu_tensor = z.cpu()
            gpu_tensor = cpu_tensor.cuda()
            self.log("Successfully moved tensor CPU -> GPU")
            
            # Memory management
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated(0) // 1024**2
            self.log(f"GPU memory allocated: {memory_allocated}MB")
            
            self.log("Basic tensor operations successful", "success")
            return True
            
        except Exception as e:
            self.log(f"Tensor operations failed: {e}", "error")
            return False
    
    def test_neural_network(self) -> bool:
        """Test neural network operations on GPU."""
        console.print("\n[bold cyan]Neural Network GPU Tests[/bold cyan]")
        
        if not torch.cuda.is_available():
            self.log("Skipping neural network tests - CUDA not available", "warning")
            return False
        
        try:
            device = torch.device('cuda:0')
            
            # Simple neural network
            class TestNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(784, 128)
                    self.fc2 = nn.Linear(128, 64)
                    self.fc3 = nn.Linear(64, 10)
                    self.dropout = nn.Dropout(0.2)
                
                def forward(self, x):
                    x = x.view(-1, 784)
                    x = F.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = F.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
            
            # Create and move model to GPU
            model = TestNet().to(device)
            self.log(f"Model created and moved to GPU")
            
            # Test forward pass
            batch_size = 32
            dummy_input = torch.randn(batch_size, 1, 28, 28, device=device)
            output = model(dummy_input)
            self.log(f"Forward pass successful: {output.shape}")
            
            # Test backward pass
            loss_fn = nn.CrossEntropyLoss()
            target = torch.randint(0, 10, (batch_size,), device=device)
            loss = loss_fn(output, target)
            loss.backward()
            self.log(f"Backward pass successful: loss = {loss.item():.4f}")
            
            # Test optimizer
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.step()
            self.log("Optimizer step successful")
            
            self.log("Neural network operations successful", "success")
            return True
            
        except Exception as e:
            self.log(f"Neural network test failed: {e}", "error")
            return False
    
    def test_transformers_gpu(self) -> bool:
        """Test Hugging Face Transformers with GPU."""
        console.print("\n[bold cyan]Transformers GPU Tests[/bold cyan]")
        
        if not torch.cuda.is_available():
            self.log("Skipping transformers tests - CUDA not available", "warning")
            return False
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # Use a small model for testing
            model_name = "distilbert-base-uncased"
            self.log(f"Loading model: {model_name}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Move model to GPU
            device = torch.device('cuda:0')
            model = model.to(device)
            self.log("Model moved to GPU")
            
            # Test inference
            text = "Hello, this is a test for GPU inference with transformers."
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            self.log(f"Inference successful: output shape = {outputs.last_hidden_state.shape}")
            self.log("Transformers GPU test successful", "success")
            return True
            
        except Exception as e:
            self.log(f"Transformers GPU test failed: {e}", "error")
            self.log("This might be due to missing model files - not critical", "warning")
            return False
    
    def benchmark_gpu_performance(self) -> Optional[Dict[str, float]]:
        """Run GPU performance benchmarks."""
        console.print("\n[bold cyan]GPU Performance Benchmarks[/bold cyan]")
        
        if not torch.cuda.is_available():
            self.log("Skipping benchmarks - CUDA not available", "warning")
            return None
        
        benchmarks = {}
        device = torch.device('cuda:0')
        
        try:
            # Matrix multiplication benchmark
            sizes = [1000, 2000, 4000]
            for size in sizes:
                # Warmup
                for _ in range(3):
                    a = torch.randn(size, size, device=device)
                    b = torch.randn(size, size, device=device)
                    torch.mm(a, b)
                    torch.cuda.synchronize()
                
                # Benchmark
                torch.cuda.synchronize()
                start_time = time.time()
                
                for _ in range(10):
                    a = torch.randn(size, size, device=device)
                    b = torch.randn(size, size, device=device)
                    result = torch.mm(a, b)
                    torch.cuda.synchronize()
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 10
                benchmarks[f"matmul_{size}x{size}"] = avg_time
                
                # Calculate FLOPS (rough estimate)
                flops = 2 * size**3  # Multiply-accumulate operations
                gflops = (flops / avg_time) / 1e9
                self.log(f"Matrix multiplication {size}x{size}: {avg_time:.4f}s ({gflops:.1f} GFLOPS)")
            
            # Memory bandwidth test
            size_mb = 100
            elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
            
            # Warmup
            for _ in range(3):
                data = torch.randn(elements, device=device)
                result = data * 2.0
                torch.cuda.synchronize()
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(100):
                data = torch.randn(elements, device=device)
                result = data * 2.0
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 100
            bandwidth = (size_mb * 2) / avg_time  # Read + Write
            benchmarks["memory_bandwidth_mb_per_s"] = bandwidth
            
            self.log(f"Memory bandwidth: {bandwidth:.1f} MB/s")
            
            return benchmarks
            
        except Exception as e:
            self.log(f"Benchmarking failed: {e}", "error")
            return None
    
    def create_summary_table(self) -> Table:
        """Create a summary table of test results."""
        table = Table(title="GPU Test Summary")
        table.add_column("Test", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="green")
        
        # Add system info
        table.add_row("Python Version", "✅", self.info.get("python_version", "Unknown"))
        table.add_row("PyTorch Version", "✅", self.info.get("torch_version", "Unknown"))
        table.add_row("CUDA Version", "✅" if self.info.get("cuda_version") != "Not available" else "❌", 
                     self.info.get("cuda_version", "Unknown"))
        table.add_row("cuDNN Version", "✅" if self.info.get("cudnn_version") != "Not available" else "❌", 
                     self.info.get("cudnn_version", "Unknown"))
        
        # Add test results
        for test_name, passed in self.results.items():
            status = "✅" if passed else "❌"
            table.add_row(test_name.replace("_", " ").title(), status, "")
        
        return table
    
    def run_all_tests(self, run_benchmarks: bool = False) -> bool:
        """Run all GPU tests."""
        console.print(Panel.fit("🚀 GPU Test Suite for torch-starter DevContainer", style="bold blue"))
        
        # Run tests
        tests = [
            ("system_info", self.test_system_info),
            ("nvidia_smi", self.test_nvidia_smi),
            ("cuda_availability", self.test_cuda_availability),
            ("tensor_operations", self.test_tensor_operations),
            ("neural_network", self.test_neural_network),
            ("transformers_gpu", self.test_transformers_gpu),
        ]
        
        all_passed = True
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            for test_name, test_func in tests:
                task = progress.add_task(f"Running {test_name.replace('_', ' ')}...", total=None)
                try:
                    result = test_func()
                    self.results[test_name] = result
                    if not result:
                        all_passed = False
                except Exception as e:
                    console.print(f"❌ Test {test_name} crashed: {e}", style="red")
                    self.results[test_name] = False
                    all_passed = False
                finally:
                    progress.remove_task(task)
        
        # Run benchmarks if requested
        if run_benchmarks and torch.cuda.is_available():
            console.print("\n" + "="*50)
            self.benchmark_gpu_performance()
        
        # Show summary
        console.print("\n" + "="*50)
        console.print(self.create_summary_table())
        
        # Show warnings
        if self.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in self.warnings:
                console.print(f"  ⚠️  {warning}", style="yellow")
        
        # Final status
        console.print("\n" + "="*50)
        if all_passed:
            console.print("🎉 All tests passed! GPU setup is working correctly.", style="bold green")
        else:
            console.print("❌ Some tests failed. Check the output above for details.", style="bold red")
            if not torch.cuda.is_available():
                console.print("\n💡 If you're running without a GPU, this is expected.", style="blue")
                console.print("💡 The container will still work for CPU-only operations.", style="blue")
        
        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GPU Test Suite for torch-starter DevContainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_setup/test_gpu.py                    # Run basic tests
  python verify_setup/test_gpu.py --verbose         # Run with detailed output
  python verify_setup/test_gpu.py --benchmark       # Run with performance benchmarks
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run performance benchmarks"
    )
    
    args = parser.parse_args()
    
    # Create tester and run
    tester = GPUTester(verbose=args.verbose)
    success = tester.run_all_tests(run_benchmarks=args.benchmark)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()