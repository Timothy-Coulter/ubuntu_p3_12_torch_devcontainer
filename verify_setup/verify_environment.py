#!/usr/bin/env python3
"""
Complete Development Environment Verification Script

This script verifies that all components of the torch-starter development
environment are properly configured and working.

Usage:
    python verify_setup/verify_environment.py
    python verify_setup/verify_environment.py --full
    python verify_setup/verify_environment.py --export-report
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text


console = Console()


class EnvironmentVerifier:
    """Comprehensive environment verification."""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.report_data: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "environment": "torch-starter-devcontainer",
            "tests": {}
        }
    
    def log_result(self, category: str, test_name: str, passed: bool, 
                   details: Optional[str] = None, value: Optional[str] = None) -> None:
        """Log test result."""
        if category not in self.results:
            self.results[category] = {}
        
        self.results[category][test_name] = {
            "passed": passed,
            "details": details,
            "value": value
        }
        
        # Also store in report data
        if category not in self.report_data["tests"]:
            self.report_data["tests"][category] = {}
        
        self.report_data["tests"][category][test_name] = {
            "status": "pass" if passed else "fail",
            "details": details,
            "value": value
        }
    
    def test_python_environment(self) -> bool:
        """Test Python environment and basic packages."""
        console.print("\n[bold cyan]Python Environment[/bold cyan]")
        category = "python"
        all_passed = True
        
        # Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        expected_version = "3.12"
        version_ok = python_version.startswith(expected_version)
        
        self.log_result(category, "python_version", version_ok, 
                       f"Expected: {expected_version}.x, Got: {python_version}", python_version)
        
        if version_ok:
            console.print(f"✅ Python version: {python_version}")
        else:
            console.print(f"❌ Python version: {python_version} (expected {expected_version}.x)", style="red")
            all_passed = False
        
        # Virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        venv_path = os.environ.get('VIRTUAL_ENV', '')
        
        self.log_result(category, "virtual_environment", in_venv, 
                       f"Virtual env path: {venv_path}", str(in_venv))
        
        if in_venv:
            console.print(f"✅ Virtual environment active: {venv_path}")
        else:
            console.print("⚠️  No virtual environment detected", style="yellow")
        
        # Package manager (uv)
        try:
            uv_result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
            uv_available = uv_result.returncode == 0
            uv_version = uv_result.stdout.strip() if uv_available else "Not available"
        except FileNotFoundError:
            uv_available = False
            uv_version = "Not found"
        
        self.log_result(category, "uv_package_manager", uv_available, uv_version, uv_version)
        
        if uv_available:
            console.print(f"✅ UV package manager: {uv_version}")
        else:
            console.print("❌ UV package manager not found", style="red")
            all_passed = False
        
        return all_passed
    
    def test_core_ml_packages(self) -> bool:
        """Test core ML packages."""
        console.print("\n[bold cyan]Core ML Packages[/bold cyan]")
        category = "ml_packages"
        all_passed = True
        
        # Required packages with version checks
        required_packages = {
            "torch": ">=2.0.0",
            "torchvision": ">=0.15.0",
            "transformers": ">=4.44.0",
            "numpy": ">=1.26.0",
            "pandas": ">=2.2.0",
            "datasets": ">=2.19.0",
            "accelerate": ">=0.33.0",
        }
        
        for package_name, min_version in required_packages.items():
            try:
                module = importlib.import_module(package_name)
                version = getattr(module, '__version__', 'Unknown')
                
                # Simple version check (just check if module loads)
                import_ok = True
                self.log_result(category, f"{package_name}_import", import_ok, 
                               f"Version: {version}", version)
                console.print(f"✅ {package_name}: {version}")
                
            except ImportError as e:
                self.log_result(category, f"{package_name}_import", False, 
                               f"Import error: {str(e)}", "Not available")
                console.print(f"❌ {package_name}: Import failed - {e}", style="red")
                all_passed = False
        
        return all_passed
    
    def test_jupyter_environment(self) -> bool:
        """Test Jupyter environment."""
        console.print("\n[bold cyan]Jupyter Environment[/bold cyan]")
        category = "jupyter"
        all_passed = True
        
        # Test Jupyter packages
        jupyter_packages = ["jupyter", "ipykernel", "jupyterlab"]
        
        for package in jupyter_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'Unknown')
                self.log_result(category, f"{package}_available", True, 
                               f"Version: {version}", version)
                console.print(f"✅ {package}: {version}")
            except ImportError:
                self.log_result(category, f"{package}_available", False, 
                               "Not available", "Not available")
                console.print(f"❌ {package}: Not available", style="red")
                all_passed = False
        
        # Check for kernel installation
        try:
            result = subprocess.run(["jupyter", "kernelspec", "list"], 
                                  capture_output=True, text=True)
            kernel_list_ok = result.returncode == 0
            kernel_output = result.stdout if kernel_list_ok else result.stderr
            
            # Look for our project kernel
            project_kernel_found = "torch_starter" in kernel_output
            
            self.log_result(category, "kernel_installed", project_kernel_found, 
                           kernel_output, str(project_kernel_found))
            
            if project_kernel_found:
                console.print("✅ Project Jupyter kernel installed")
            else:
                console.print("⚠️  Project Jupyter kernel not found", style="yellow")
                
        except FileNotFoundError:
            self.log_result(category, "jupyter_command", False, 
                           "Jupyter command not found", "Not available")
            console.print("❌ Jupyter command not available", style="red")
            all_passed = False
        
        return all_passed
    
    def test_development_tools(self) -> bool:
        """Test development tools."""
        console.print("\n[bold cyan]Development Tools[/bold cyan]")
        category = "dev_tools"
        all_passed = True
        
        # Test development packages
        dev_packages = ["ruff", "mypy", "pytest"]
        
        for package in dev_packages:
            try:
                # Try importing first
                importlib.import_module(package)
                import_ok = True
                console.print(f"✅ {package}: Available as module")
            except ImportError:
                import_ok = False
                
                # Try as command line tool
                try:
                    result = subprocess.run([package, "--version"], 
                                          capture_output=True, text=True)
                    cmd_ok = result.returncode == 0
                    if cmd_ok:
                        version = result.stdout.strip()
                        console.print(f"✅ {package}: {version}")
                        import_ok = True
                    else:
                        console.print(f"❌ {package}: Not available", style="red")
                        all_passed = False
                except FileNotFoundError:
                    console.print(f"❌ {package}: Not available", style="red")
                    all_passed = False
            
            self.log_result(category, f"{package}_available", import_ok, 
                           "Available" if import_ok else "Not available", 
                           str(import_ok))
        
        # Test git
        try:
            git_result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            git_ok = git_result.returncode == 0
            git_version = git_result.stdout.strip() if git_ok else "Not available"
            
            self.log_result(category, "git_available", git_ok, git_version, git_version)
            
            if git_ok:
                console.print(f"✅ Git: {git_version}")
            else:
                console.print("❌ Git: Not available", style="red")
                all_passed = False
        except FileNotFoundError:
            console.print("❌ Git: Not found", style="red")
            all_passed = False
        
        return all_passed
    
    def test_environment_variables(self) -> bool:
        """Test important environment variables."""
        console.print("\n[bold cyan]Environment Variables[/bold cyan]")
        category = "env_vars"
        all_passed = True
        
        # Important environment variables
        important_vars = {
            "CUDA_VISIBLE_DEVICES": "GPU visibility",
            "HF_HOME": "Hugging Face cache directory",
            "TORCH_HOME": "PyTorch cache directory",
            "PYTHONPATH": "Python module search path",
        }
        
        for var_name, description in important_vars.items():
            value = os.environ.get(var_name)
            is_set = value is not None
            
            self.log_result(category, var_name.lower(), is_set, 
                           f"{description}: {value}" if is_set else f"{description}: Not set", 
                           value or "Not set")
            
            if is_set:
                console.print(f"✅ {var_name}: {value}")
            else:
                console.print(f"⚠️  {var_name}: Not set", style="yellow")
        
        return all_passed
    
    def test_file_permissions(self) -> bool:
        """Test file permissions and directory access."""
        console.print("\n[bold cyan]File System Permissions[/bold cyan]")
        category = "file_system"
        all_passed = True
        
        # Test important directories
        test_dirs = {
            "/workspaces": "Workspace directory",
            "/home/ubuntu/.cache": "User cache directory", 
            "/home/ubuntu/.local": "User local directory",
        }
        
        for dir_path, description in test_dirs.items():
            path_obj = Path(dir_path)
            exists = path_obj.exists()
            readable = path_obj.is_dir() and os.access(dir_path, os.R_OK) if exists else False
            writable = os.access(dir_path, os.W_OK) if exists else False
            
            status = exists and readable and writable
            details = f"Exists: {exists}, Readable: {readable}, Writable: {writable}"
            
            self.log_result(category, f"{dir_path.replace('/', '_').strip('_')}_access", 
                           status, f"{description} - {details}", str(status))
            
            if status:
                console.print(f"✅ {dir_path}: {description} accessible")
            else:
                console.print(f"❌ {dir_path}: {description} - {details}", style="red")
                all_passed = False
        
        return all_passed
    
    def create_summary_table(self) -> Table:
        """Create summary table of all test results."""
        table = Table(title="Environment Verification Summary")
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Tests", justify="right", style="magenta")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Status", style="bold")
        
        for category, tests in self.results.items():
            total_tests = len(tests)
            passed_tests = sum(1 for test in tests.values() if test["passed"])
            failed_tests = total_tests - passed_tests
            
            if failed_tests == 0:
                status = "✅ All Pass"
                status_style = "green"
            elif passed_tests > failed_tests:
                status = "⚠️  Mostly Pass"
                status_style = "yellow"
            else:
                status = "❌ Issues Found"
                status_style = "red"
            
            table.add_row(
                category.replace("_", " ").title(),
                str(total_tests),
                str(passed_tests),
                str(failed_tests),
                Text(status, style=status_style)
            )
        
        return table
    
    def export_report(self, filename: str = "environment_verification_report.json") -> None:
        """Export detailed report to JSON file."""
        report_path = Path(filename)
        
        # Add summary to report
        self.report_data["summary"] = {}
        for category, tests in self.results.items():
            total_tests = len(tests)
            passed_tests = sum(1 for test in tests.values() if test["passed"])
            self.report_data["summary"][category] = {
                "total": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            }
        
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self.report_data, f, indent=2, ensure_ascii=False)
            console.print(f"✅ Report exported to: {report_path}", style="green")
        except Exception as e:
            console.print(f"❌ Failed to export report: {e}", style="red")
    
    def run_verification(self, full_test: bool = False) -> bool:
        """Run all verification tests."""
        console.print(Panel.fit("🔍 Environment Verification for torch-starter DevContainer", 
                               style="bold blue"))
        
        # Define test suite
        tests = [
            ("Python Environment", self.test_python_environment),
            ("ML Packages", self.test_core_ml_packages),
            ("Jupyter Environment", self.test_jupyter_environment),
            ("Development Tools", self.test_development_tools),
            ("Environment Variables", self.test_environment_variables),
        ]
        
        if full_test:
            tests.append(("File System", self.test_file_permissions))
        
        # Run tests
        overall_success = True
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            for test_name, test_func in tests:
                task = progress.add_task(f"Testing {test_name}...", total=None)
                try:
                    success = test_func()
                    if not success:
                        overall_success = False
                except Exception as e:
                    console.print(f"❌ {test_name} crashed: {e}", style="red")
                    overall_success = False
                finally:
                    progress.remove_task(task)
        
        # Show summary
        console.print("\n" + "="*60)
        console.print(self.create_summary_table())
        
        # Final status
        console.print("\n" + "="*60)
        if overall_success:
            console.print("🎉 Environment verification completed successfully!", style="bold green")
            console.print("Your development environment is ready for ML development.", style="green")
        else:
            console.print("⚠️  Some verification tests failed.", style="bold yellow")
            console.print("Check the details above and fix any issues.", style="yellow")
        
        return overall_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Environment Verification for torch-starter DevContainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_setup/verify_environment.py                # Run basic verification
  python verify_setup/verify_environment.py --full        # Run comprehensive tests
  python verify_setup/verify_environment.py --export-report # Export detailed report
        """
    )
    
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Run comprehensive verification including file system tests"
    )
    
    parser.add_argument(
        "--export-report", "-e",
        action="store_true",
        help="Export detailed verification report to JSON"
    )
    
    parser.add_argument(
        "--report-file", "-r",
        type=str,
        default="environment_verification_report.json",
        help="Specify report filename (default: environment_verification_report.json)"
    )
    
    args = parser.parse_args()
    
    # Create verifier and run tests
    verifier = EnvironmentVerifier()
    success = verifier.run_verification(full_test=args.full)
    
    # Export report if requested
    if args.export_report:
        verifier.export_report(args.report_file)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()