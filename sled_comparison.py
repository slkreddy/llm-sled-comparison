#!/usr/bin/env python3
"""
LLM SLED Comparison Tool

Compares standard LLM decoding with SLED (Sliding Window LLM Decoding) optimization technique.
SLED provides memory-efficient inference by using a sliding window approach.

Author: LaxmiKumar Reddy Sammeta
License: MIT
"""

import torch
import torch.nn.functional as F
import time
import psutil
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class StandardLLMDecoder:
    """Standard autoregressive LLM decoder implementation."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> Tuple[str, Dict]:
        """Generate text using standard autoregressive decoding."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Track memory and time
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        metrics = {
            "generation_time": end_time - start_time,
            "memory_used": end_memory - start_memory,
            "tokens_generated": len(outputs[0]) - len(input_ids[0]),
            "method": "Standard"
        }
        
        return generated_text, metrics


class SLEDDecoder:
    """SLED (Sliding Window LLM Decoding) implementation for memory-efficient inference."""
    
    def __init__(self, model_name: str = "gpt2", window_size: int = 50, stride: int = 25):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.window_size = window_size
        self.stride = stride
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> Tuple[str, Dict]:
        """Generate text using SLED sliding window approach."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        
        # Track memory and time
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        generated_tokens = input_ids.tolist()
        
        with torch.no_grad():
            while len(generated_tokens) < max_length:
                # Apply sliding window
                if len(generated_tokens) > self.window_size:
                    window_start = len(generated_tokens) - self.window_size
                    context_ids = torch.tensor([generated_tokens[window_start:]])
                else:
                    context_ids = torch.tensor([generated_tokens])
                
                # Generate next token
                outputs = self.model(context_ids)
                logits = outputs.logits[0, -1, :]
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated_tokens.append(next_token)
                
                # Stop if EOS token is generated
                if next_token == self.tokenizer.eos_token_id:
                    break
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        metrics = {
            "generation_time": end_time - start_time,
            "memory_used": end_memory - start_memory,
            "tokens_generated": len(generated_tokens) - len(input_ids),
            "method": "SLED",
            "window_size": self.window_size,
            "stride": self.stride
        }
        
        return generated_text, metrics


class ComparisonRunner:
    """Runs and compares Standard vs SLED decoding methods."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.standard_decoder = StandardLLMDecoder(model_name)
        self.sled_decoder = SLEDDecoder(model_name)
        self.results = []
    
    def run_comparison(self, prompts: List[str], max_length: int = 100, 
                     temperature: float = 1.0, num_runs: int = 3) -> List[Dict]:
        """Run comparison between Standard and SLED decoders."""
        
        print(f"Running comparison with {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Run multiple times for statistical significance
            for run in range(num_runs):
                print(f"  Run {run+1}/{num_runs}")
                
                # Standard decoder
                try:
                    std_text, std_metrics = self.standard_decoder.generate(
                        prompt, max_length, temperature
                    )
                    std_metrics["prompt_id"] = i
                    std_metrics["run"] = run
                    std_metrics["prompt"] = prompt
                    self.results.append(std_metrics)
                    print(f"    Standard: {std_metrics['generation_time']:.3f}s, "
                          f"{std_metrics['memory_used']:.1f}MB")
                except Exception as e:
                    print(f"    Standard decoder failed: {e}")
                
                # SLED decoder
                try:
                    sled_text, sled_metrics = self.sled_decoder.generate(
                        prompt, max_length, temperature
                    )
                    sled_metrics["prompt_id"] = i
                    sled_metrics["run"] = run
                    sled_metrics["prompt"] = prompt
                    self.results.append(sled_metrics)
                    print(f"    SLED: {sled_metrics['generation_time']:.3f}s, "
                          f"{sled_metrics['memory_used']:.1f}MB")
                except Exception as e:
                    print(f"    SLED decoder failed: {e}")
        
        return self.results
    
    def analyze_results(self):
        """Analyze and visualize comparison results."""
        if not self.results:
            print("No results to analyze. Run comparison first.")
            return
        
        # Separate results by method
        standard_results = [r for r in self.results if r['method'] == 'Standard']
        sled_results = [r for r in self.results if r['method'] == 'SLED']
        
        if not standard_results or not sled_results:
            print("Need results from both methods to compare.")
            return
        
        # Calculate average metrics
        std_avg_time = sum(r['generation_time'] for r in standard_results) / len(standard_results)
        sled_avg_time = sum(r['generation_time'] for r in sled_results) / len(sled_results)
        
        std_avg_memory = sum(r['memory_used'] for r in standard_results) / len(standard_results)
        sled_avg_memory = sum(r['memory_used'] for r in sled_results) / len(sled_results)
        
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Total runs: {len(self.results)}")
        print(f"Standard decoder runs: {len(standard_results)}")
        print(f"SLED decoder runs: {len(sled_results)}")
        
        print("\nPerformance Metrics:")
        print(f"  Generation Time:")
        print(f"    Standard: {std_avg_time:.3f}s")
        print(f"    SLED:     {sled_avg_time:.3f}s")
        print(f"    Speedup:  {std_avg_time/sled_avg_time:.2f}x" if sled_avg_time > 0 else "    N/A")
        
        print(f"\n  Memory Usage:")
        print(f"    Standard: {std_avg_memory:.1f}MB")
        print(f"    SLED:     {sled_avg_memory:.1f}MB")
        print(f"    Reduction: {((std_avg_memory-sled_avg_memory)/std_avg_memory)*100:.1f}%" 
              if std_avg_memory > 0 else "    N/A")
        
        # Create visualization
        self.create_visualizations()
    
    def create_visualizations(self):
        """Create comparison visualizations."""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            df = pd.DataFrame(self.results)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Generation time comparison
            sns.boxplot(data=df, x='method', y='generation_time', ax=axes[0])
            axes[0].set_title('Generation Time Comparison')
            axes[0].set_ylabel('Time (seconds)')
            
            # Memory usage comparison
            sns.boxplot(data=df, x='method', y='memory_used', ax=axes[1])
            axes[1].set_title('Memory Usage Comparison')
            axes[1].set_ylabel('Memory (MB)')
            
            plt.tight_layout()
            plt.savefig('sled_comparison_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("\nVisualization saved as 'sled_comparison_results.png'")
            
        except ImportError:
            print("\nMatplotlib/Seaborn not available. Skipping visualization.")
        except Exception as e:
            print(f"\nVisualization failed: {e}")


def main():
    """Main function to run the SLED comparison."""
    print("LLM SLED Comparison Tool")
    print("========================")
    
    # Sample prompts for testing
    test_prompts = [
        "The future of artificial intelligence is",
        "Climate change is a global challenge that requires",
        "In the world of technology, innovation drives",
        "Machine learning algorithms have revolutionized",
        "The importance of sustainable development cannot"
    ]
    
    # Initialize comparison runner
    try:
        print("Initializing models...")
        runner = ComparisonRunner(model_name="gpt2")
        
        # Run comparison
        print("\nStarting comparison...")
        results = runner.run_comparison(
            prompts=test_prompts,
            max_length=80,
            temperature=0.8,
            num_runs=2  # Reduce for faster testing
        )
        
        # Analyze results
        runner.analyze_results()
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install torch transformers psutil matplotlib seaborn pandas")


if __name__ == "__main__":
    main()
