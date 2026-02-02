import torch
import vllm
from vllm import LLM, SamplingParams

def verify_cluster_setup():
    print(f"--- Software Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"vLLM Version:    {vllm.__version__}")
    print(f"Nodes available: {torch.cuda.device_count()} GPUs")

    # Dummy load
    try:
        print(f"\n--- Initializing Dummy 14B Model ---")
        # We use the Qwen2.5-14B architecture as the template
        llm = LLM(
            model="Qwen/Qwen2.5-14B-Instruct", 
            load_format="dummy",      
            tensor_parallel_size=1,   
            enforce_eager=True,       
        )
        
        print("\n[SUCCESS] vLLM successfully reserved VRAM and loaded kernels.")
        
    except Exception as e:
        print(f"\n[FAILURE] vLLM could not initialize: {e}")

if __name__ == "__main__":
    verify_cluster_setup()