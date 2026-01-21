from config import CONSTITUTION_FILE_PATH, OUTPUT_PATH
from data_manager import DataManager
from model_wrapper import ModelWrapper
from constitutional_critic import ConstitutionalCritic
from constitutional_ai_pipeline import ConstitutionalAIPipeline
from sampling import SamplingParams
from datetime import datetime
import os

def main():
    # -------------------------------------------------
    # Load constitution
    # -------------------------------------------------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    constitution_file = os.path.join(base_dir, CONSTITUTION_FILE_PATH)
    if os.path.exists(constitution_file):
        with open(constitution_file, "r", encoding="utf-8") as f:
            constitution = f.read()
        print(f"[INFO] Loaded constitution from {constitution_file}")
    else:
        print(f"[WARNING] Constitution file not found at {constitution_file}. Using default system prompt.")
        constitution = "You are a helpful, harmless, and honest AI assistant."

    # -------------------------------------------------
    # Load model + tokenizer (via ModelWrapper)
    # -------------------------------------------------
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct"
    model_wrapper = ModelWrapper(
        model_name_or_path=model_name_or_path,
        cache_dir="./cache",
        tensor_parallel_size= 1,
        gpu_memory_utilization= 0.9,
    )

    # -------------------------------------------------
    # Sampling parameters
    # -------------------------------------------------
    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=0.2,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
    )


    # -------------------------------------------------
    # CAI components
    # -------------------------------------------------
    critic = ConstitutionalCritic(
        constitution=constitution,
        model_wrapper=model_wrapper,
    )

    enable_critique=True
    max_revisions=1
    pipeline = ConstitutionalAIPipeline(
        critic=critic,
        sampling_params=sampling_params,
        enable_critique=enable_critique,
        max_revisions=max_revisions,
    )

    # -------------------------------------------------
    # Output setup
    # -------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output_{timestamp}.jsonl"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_path = os.path.join(base_dir, OUTPUT_PATH, output_file)

    if not os.path.exists(output_path):
        open(output_path, "w").close()

    print("[INFO] Starting Constitutional AI generation...")
    print(f"[INFO] Model: {model_name_or_path}")
    print(f"[INFO] Critique enabled: {enable_critique}")
    print(f"[INFO] Max revisions: {max_revisions}")

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------
    data_mgr = DataManager()
    data_mgr.prepare(count=1,force_download=False)

    dataset = data_mgr.load_local_prompts()
    print(f"[INFO] Loaded dataset with {len(dataset)} samples.")

    # -------------------------------------------------
    # Process samples
    # -------------------------------------------------
    for data in enumerate(dataset):
        sample = data[1]["prompt"]
        print(f"[CAI] Processing sample {sample}...")

        # Construct user prompt
        user_prompt = (
            f"""Please provide a helpful, factual answer to the following question.

            Question:
            {sample}"""
        )

        cai_result = pipeline.run(user_prompt)

        pipeline.save_constitutional_output(
            user_prompt=user_prompt,
            cai_result=cai_result,
            output_file=output_path,
        )

    print("[INFO] Constitutional AI generation completed.")

if __name__ == "__main__":
    main()

