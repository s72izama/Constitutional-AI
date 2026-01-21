import os
import json
from datasets import load_dataset

class DataManager:
    def __init__(self, output_path="data/seed_prompts.json"):
        base_dir = os.path.dirname(os.path.dirname(__file__))  # project root
        self.output_path = base_dir + "/" + output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def prepare(self, count=100, force_download=False):
        """
        Runs once to ensure the seed dataset is ready.
        """
        if os.path.exists(self.output_path) and not force_download:
            print(f"[*] {self.output_path} already exists. Skipping download.")
            return

        print(f"[!] Preparing seed prompts. Download=True (force_redownload if requested)")
        
        # load_dataset manages caching automatically. 
        # force_redownload ensures we fetch fresh data if needed.
        dataset = load_dataset(
            "declare-lab/HarmfulQA", 
            split="train",
            download_mode="force_redownload" if force_download else "reuse_dataset_if_exists"
        )
        
        subset = dataset.select(range(min(count, len(dataset))))
        
        seed_data = []
        for item in subset:
            seed_data.append({
                "id": item.get('id'),
                "category": item.get('category'),
                "prompt": item.get('question')
            })

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(seed_data, f, indent=4)
        
        print(f"Seed prompts saved to {self.output_path}")

    def load_local_prompts(self):
        """Helper to load the file once it exists."""
        with open(self.output_path, "r") as f:
            return json.load(f)