from constitutional_critic import ConstitutionalCritic

class ConstitutionalAIPipeline:
    def __init__(
        self,
        critic: ConstitutionalCritic,
        sampling_params,
        enable_critique=True,
        max_revisions=1,
    ):
        self.critic = critic
        self.sampling_params = sampling_params
        self.enable_critique = enable_critique
        self.max_revisions = max_revisions

    def run(self, user_prompt):
        print("[CAI] Generating initial response...")
        initial_response = self.critic.generate_initial(
            user_prompt, self.sampling_params
        )

        if not self.enable_critique:
            return {
                "initial_response": initial_response,
                "final_response": initial_response,
                "critiques": [],
                "revisions": [],
            }

        current_response = initial_response
        critiques, revisions = [], []

        for i in range(self.max_revisions):
            print(f"[CAI] Critique iteration {i + 1}/{self.max_revisions}")
            critique = self.critic.critique_response(
                user_prompt, current_response, self.sampling_params
            )
            critiques.append(critique)

            print(f"[CAI] Revision iteration {i + 1}/{self.max_revisions}")
            revised = self.critic.revise_response(
                user_prompt, current_response, critique, self.sampling_params
            )
            revisions.append(revised)

            current_response = revised

        return {
            "initial_response": initial_response,
            "final_response": current_response,
            "critiques": critiques,
            "revisions": revisions,
        }
    
    def save_constitutional_output(self, user_prompt, cai_result, output_file):
        import json, os

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        record = {
            "user_prompt": user_prompt,
            "cai_result": cai_result,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

