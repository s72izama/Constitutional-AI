from model_wrapper import ModelWrapper


class ConstitutionalCritic:
    def __init__(self, constitution, model_wrapper: ModelWrapper):
        self.constitution = constitution
        self.model = model_wrapper

    def generate_initial(self, user_prompt, sampling_params):
        messages = [
            {"role": "system", "content": self.constitution},
            {"role": "user", "content": user_prompt},
        ]
        response_text = self.model.generate(messages, sampling_params)
        assistant_text = response_text[0].split("assistant\n", 1)[-1].strip()
        print(assistant_text)
        return assistant_text

    def critique_response(self, user_prompt, response, sampling_params):
        """Critique the response based on constitutional principles."""
        critique_prompt = f"""Review the following response and identify any ways it could be improved to better align with the constitutional principles.

            Original user request: {user_prompt}

            Response to critique:
            {response}

            Provide specific suggestions for improvement based on the constitution's guidelines for clarity, helpfulness, safety, and honesty. Be concise."""
                
        messages = [
            {"role": "system", "content": self.constitution},
            {"role": "user", "content": critique_prompt}
        ]
        response_text = self.model.generate(messages, sampling_params)
        assistant_text = response_text[0].split("assistant\n", 1)[-1].strip()
        print(response_text)
        return assistant_text

    def revise_response(self, user_prompt, original_response, critique, sampling_params):
        """Revise the response based on the critique."""
        revision_prompt = f"""Based on the following critique, provide an improved response to the original user request.

        Original user request: {user_prompt}

        Original response:
        {original_response}

        Critique:
        {critique}

        Provide the revised response that addresses the critique while following all constitutional principles:"""
        
        messages = [
            {"role": "system", "content": self.constitution},
            {"role": "user", "content": revision_prompt}
        ]
        response_text = self.model.generate(messages, sampling_params)
        assistant_text = response_text[0].split("assistant\n", 1)[-1].strip()
        print(response_text)
        return assistant_text
