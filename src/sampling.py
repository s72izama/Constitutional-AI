
class SamplingParams:
    def __init__(
        self,
        max_tokens=200,
        temperature=0.2,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        n=1,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.n = n
