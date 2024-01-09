from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        model = "tiiuae/falcon-180b"
        start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipeline = transformers.pipeline(
           "text-generation",
           model=model,
           tokenizer=self.tokenizer,
           torch_dtype=torch.bfloat16,
           trust_remote_code=True,
           device_map="auto",
        )
        logger.info(f"Setup took {time.time() - start} seconds")

    def predict(
        self,
        prompt: str = Input(
            description=f"Input Prompt.",
            default="Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
        ),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=500,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1.2,
        ),
        top_k: int = Input(
            description="Valid if you choose top_k decoding. The number of highest probability vocabulary tokens to keep for top-k-filtering",
            default=50,
        ),
    ):
        start = time.time()
        sequences = pipeline(
           prompt,
           max_length=max_length,
           do_sample=True,
           top_k=10,
           num_return_sequences=1,
           eos_token_id=self.tokenizer.eos_token_id,
        )
        logger.info(f"Inference took {time.time() - start} seconds")
        return [text["generated_text"] for text in sequences].join("")
        

