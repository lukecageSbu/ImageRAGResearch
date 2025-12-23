import os
import torch
from dataclasses import dataclass, field

def _get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@dataclass
class FraudDetectionConfig:
    # Paths
    BASE_DIR: str = field(default_factory=lambda: os.environ.get('BASE_DIR', '/Users/darkmaster/Documents/ReserachCodeMuster/Dataset'))
    VECTOR_STORE_PATH: str = "./vectorstore"

    # Model settings
    MODEL_NAME: str = "jinaai/jina-embeddings-v4"
    COLQWEN_MODEL_NAME: str = "vidore/colqwen2-v0.1"
    INVESTIGATOR_MODEL: str = "global.anthropic.claude-opus-4-5-20251101-v1:0"

    # Device settings
    DEVICE: str = field(default_factory=_get_best_device)
    JINA_GPU_INDEX: int = 0
    COLQWEN_GPU_INDEX: int = 0

    # Embedding settings
    SIMILARITY_THRESHOLD: float = 0.25
    EMBEDDING_DIM: int = 128
    MAX_PATCHES: int = 200

    # Retrieval settings
    DEFAULT_TOP_K: int = 15
    DEFAULT_TOP_K_PER_SECTION: int = 3

    # AWS settings
    AWS_REGION: str = "us-east-1"

    @property
    def DATA_DIR(self):
        return self.BASE_DIR

    @property
    def TRAIN_DATA_PATH(self):
        return os.path.join(self.DATA_DIR, 'train.txt')

    @property
    def VALIDATION_DATA_PATH(self):
        return os.path.join(self.DATA_DIR, 'val.txt')

    @property
    def TEST_DATA_PATH(self):
        return os.path.join(self.DATA_DIR, 'test.txt')

    def get_device(self, model_type: str = "jina") -> str:
        if self.DEVICE != "cuda":
            return self.DEVICE
        gpu_index = self.JINA_GPU_INDEX if model_type == "jina" else self.COLQWEN_GPU_INDEX
        return f"cuda:{gpu_index}"

    def get_bedrock_client(self):
        import boto3
        return boto3.client('bedrock-runtime', region_name=self.AWS_REGION)

