import numpy as np

class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        embeddings=  self.langchain_embeddings.embed_documents(input)
        return np.array(embeddings, dtype=np.float32)