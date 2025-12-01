import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_ids):
        # input_ids are token indices (equivalent to T in one-hot form)
        embeddings = self.embedding(input_ids)
        return embeddings
