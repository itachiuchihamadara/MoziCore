from MoziCore.tokenization.LearnableTokenizer import Embedder
import torch
import tiktoken

class InputEmbeddings:

    embedder = None
    tokenizer = tiktoken.get_encoding("gpt2")

    def __init__(self):
        self.embedder = Embedder(vocab_size = 50257, embed_dim = 128)

    def get_embeddings(self, input_string):
        in_ids = torch.tensor(self.tokenizer.encode(input_string.lower()))
        return self.embedder(in_ids)

    def get_embeddings_from_ids(self, input_ids):
        in_ids = torch.tensor(input_ids)
        return self.embedder(in_ids)

    def get_ids_from_list_of_strings(self, input_strs):
        input_ids = []
        for string in input_strs:
            encoded = self.tokenizer.encode(string.lower())
            input_ids.extend(encoded)
        return torch.tensor(input_ids)

    def decode_id(self, id):
        return self.tokenizer.decode([id])

    def token_ids_to_text(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def tokens_to_indices(self, string):
        return self.tokenizer.encode(string)

    def get_text_from_id(self, id):
        return self.tokenizer.decode([id])

    def get_embedder(self):
        return self.embedder

    def set_embedder(self, new_embedder):
        self.embedder = new_embedder




if __name__ == "__main__":
    tokenizer = InputEmbeddings()
    # txt1 = "SELECT * FROM customer"
    # emb = tokenizer.get_embeddings(txt1)
    # print(emb[0])

    encoded_select = tokenizer.get_embeddings("SELECT")
    # print(f"Encoded SELECT: {encoded_select}")

    f = tokenizer.get_embeddings("SELECT")
    t= tokenizer.get_embeddings("select")

    print(f, end='\n')
    print(t, end='\n')

    decoded_select = tokenizer.decode_id(encoded_select)
    # print(f"Decoded SELECT: {decoded_select}")

