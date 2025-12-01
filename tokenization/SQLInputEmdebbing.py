from MoziCore.tokenization.LearnableTokenizer import Embedder
import re
import torch

class QueryInputEmbeddings:

    embedder = None
    vocab = None

    def __init__(self, text_file="C:\\Project\\PythonEngine\\MoziCore\\train_files\\SQLQueries.txt"):
        with open(text_file, "r", encoding="utf-8") as f:
            self.raw_text = f.read()
        preprocessed = re.split(r'([,.:;?!"()\']|--|\s)', self.raw_text)
        preprocessed = [item.strip().lower() for item in preprocessed if item.strip()]
        all_words = sorted(set(preprocessed))
        vocab_size = len(all_words)
        print("Vocabulary Size:", vocab_size)
        self.vocab = {token: integer for integer, token in enumerate(all_words)}
        self.reverse_dict = {id: token for token, id in self.vocab.items()}
        self.embedder = Embedder(vocab_size=vocab_size, embed_dim=128)

    def get_embeddings(self, input_string):
        preprocessed = re.split(r'([,.:;?!"()\']|--|\s)', input_string)
        preprocessed = [item.strip().lower()  for item in preprocessed if item.strip()]
        input_tokens = []
        for tokens in preprocessed:
            input_tokens.append(self.vocab.get(tokens, None))
        return self.embedder(torch.tensor(input_tokens))

    def get_ids_from_list_of_strings(self, input_strs):
        input_ids = [self.tokens_to_indices(string) for string in input_strs]
        return torch.tensor(input_ids)

    def get_ids_from_string(self, input_string):
        preprocessed = re.split(r'([,.:;?!"()\']|--|\s)', input_string)
        preprocessed = [item.strip().lower() for item in preprocessed if item.strip()]
        input_tokens = []
        for tokens in preprocessed:
            input_tokens.append(self.vocab.get(tokens, None))
        return input_tokens

    def decode_id(self, id):
        return self.get_embeddings(self.reverse_dict.get(id, None))

    def get_embeddings_from_ids(self, input_ids):
        in_ids = torch.tensor(input_ids)
        return self.embedder(in_ids)

    def get_text_from_id(self, id):
        return self.reverse_dict.get(id, None)

    def token_ids_to_text(self, token_ids):
        tokens = [self.get_text_from_id(id) for id in token_ids]
        return " ".join(tokens)

    def tokens_to_indices(self, string):
        return self.vocab.get(string.lower(), None)

    def get_embedder(self):
        return self.embedder

    def set_embedder(self, new_embedder):
        self.embedder = new_embedder

# import torch
# input_embedder = InputEmbeddings()
# token_index = input_embedder.tokens_to_indices("SELECT")
# print("Token index for 'SELECT':", token_index)
# input_tokens = torch.tensor([[token_index]])
# embeddings = input_embedder.get_embeddings(input_tokens)
# print("Embeddings shape:", embeddings.shape)

