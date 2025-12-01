import torch

# from MoziCore.tokenization.InputEmbedings import InputEmbeddings
from MoziCore.tokenization.SQLInputEmdebbing import QueryInputEmbeddings as InputEmbeddings
from MoziCore.GPT import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 364, #50257, #259
    "context_length": 256,
    "emb_dim": 128,
    "n_heads": 2,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}



def generate_text_simple(model, idx, max_new_tokens, context_size, tokenizer=None):
    ids =[]
    for run in range(max_new_tokens):
        # batch_size, seq_len, _ = idx.size()
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            raw_out = model(idx_cond)

        logits = raw_out[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        idx_next = idx_next[0][0].item()
        text = tokenizer.get_text_from_id(idx_next)
        if text == ";": break

        ids.append(idx_next)

        idx_next =  tokenizer.decode_id(idx_next).unsqueeze(0)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)
    return ids


# if __name__ == "__main__":
#
#     tokenizer = InputEmbeddings()
#
#     txt1 = "SELECT * FROM ANAGRAFICA_STANZE"
#     targets = ['*', 'FROM', 'ANAGRAFICA_STANZE', 'as']
#
#     batch = [tokenizer.get_embeddings(txt1), tokenizer.get_embeddings(txt1)]
#     batch = torch.stack(batch, dim=0)
#
#     model = GPTModel(GPT_CONFIG_124M)
#     model.eval()
#
#     l = [tokenizer.get_ids_from_list_of_strings(targets), tokenizer.get_ids_from_list_of_strings(targets)]
#     targ_ids = torch.stack(l, dim=0)
#
#     logits = generate_text_simple(model, batch, max_new_tokens = 4, context_size = GPT_CONFIG_124M["context_length"])
#     logits.flatten(0, 1)
#     targ_ids.flatten()
#
#     print(logits.shape)
#     print(targ_ids.shape)

    # total_batch_loss = torch.nn.functional.cross_entropy(logits, targ_ids)
    # print("Loss:", total_batch_loss.item())