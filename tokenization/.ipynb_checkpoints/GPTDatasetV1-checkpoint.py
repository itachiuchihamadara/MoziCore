import torch
from lxml.html.diff import token
from torch.distributed.checkpoint import load_state_dict
from torch.utils.data import DataLoader, TensorDataset
from MoziCore.tokenization.SQLInputEmdebbing import QueryInputEmbeddings as InputEmbeddings
from MoziCore.tokenization.LearnableTokenizer import Embedder
from MoziCore.loss.LossFunction import calc_loss_batch, calc_loss_loader
from MoziCore.GPT import GPTModel
from MoziCore.TextGen import generate_text_simple

def create_dataloader_v1(txt, tokenizer, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    with open(txt, "r", encoding="utf-8") as f:
        train_data = f.read()

    batch_input_tensors = []
    batch_target_tensors = []

    for lines in train_data.splitlines():
        token_ids = tokenizer.get_ids_from_string(lines)
        if len(token_ids) <= max_length:
            continue

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            batch_input_tensors.append(torch.tensor(input_chunk))
            batch_target_tensors.append(torch.tensor(target_chunk))

    batch_input_tensors = torch.stack(batch_input_tensors, dim=0)
    batch_target_tensors = torch.stack(batch_target_tensors, dim=0)

    dataset_ = TensorDataset(batch_input_tensors, batch_target_tensors)
    return DataLoader(dataset_, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


GPT_CONFIG_124M = {
    "vocab_size": 367,
    "context_length": 15,
    "emb_dim": 128,
    "n_heads": 2,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device, tokenizer)  # Compute loss for current batch
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, tokenizer, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, tokenizer, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size, tokenizer=tokenizer
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def text_to_token_ids(text, tokenizer):
    encoded_ids = tokenizer.get_embeddings(text)
    return encoded_ids.unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.token_ids_to_text(token_ids)

if __name__ == "__main__":

    # tokenizer = InputEmbeddings()
    # train_loader = create_dataloader_v1(
    #     txt="C:\\Project\\PythonEngine\\MoziCore\\train_files\\SQLQueries.txt",
    #     tokenizer=tokenizer,
    #     batch_size=2,
    #     max_length=GPT_CONFIG_124M["context_length"],
    #     stride=1,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=0
    # )
    # val_loader = create_dataloader_v1(
    #     txt="C:\\Project\\PythonEngine\\MoziCore\\train_files\\SQLQueries_Val.txt",
    #     tokenizer=tokenizer,
    #     batch_size=2,
    #     max_length=GPT_CONFIG_124M["context_length"],
    #     stride=1,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=0
    # )
    # model = GPTModel(GPT_CONFIG_124M)
    # device = torch.device("cpu")
    # torch.manual_seed(123)
    # model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    #
    # num_epochs = 10
    # train_losses, val_losses, tokens_seen = train_model_simple(
    #     model, train_loader, val_loader, optimizer, device,
    #     num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    #     start_context="get smart card code for hotel code :hotelCode THETA ", tokenizer=tokenizer
    # )
    #
    # torch.save(model.state_dict(), "model.pth")

    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cpu")
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
    model.eval()

    tokenizer = InputEmbeddings()
    embedder = Embedder(vocab_size=GPT_CONFIG_124M["vocab_size"], embed_dim=GPT_CONFIG_124M["emb_dim"])
    embedder.load_state_dict(torch.load("embedder.pth", map_location=device, weights_only=True))
    embedder.eval()

    tokenizer.set_embedder(embedder)

    ids = generate_text_simple(
        model=model, idx=text_to_token_ids("delete smart card  ", tokenizer),
        max_new_tokens=50, context_size=GPT_CONFIG_124M["context_length"], tokenizer=tokenizer
    )
    print(token_ids_to_text(ids, tokenizer))






