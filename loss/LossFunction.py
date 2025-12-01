import torch

def calc_loss_batch(input_batch, target_batch, model, device, tokenizer):

    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    outer = []
    for row in input_batch:
        inner = []
        for id in row:
            inner.append(tokenizer.decode_id(id.item()))
        outer.append(torch.cat(inner, dim=0))
    input_batch = torch.stack(outer, dim=0)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, tokenizer, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device, tokenizer)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches