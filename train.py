import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from tqdm import tqdm
import math

from utils import create_lstm
from config import get_config
from dataset import ShakespeareDataset, create_datasets, create_dataloaders
from load_data import tokenize_sequence, detokenize_sequence

def train(m=None):
    cfg = get_config()
    # create dataloaders
    dataset = ShakespeareDataset(annotations_file=cfg["annotations_file"], max_length=cfg["max_length"])
    train_data, test_data = create_datasets(dataset)
    train_dataloader, test_dataloader = create_dataloaders(train_data, test_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_lstm(config=cfg, device=device)
    model.to(device)
    if m is not None:
        m_data = torch.load(f=m, map_location=device, weights_only=True)
        
        model_state_dict = m_data["model_state_dict"]
        
        model.load_state_dict(model_state_dict)

        optimizer = torch.optim.Adam(model.parameters(), lr=m_data["learning_rate"][0]) 
        optimizer.load_state_dict(m_data["optimizer_state_dict"])

        new_max_lr = m_data["learning_rate"][0] + 0.03e-5
        scheduler_total_steps = len(train_dataloader)*cfg["num_epochs"]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=new_max_lr, total_steps=scheduler_total_steps, pct_start=0.3)
        scheduler_state_dict = m_data["scheduler_state_dict"]
        scheduler_state_dict["total_steps"] = scheduler_state_dict["total_steps"] + scheduler_total_steps
        scheduler.load_state_dict(scheduler_state_dict)
        start_epoch = m_data["epoch"]
        lr = m_data["learning_rate"]
        try:
            loss = m_data["loss"]
        except KeyError:
            loss = float('inf')
    else:
        start_epoch = 0
        lr=1e-4
        loss = float('inf')

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, pct_start=0.3, total_steps=cfg["epochs"]*int(len(train_dataloader)))

    end_epochs = start_epoch + cfg["num_epochs"]

    loss_fn = nn.CrossEntropyLoss()
        
    best_loss = loss

    # Print some information about the device
    print(f"Using: {device}")
    print(f"Epochs to train: {cfg['num_epochs']}")
    print(f"Learning rate: {lr}")
    print(f"Previous loss: {loss}")
    if m is not None:
        print(f"Model - {m} loaded successfully !")

    for epoch in range(start_epoch, end_epochs):
        batch_iterator = tqdm((train_dataloader), f"Epoch: {epoch}")
        model.train()

        final_loss = 0
        step = 0

        hidden = None
        for batch_idx, batch in enumerate(batch_iterator):
            src_input = batch["sequence"].to(device)
            tgt_output = batch["next_character"].to(device)

            if hidden is None or hidden[0].shape[1] != src_input.size(0):
                hidden = model.init_hidden(src_input.size(0))

            logits, hidden = model(src_input, hidden)
            logits = logits.view(-1, logits.size(-1)) # [batch_size, vocab_size]
            tgt_output = tgt_output.view(-1)

            # Detach the state for backpropagation
            if hidden is not None:
                hidden = tuple(h.detach() for h in hidden)


            loss = loss_fn(logits, tgt_output)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            scheduler.step()

            batch_iterator.set_postfix({f"Loss" : f"{loss.item():.2f}"})
            step += 1
            final_loss += loss.item()

        final_loss /= step
        print("-"*50)
        print(f"Epoch: {epoch} | Average Loss: {final_loss}")
        # Save the model after every epoch
        if final_loss < best_loss:
            best_loss = final_loss
            model_name = f"me{epoch}l{math.floor(best_loss * 100)}.pth"
            torch.save(
                obj={
                       "model_state_dict" : model.state_dict(),
                       "optimizer_state_dict" : optimizer.state_dict(),
                       "scheduler_state_dict" : scheduler.state_dict(),
                        "learning_rate" : scheduler.get_last_lr(),
                        "epoch" : epoch,
                        "loss" : best_loss
                   },
                f=f"./models_8/{model_name}"
            )
            print(f"Model successfully saved in the `models/{model_name}`")

def beam_search(sequence, model, device, max_length, beam_width):
    """
    Implements beam search for an LSTM model.

    Args:
        model: The trained LSTM model
        sequence: Starting sequence tensor of shape [1, seq_length]
        beam_width: Number of best sequences to keep at each step
        max_length: Maximum length of generated sequence
        device: Device of the tensors

    Returns:
        List of (sequence, score) tuples, sorted by score
    """
    beams = [
        (sequence, 0.0, None)
    ]

    for _ in range(max_length):
        candidates = []

        for seq, score, prev_hidden in beams:
            seq = seq.to(device)
            with torch.inference_mode():
                # Do the forward pass
                # print(prev_hidden)
                if prev_hidden is None:
                    logits, hidden_state = model(seq, None)
                else:
                    logits, hidden_state = model(seq[:, -1:], prev_hidden)

            probs = F.softmax(logits[:, -1], dim=-1)

            top_probs, top_indices = torch.topk(probs, beam_width)

            # print(top_probs)
            # print(top_indices)

            for probs, idx in zip(top_probs[0], top_indices[0]):
                # print(seq.shape, seq)
                # print(idx, idx.view(1, 1))
                new_seq = torch.cat([seq.view(1, -1), idx.view(1, 1)], dim=1)

                new_score = score + torch.log(probs).item()
                candidates.append((new_seq, new_score, hidden_state))

        candidates.sort(reverse=True, key=lambda x: x[1])
        beams = candidates[:beam_width]

    return [(seq_len, score) for seq_len, score, _ in beams]

def translate(sequence : str, model_path : str, response_max_length : int, device : torch.device, number_of_beam : int = 0):
    # Load the model data=
    m_data = torch.load(f=model_path, map_location=device, weights_only=True)
    model_state_dict = m_data["model_state_dict"]

    # Create a model
    cfg = get_config()

    model = create_lstm(config=cfg, device=device)

    model.to(device)

    model.load_state_dict(model_state_dict)

    max_length = response_max_length

    # tokenize the sequence
    t_sequence = torch.tensor(tokenize_sequence(sequence))
    t_sequence = t_sequence.unsqueeze(1)

    result = beam_search(
        sequence=t_sequence,
        model=model,
        device=device,
        max_length=max_length,
        beam_width=3
    )
    score = result[number_of_beam][1]
    result = result[number_of_beam][0].squeeze().cpu().numpy()
    result = detokenize_sequence(result)
    result = "".join(f"{char}" for char in result)

    return result, score

if __name__ == "__main__":
    # train(m="models_8/me27l95.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result, score = translate(
        sequence="King:\nThe Scotch Highlands",
        response_max_length=1000,
        model_path="./models_8/me91l32.pth",
        number_of_beam=0,
        device=device
    )
    print(result + "-"*50 + f"\nScore: {score}")

