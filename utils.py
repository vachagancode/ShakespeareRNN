import re

import torch.cuda
import spacy
from models import LSTM

def prettify_the_output(model_output, nlp): # detokenized
    shakespeare_titles = {'king', 'queen', 'duke', 'prince', 'lord', 'lady', 'sir', 'madam', 'mistress'}
    roman_numbers = ["i", "ii", "iii", "iv", "v", 'vi', "vii", "viii", "ix", "x", "l"]
    # Split into lines while preserving empty lines
    lines = model_output.split('\n')
    capitalized_lines = []

    for line in lines:
        if not line.strip():
            capitalized_lines.append(line)
            continue

        doc = nlp(line)
        words = []

        is_character_name = ":" in line and line.strip().endswith(":")

        for i, token in enumerate(doc):
            word = token.text

            if (
                i == 0 or
                token.is_sent_start or
                token.pos_ == "PROPN" or
                word.lower() == "i" or
                (word.lower() in shakespeare_titles and (i == 0 or doc[i-1].text == "the")) or
                is_character_name
            ): word = word.capitalize()

            if (
                word.lower() in roman_numbers
            ): word = word.upper()

            words.append(word)

        # Join words back together
        new_line = " ".join(words)

        new_line = re.sub(r'\s+([.,?!:;])', r'\1', new_line)

        capitalized_lines.append(new_line)

    return "\n".join(capitalized_lines)

def create_lstm(config, device, layers : int = 8):
    lstm = LSTM(
        input_size = config["input_size"],
        hidden_size = config["hidden_size"],
        num_layers = config["num_layers"],
        vocab_size = config["vocab_size"],
        dropout = config["dropout"],
        device = device
    )

    return lstm

def reform_the_model(model_path : str, output_name : str):
    """
    This function simply takes the trained model and removes other useless parameters for predicting like optimizer_state_dict, scheduler_state_dict, learning_rate
    :model_path: str
    :output_name: str
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f=model_path, map_location=device)
    model_state_dict = model["model_state_dict"]

    torch.save(
        obj={
            "model_state_dict" : model_state_dict
        },
        f=f"models/{output_name}"

    )

if __name__ == "__main__":
    reform_the_model(
        "./models_8/me22l100.pth", "me22l100.pth"
        )