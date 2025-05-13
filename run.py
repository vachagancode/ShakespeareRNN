import sys

import torch
import time
import spacy

from utils import prettify_the_output
from train import translate

def console_run(model_path : str):
    welcome_text = """<!--                                                  -->
    <!-- |_   _|__   | |__   ___    ___  _ __   _ __   ___ | |_  | |_ ___   -->
    <!--   | |/ _ \  | '_ \ / _ \  / _ \| '__| | '_ \ / _ \| __| | __/ _ \  -->
    <!--  _| | (_) | | |_) |  __/ | (_) | |    | | | | (_) | |_  | || (_) | -->
    <!-- | |_|\___/_ |_.__/ \___|  \___/|_|    |_| |_|\___/ \__|  \__\___/  -->
    <!-- | '_ \ / _ \                                                       -->
    <!-- | |_) |  __/                                                       -->
    <!-- |_.__/ \___|                                                       -->"""
    print(welcome_text)
    print("<!                                                                      -->")
    nlp = spacy.load("en_core_web_sm")
    while True:
        print('-' * 70)
        print("Genspeare: Input a beginning for the Genspeare a wait to see the magic happen\nGenspeare: If you want to quit simply input `q`")
        user_input = input("Your input: ")
        if user_input == "q":
            break
        else:
            print("Genspeare: Okay, give me some time to think about it...")
            # Set up the device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            output = translate(
                sequence=user_input,
                model_path="models/me69l67.pth",
                response_max_length=1000,
                device=device
            )

            result = prettify_the_output(output[0], nlp)
            for char in result:
                sys.stdout.write(char)
                sys.stdout.flush()

                delay = 0.05
                if char in [".", ",", "!", "?", ";", ":"]:
                    delay = 0.1
                time.sleep(delay)



def compute_result(model_path : str, prompt : str):
    nlp = spacy.load("en_core_web_sm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output = translate(
        sequence=prompt,
        model_path=model_path,
        response_max_length=5,
        device=device
    )
    result = prettify_the_output(output[0], nlp)
    return result

if __name__ == "__main__":
    # print(compute_result("./models/me69l67.pth", "Hey, paronner, akanj areq taparakan ashuxin..."))
    console_run("./models/me91l32.pth")