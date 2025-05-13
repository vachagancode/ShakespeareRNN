import csv

text = open("tiny_shakespeare.txt").read()
text = text.lower()

chars = sorted(set(text))
# print(chars)

char2idx = {char: i for i, char in  enumerate(chars)}
idx2char = {i: char for i, char in enumerate(chars)}

special_tokens = {
    "<s>": 100,
    "<p>": 200,
    "<e>": 300
}
char2idx.update(special_tokens)

seq_len = 100

sequences = []
next_chars = []

def create_csv(data, file_name="annotations_file.csv"):
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        fields = ["sequence", "next_character"]

        writer.writerow(fields)

        for d in data:
            writer.writerow([d["sequence"], d["next_char"]])

    print(f"File data loaded successfully !")


def form_data(text, seq_len : int = 100):
    data = []

    for i in range(0, len(text), 1):
        try:
            sequence = text[i: i + seq_len]
            next_char = text[i + seq_len]
            data.append(
                {
                    "sequence" : sequence,
                    "next_char" : next_char
                }
            )
        except IndexError:
            break

    return data

def tokenize_sequence(sequence):
    tokenized_sequence = [char2idx[i] for i in sequence.lower()]

    return tokenized_sequence

def detokenize_sequence(sequence):
    detokenized_sequence = [idx2char[i] for i in sequence]

    return detokenized_sequence

if __name__ == "__main__":
    # print(detokenize_sequence([1, 2, 3, 5, 8, 9, 13]))
    pass



