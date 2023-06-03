import json

READSPEECH_DIR = "/data/akee511/datasets/data_readspeech_am/"


def main():
    with open(f"{READSPEECH_DIR}/lang/nonsilence_phones.txt") as fid:
        phones = [line.strip() for line in fid]
    phones = ["<pad>", "<s>", "</s>", "<unk>", " "] + phones
    phone_dict = {p: i for i, p in enumerate(phones)}

    with open("vocab.json", "w") as fid:
        json.dump(phone_dict, fid)


if __name__ == "__main__":
    main()
