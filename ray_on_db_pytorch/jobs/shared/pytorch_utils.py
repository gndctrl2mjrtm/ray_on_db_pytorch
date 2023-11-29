from transformers import AutoTokenizer


def tokenize_sentence(batch):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    outputs = tokenizer(
        batch["sentence"].tolist(),
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    outputs["label"] = batch["label"]
    return outputs
