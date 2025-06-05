from datasets import load_dataset
from deploy.deployer import MODEL_PATH
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from functools import partial
import joblib
import torch
import requests
import json


DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 5


def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], padding=True, truncation=True)


def train(dataset_path, batch_size, num_epochs):
    # Load your own dataset (or make a dummy one)
    dataset = load_dataset("csv", data_files={"train": dataset_path}, split="train")

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset["label"])
    dataset = dataset.map(lambda x: {"label": label_encoder.transform([x["label"]])[0]})

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer_helper = partial(
        tokenize,
        tokenizer=tokenizer
    )
    dataset = dataset.map(tokenizer_helper, batched=True)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_encoder.classes_)
    )

    # Trainer
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        output_dir=MODEL_PATH,
        save_strategy="no"
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    # Train
    trainer.train()

    # Save model, tokenizer, and label encoder
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    joblib.dump(label_encoder, f"{MODEL_PATH}/label_encoder.joblib")


def query(line, deployed=False):
    if deployed:
        response = requests.post(
            "https://model.jenet.ai/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"text": line})
        )
        if response.status_code == 200:
            return response.json()[0].get("label")
        else:
            raise RuntimeError(f"API request failed: {response.status_code} {response.text}")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    label_encoder = joblib.load(f"{MODEL_PATH}/label_encoder.joblib")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Tokenize input line
    inputs = tokenizer([line], padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

    # Decode prediction to label name
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    return predicted_label
