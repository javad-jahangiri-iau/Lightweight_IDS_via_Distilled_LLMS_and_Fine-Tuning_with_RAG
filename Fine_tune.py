import os
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from rich.table import Table
from rich.console import Console
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import logging as transformers_logging
import pickle
from PIL import Image
from datetime import datetime

transformers_logging.set_verbosity_error()

# Configuration
EPOCHS = 3
BATCH_SIZE = 8
MAX_LEN = 128
LEARNING_RATE = 5e-5
DATA_PATH = Path("data")
BASE_RESULTS_PATH = Path("results")
LOG_PATH = Path("logs")
DATA_FRACTION = 0.1  # Using 0.1 as it is the maximum value among provided files

# Define model and dataset combinations
MODEL_DATASET_CONFIGS = [
    {"model_name": "distilbert-base-uncased", "dataset_name": "CICIDS2017", "data_fraction": 0.01},
    {"model_name": "distilbert-base-uncased", "dataset_name": "NSL-KDD", "data_fraction": 0.01},
    {"model_name": "distilbert-base-uncased", "dataset_name": "UNSW", "data_fraction": 0.01},
    {"model_name": "distilbert-distilroberta-base", "dataset_name": "NSL-KDD", "data_fraction": 0.01},
    {"model_name": "distilbert-distilroberta-base", "dataset_name": "CICIDS2017", "data_fraction": 0.01},
    {"model_name": "distilbert-distilroberta-base", "dataset_name": "UNSW", "data_fraction": 0.01},
    {"model_name": "google-mobilebert-uncased", "dataset_name": "NSL-KDD", "data_fraction": 0.1},
    {"model_name": "google-mobilebert-uncased", "dataset_name": "CICIDS2017", "data_fraction": 0.1},
    {"model_name": "google-mobilebert-uncased", "dataset_name": "UNSW", "data_fraction": 0.1},
    {"model_name": "huawei-noah-TinyBERT_General_4L_312D", "dataset_name": "NSL-KDD", "data_fraction": 0.1},
    {"model_name": "huawei-noah-TinyBERT_General_4L_312D", "dataset_name": "CICIDS2017", "data_fraction": 0.1},
    {"model_name": "huawei-noah-TinyBERT_General_4L_312D", "dataset_name": "UNSW", "data_fraction": 0.1},
]

# Generate timestamp for results folder and log file
EXECUTION_TIME = datetime.now().strftime("%Y%m%d_%H%M")
RESULTS_PATH = BASE_RESULTS_PATH / EXECUTION_TIME
LOG_FILE = RESULTS_PATH / "run.log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"Using device: {DEVICE}")

# List to store all evaluation metrics
ALL_EVALUATION_METRICS = []


# Function to log configuration parameters
def log_config_parameters():
    logging.info("Configuration Parameters:")
    logging.info(f"  EPOCHS: {EPOCHS}")
    logging.info(f"  BATCH_SIZE: {BATCH_SIZE}")
    logging.info(f"  MAX_LEN: {MAX_LEN}")
    logging.info(f"  LEARNING_RATE: {LEARNING_RATE}")
    logging.info(f"  DATA_PATH: {DATA_PATH}")
    logging.info(f"  BASE_RESULTS_PATH: {BASE_RESULTS_PATH}")
    logging.info(f"  RESULTS_PATH: {RESULTS_PATH}")
    logging.info(f"  LOG_FILE: {LOG_FILE}")
    logging.info(f"  DATA_FRACTION: {DATA_FRACTION}")
    logging.info(f"  MODEL_DATASET_CONFIGS: {MODEL_DATASET_CONFIGS}")


# Dataset loading functions
def load_cicids2017():
    logging.info("Loading CICIDS2017 dataset")
    local_file = DATA_PATH / "CICIDS2017.csv"
    df = pd.read_csv(local_file)
    df = df[df['Label'].notna()]
    return df


def load_nsl_kdd():
    logging.info("Loading NSL-KDD dataset")
    DATA_PATH.mkdir(exist_ok=True)
    local_file = DATA_PATH / "NSL_KDD.csv"
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]
    if local_file.exists():
        df = pd.read_csv(local_file)
        logging.info("Dataset loaded from local path")
    else:
        url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt'
        df = pd.read_csv(url, names=columns)
        df.to_csv(local_file, index=False)
        logging.info("Dataset downloaded and saved")
    return df[df['label'].notna()]


def load_unsw():
    logging.info("Loading UNSW dataset")
    df = pd.read_csv(DATA_PATH / "UNSW.csv")
    df = df[df['Label'].notna()]
    return df


# Preprocessing functions
def preprocess_cicids2017(df, data_fraction, cache_path):
    if cache_path.exists():
        logging.info("Loading cached texts and labels")
        with open(cache_path, 'rb') as f:
            texts, labels = pickle.load(f)
        return texts, labels

    logging.info("Preprocessing CICIDS2017 dataset")
    df = df.sample(frac=data_fraction, random_state=42).reset_index(drop=True)

    features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                'Fwd Packet Length Max', 'Fwd Packet Length Mean',
                'Bwd Packet Length Max', 'Bwd Packet Length Mean',
                'Flow Bytes/s', 'Flow Packets/s']
    df = df[features + ['Label']].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    y = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)

    texts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating text from features"):
        text = (
            f"Flow duration: {row['Flow Duration']:.2f}, "
            f"Fwd packets: {row['Total Fwd Packets']:.2f}, "
            f"Bwd packets: {row['Total Backward Packets']:.2f}, "
            f"Fwd max len: {row['Fwd Packet Length Max']:.2f}, "
            f"Fwd mean len: {row['Fwd Packet Length Mean']:.2f}, "
            f"Bwd max len: {row['Bwd Packet Length Max']:.2f}, "
            f"Bwd mean len: {row['Bwd Packet Length Mean']:.2f}, "
            f"Flow bytes/sec: {row['Flow Bytes/s']:.2f}, "
            f"Flow packets/sec: {row['Flow Packets/s']:.2f}"
        )
        texts.append(text)

    labels = list(y)

    with open(cache_path, 'wb') as f:
        pickle.dump((texts, labels), f)
        logging.info("Cached texts and labels saved")

    return texts, labels


def preprocess_nsl_kdd(df, data_fraction, cache_path):
    if cache_path.exists():
        logging.info("Loading cached texts and labels")
        with open(cache_path, 'rb') as f:
            texts, labels = pickle.load(f)
        return texts, labels

    logging.info("Preprocessing NSL-KDD dataset")
    df = df.sample(frac=data_fraction, random_state=42).reset_index(drop=True)

    features = ['protocol_type', 'service', 'flag', 'duration', 'src_bytes', 'dst_bytes']
    df = df[features + ['label']].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    scaler = StandardScaler()
    df[['duration', 'src_bytes', 'dst_bytes']] = scaler.fit_transform(df[['duration', 'src_bytes', 'dst_bytes']])
    y = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    le_protocol = LabelEncoder()
    le_service = LabelEncoder()
    le_flag = LabelEncoder()
    df['protocol_type_enc'] = le_protocol.fit_transform(df['protocol_type'])
    df['service_enc'] = le_service.fit_transform(df['service'])
    df['flag_enc'] = le_flag.fit_transform(df['flag'])

    X = df[['protocol_type_enc', 'service_enc', 'flag_enc', 'duration', 'src_bytes', 'dst_bytes']]
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_resampled_np = X_resampled.to_numpy()
    protocol_texts = le_protocol.inverse_transform(X_resampled_np[:, 0].astype(int))
    service_texts = le_service.inverse_transform(X_resampled_np[:, 1].astype(int))
    flag_texts = le_flag.inverse_transform(X_resampled_np[:, 2].astype(int))
    duration_texts = [f"{x:.2f}" for x in X_resampled_np[:, 3]]
    src_bytes_texts = [f"{x:.2f}" for x in X_resampled_np[:, 4]]
    dst_bytes_texts = [f"{x:.2f}" for x in X_resampled_np[:, 5]]

    texts = [
        f"Protocol: {p}, Service: {s}, Flag: {f}, Duration: {d}, SrcBytes: {src}, DstBytes: {dst}"
        for p, s, f, d, src, dst in
        zip(protocol_texts, service_texts, flag_texts, duration_texts, src_bytes_texts, dst_bytes_texts)
    ]
    labels = list(y_resampled)

    with open(cache_path, 'wb') as f:
        pickle.dump((texts, labels), f)
        logging.info("Cached texts and labels saved")

    return texts, labels


def preprocess_unsw(df, data_fraction, cache_path):
    if cache_path.exists():
        logging.info("Loading cached texts and labels")
        with open(cache_path, 'rb') as f:
            texts, labels = pickle.load(f)
        return texts, labels

    logging.info("Preprocessing UNSW dataset")
    df = df.sample(frac=data_fraction, random_state=42).reset_index(drop=True)

    features = [
        'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'Sload', 'Dload',
        'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
        'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt',
        'tcprtt', 'synack', 'ackdat'
    ]
    df = df[features + ['Label']].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    y = df['Label'].astype(int)

    texts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating text from features"):
        text = (
            f"Duration: {row['dur']:.2f}, SrcBytes: {row['sbytes']:.2f}, DstBytes: {row['dbytes']:.2f}, "
            f"SrcTTL: {row['sttl']:.2f}, DstTTL: {row['dttl']:.2f}, SLoad: {row['Sload']:.2f}, DLoad: {row['Dload']:.2f}, "
            f"SPkts: {row['Spkts']:.2f}, DPkts: {row['Dpkts']:.2f}, TCP RTT: {row['tcprtt']:.2f}, SYNACK: {row['synack']:.2f}, ACKDAT: {row['ackdat']:.2f}"
        )
        texts.append(text)

    labels = list(y)

    with open(cache_path, 'wb') as f:
        pickle.dump((texts, labels), f)
        logging.info("Cached texts and labels saved")

    return texts, labels


# Common dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        logging.info("Initializing TextDataset")
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN)
        self.labels = torch.tensor(labels, dtype=torch.long)
        logging.info(f"TextDataset created with {len(self.labels)} samples")

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# Function to combine confusion matrix images
def combine_confusion_matrices(results_path, output_filename="combined_confusion_matrices_fine_tune.png"):
    image_files = [f for f in results_path.glob("confusion_matrix_*.png")]
    if not image_files:
        logging.warning("No confusion matrix images found to combine")
        return

    images = [Image.open(f) for f in image_files]
    widths, heights = zip(*(img.size for img in images))

    # Calculate grid dimensions
    num_images = len(images)
    cols = min(num_images, 3)  # Maximum 3 images per row
    rows = (num_images + cols - 1) // cols
    max_width = max(widths)
    max_height = max(heights)

    # Create new image with appropriate size
    total_width = max_width * cols
    total_height = max_height * rows
    combined_image = Image.new('RGB', (total_width, total_height), color='white')

    # Paste images into the grid
    for idx, img in enumerate(images):
        x = (idx % cols) * max_width
        y = (idx // cols) * max_height
        combined_image.paste(img, (x, y))

    # Save combined image
    combined_image.save(results_path / output_filename)
    logging.info(f"Combined confusion matrices saved to {results_path / output_filename}")


# Common training and evaluation function
def train_and_evaluate(texts, labels, model_name, dataset_name):
    PRETRAINED_MODEL_PATH = Path(f"models/pretrained_model/{model_name}")
    SAVE_MODEL_PATH = Path(f"models/finetuned_model/{model_name}_{dataset_name}")
    SAVE_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    if (SAVE_MODEL_PATH / "model.safetensors").exists():
        logging.info(f"Fine-tuned model found at {SAVE_MODEL_PATH}. Skipping fine-tuning.")
        print(f"Fine-tuned model found at {SAVE_MODEL_PATH}. Skipping fine-tuning.")
        model = AutoModelForSequenceClassification.from_pretrained(SAVE_MODEL_PATH).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(SAVE_MODEL_PATH)
    else:
        logging.info("Fine-tuned model not found. Starting fine-tuning process.")
        print("Fine-tuned model not found. Starting fine-tuning process.")

        X_train_val, X_test, y_train_val, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)

        if PRETRAINED_MODEL_PATH.exists():
            model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH).to(DEVICE)
            tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = TextDataset(X_train, y_train, tokenizer)
        val_dataset = TextDataset(X_val, y_val, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        num_training_steps = EPOCHS * len(train_loader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                     num_training_steps=num_training_steps)

        model.train()
        for epoch in range(EPOCHS):
            loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch")
            for batch in loop:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                loop.set_postfix(loss=loss.item())

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    preds = torch.argmax(outputs.logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / total
            logging.info(f"Validation accuracy: {val_acc:.4f}")
            model.train()

        model.save_pretrained(SAVE_MODEL_PATH)
        tokenizer.save_pretrained(SAVE_MODEL_PATH)
        logging.info(f"Model fine-tuned and saved to {SAVE_MODEL_PATH}")
        print(f"Fine-tuned model saved to {SAVE_MODEL_PATH}")

    test_dataset = TextDataset(X_test, y_test, AutoTokenizer.from_pretrained(SAVE_MODEL_PATH))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(SAVE_MODEL_PATH).to(DEVICE)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Save evaluation metrics and add to global list
    metrics = save_evaluation_report(all_labels, all_preds, RESULTS_PATH, model_name, dataset_name)
    ALL_EVALUATION_METRICS.append({
        'Model': model_name,
        'Dataset': dataset_name,
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1-Score': metrics['F1-Score']
    })
    print_evaluation_report_rich(all_labels, all_preds, dataset_name)


# Common evaluation report functions
def save_evaluation_report(all_labels, all_preds, results_path, model_name, dataset_name):
    metrics = {
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Precision': precision_score(all_labels, all_preds, zero_division=0),
        'Recall': recall_score(all_labels, all_preds, zero_division=0),
        'F1-Score': f1_score(all_labels, all_preds, zero_division=0)
    }
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1-Score']]
    })
    metrics_df.to_csv(results_path / f"evaluation_summary_{model_name}_{dataset_name}.csv", index=False)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
    plt.savefig(results_path / f"confusion_matrix_{model_name}_{dataset_name}.png")
    plt.close()

    # Log evaluation metrics
    logging.info(f"Evaluation metrics for {model_name} on {dataset_name}:")
    for metric, score in metrics.items():
        logging.info(f"  {metric}: {score:.4f}")

    return metrics


def print_evaluation_report_rich(all_labels, all_preds, dataset_name):
    table = Table(title=f"Evaluation Metrics for {dataset_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_row("Accuracy", f"{accuracy_score(all_labels, all_preds):.4f}")
    table.add_row("Precision", f"{precision_score(all_labels, all_preds, zero_division=0):.4f}")
    table.add_row("Recall", f"{recall_score(all_labels, all_preds, zero_division=0):.4f}")
    table.add_row("F1-Score", f"{f1_score(all_labels, all_preds, zero_division=0):.4f}")
    Console().print(table)


# Save final report to Excel
def save_final_report(results_path):
    if not ALL_EVALUATION_METRICS:
        logging.warning("No evaluation metrics to save in final report")
        return
    final_df = pd.DataFrame(ALL_EVALUATION_METRICS)
    final_df.to_excel(results_path / "final_evaluation_report_fine_tune.xlsx", index=False)
    logging.info(f"Final evaluation report saved to {results_path / 'final_evaluation_report.xlsx'}")
    print(f"Final evaluation report saved to {results_path / 'final_evaluation_report.xlsx'}")


# Main execution
if __name__ == "__main__":
    logging.info("Execution started")
    print("Execution started")

    # Log configuration parameters at the start
    log_config_parameters()

    for config in MODEL_DATASET_CONFIGS:
        model_name = config["model_name"]
        dataset_name = config["dataset_name"]
        data_fraction = config["data_fraction"]
        cache_path = DATA_PATH / f"cached_texts_{dataset_name}.pkl"

        logging.info(f"Processing {model_name} on {dataset_name}")
        print(f"Processing {model_name} on {dataset_name}")

        # Load and preprocess dataset
        if dataset_name == "CICIDS2017":
            df = load_cicids2017()
            texts, labels = preprocess_cicids2017(df, data_fraction, cache_path)
        elif dataset_name == "NSL-KDD":
            df = load_nsl_kdd()
            texts, labels = preprocess_nsl_kdd(df, data_fraction, cache_path)
        elif dataset_name == "UNSW":
            df = load_unsw()
            texts, labels = preprocess_unsw(df, data_fraction, cache_path)
        else:
            logging.error(f"Unknown dataset: {dataset_name}")
            print(f"Unknown dataset: {dataset_name}")
            continue

        # Train and evaluate
        train_and_evaluate(texts, labels, model_name, dataset_name)

    # Save final report and combine confusion matrices
    save_final_report(RESULTS_PATH)
    combine_confusion_matrices(RESULTS_PATH)

    logging.info("Execution finished")
    print("Execution finished")
