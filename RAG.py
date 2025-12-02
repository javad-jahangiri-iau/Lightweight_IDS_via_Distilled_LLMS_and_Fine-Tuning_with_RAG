import os
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from rich.table import Table
from rich.console import Console
import time
from torch.utils.data import DataLoader, TensorDataset

# Configuration
BATCH_SIZE = 8
MAX_LEN = 128
LEARNING_RATE = 5e-5
DATA_PATH = Path("data")
BASE_RESULTS_PATH = Path("results")
LOG_PATH = Path("logs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH_SENTENCE_TRANSFORMER = "models/all-MiniLM-L6-v2"  # Path to offline SentenceTransformer model

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
RAG_RESULTS_PATH = BASE_RESULTS_PATH / EXECUTION_TIME / "rag"
RAG_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
LOG_FILE = RAG_RESULTS_PATH / "rag_run.log"

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"Starting RAG integration. Using device: {DEVICE}")

# Check GPU availability
logging.info(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"CUDA Device Count: {torch.cuda.device_count()}")
    logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# Load SentenceTransformer model once
ENCODER = SentenceTransformer(MODEL_PATH_SENTENCE_TRANSFORMER, device=DEVICE)


# Function to log configuration parameters
def log_config_parameters():
    logging.info("RAG Configuration Parameters:")
    logging.info(f"  BATCH_SIZE: {BATCH_SIZE}")
    logging.info(f"  MAX_LEN: {MAX_LEN}")
    logging.info(f"  LEARNING_RATE: {LEARNING_RATE}")
    logging.info(f"  DATA_PATH: {DATA_PATH}")
    logging.info(f"  RAG_RESULTS_PATH: {RAG_RESULTS_PATH}")
    logging.info(f"  LOG_FILE: {LOG_FILE}")
    logging.info(f"  MODEL_PATH_SENTENCE_TRANSFORMER: {MODEL_PATH_SENTENCE_TRANSFORMER}")
    logging.info(f"  MODEL_DATASET_CONFIGS: {MODEL_DATASET_CONFIGS}")


# Create vector index for knowledge base
def create_vector_index(kb_dir, dataset_name, output_index_path):
    logging.info(f"Creating vector index for {dataset_name}")
    kb_files = [
        os.path.join(kb_dir, f"{dataset_name.lower()}.txt"),
        os.path.join(kb_dir, "mitre.txt"),
        os.path.join(kb_dir, "cve.txt")
    ]
    documents = []
    for kb_file in kb_files:
        if os.path.exists(kb_file):
            with open(kb_file, 'r', encoding='utf-8') as f:
                documents.extend([line.strip() for line in f if line.strip()])
        else:
            logging.warning(f"Knowledge base file {kb_file} not found")

    if not documents:
        logging.error(f"No documents found for {dataset_name}")
        return False

    logging.info(f"Encoding {len(documents)} documents for {dataset_name}")
    doc_embeddings = ENCODER.encode(documents, batch_size=32, show_progress_bar=False)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings.astype(np.float32))

    faiss.write_index(index, str(output_index_path))
    with open(os.path.join(kb_dir, f"{dataset_name.lower()}_documents.txt"), 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc + '\n')

    logging.info(f"Vector index and documents saved for {dataset_name} at {output_index_path}")
    return True


# Retrieve relevant documents
def retrieve_relevant_docs(input_text, kb_dir, dataset_name, top_k=3):
    logging.info(f"Retrieving relevant documents for input: {input_text[:50]}...")
    start_time = time.time()
    index_path = os.path.join(kb_dir, f"{dataset_name.lower()}_faiss_index.bin")
    if not os.path.exists(index_path):
        logging.error(f"FAISS index not found at {index_path}")
        return [], []

    index = faiss.read_index(index_path)
    with open(os.path.join(kb_dir, f"{dataset_name.lower()}_documents.txt"), 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]

    input_embedding = ENCODER.encode([input_text], show_progress_bar=False)[0]
    distances, indices = index.search(np.array([input_embedding]).astype(np.float32), top_k)
    relevant_docs = [documents[idx] for idx in indices[0]]
    scores = distances[0]

    logging.info(
        f"Retrieved {len(relevant_docs)} documents with scores: {scores} in {time.time() - start_time:.2f} seconds")
    return relevant_docs, scores


# Combine input with retrieved documents
def combine_input_with_docs(input_text, relevant_docs):
    context = " ".join(relevant_docs)
    combined_input = f"Input: {input_text} Context: {context}"
    return combined_input


# Predict with RAG for a batch
def predict_with_rag_batch(texts, relevant_docs_list, scores_list, model_path, dataset_name, weight_model=0.7,
                           weight_context=0.3):
    logging.info(f"Predicting with RAG for model at {model_path} for batch of {len(texts)} samples")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()

    predictions = []
    probabilities = []
    for text, relevant_docs, scores in zip(texts, relevant_docs_list, scores_list):
        combined_input = combine_input_with_docs(text, relevant_docs)
        inputs = tokenizer(combined_input, return_tensors='pt', truncation=True, padding=True, max_length=MAX_LEN)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        context_score = np.mean([1 - d for d in scores]) if scores.size > 0 else 0.0

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            model_pred = probs[0, 1].item()  # Probability of Attack class
            final_prob = weight_model * model_pred + weight_context * context_score
            prediction = 1 if final_prob > 0.5 else 0

        predictions.append(prediction)
        probabilities.append([1 - final_prob, final_prob])

    logging.info(f"Batch prediction completed in {time.time() - start_time:.2f} seconds")
    return predictions, probabilities


# Save metrics to Excel
def save_metrics_to_excel(all_metrics, results_path):
    logging.info("Saving all metrics to Excel")
    metrics_df = pd.DataFrame(all_metrics,
                              columns=['Model', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR'])
    excel_file = results_path / "rag_evaluation_summary.xlsx"
    metrics_df.to_excel(excel_file, index=False, engine='openpyxl')
    logging.info(f"Metrics saved to {excel_file}")


# Create combined confusion matrices image
def create_combined_confusion_matrices(all_confusion_matrices, results_path):
    logging.info("Creating combined confusion matrices image")
    n_configs = len(all_confusion_matrices)
    n_cols = 3
    n_rows = (n_configs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten() if n_configs > 1 else [axes]

    for idx, (model_name, dataset_name, cm) in enumerate(all_confusion_matrices):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'], ax=axes[idx])
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        axes[idx].set_title(f'{model_name} on {dataset_name}')

    # Hide unused subplots
    for idx in range(len(all_confusion_matrices), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    combined_cm_file = results_path / "combined_confusion_matrices_rag.png"
    plt.savefig(combined_cm_file, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"Combined confusion matrices saved to {combined_cm_file}")


# Evaluate with RAG
def evaluate_with_rag(test_texts, test_labels, model_path, kb_dir, dataset_name, results_path):
    logging.info(f"Evaluating model at {model_path} with RAG on {dataset_name}")
    dataset = TensorDataset(torch.tensor(range(len(test_texts))), torch.tensor(test_labels))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    predictions = []
    probabilities = []
    for batch_idx, (indices, labels) in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", unit="batch")):
        batch_texts = [test_texts[i] for i in indices]
        relevant_docs_list = []
        scores_list = []
        for text in batch_texts:
            relevant_docs, scores = retrieve_relevant_docs(text, kb_dir, dataset_name)
            relevant_docs_list.append(relevant_docs)
            scores_list.append(scores)

        batch_preds, batch_probs = predict_with_rag_batch(batch_texts, relevant_docs_list, scores_list, model_path,
                                                          dataset_name)
        predictions.extend(batch_preds)
        probabilities.extend(batch_probs)

    metrics = {
        'Accuracy': accuracy_score(test_labels, predictions),
        'Precision': precision_score(test_labels, predictions, zero_division=0),
        'Recall': recall_score(test_labels, predictions, zero_division=0),
        'F1-Score': f1_score(test_labels, predictions, zero_division=0)
    }
    cm = confusion_matrix(test_labels, predictions)
    fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) > 0 else 0
    metrics['FPR'] = fpr

    # Save metrics and confusion matrix
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR'],
        'Score': [metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1-Score'], metrics['FPR']]
    })
    metrics_file = results_path / f"rag_evaluation_summary_{model_path.name}_{dataset_name}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    logging.info(f"Metrics saved to {metrics_file}")

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'RAG Confusion Matrix - {model_path.name} on {dataset_name}')
    cm_file = results_path / f"rag_confusion_matrix_{model_path.name}_{dataset_name}.png"
    plt.savefig(cm_file)
    plt.close()
    logging.info(f"Confusion matrix saved to {cm_file}")

    # Print metrics to console
    table = Table(title=f"RAG Evaluation Metrics for {dataset_name} - {model_path.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta")
    for metric, score in metrics.items():
        table.add_row(metric, f"{score:.4f}")
    Console().print(table)

    return metrics, cm


# Load dataset (reusing preprocessing from Fine_tune.py)
def load_and_preprocess_dataset(dataset_name, data_fraction, cache_path):
    logging.info(f"Loading and preprocessing {dataset_name}")
    if dataset_name == "CICIDS2017":
        from Fine_tune import load_cicids2017, preprocess_cicids2017
        df = load_cicids2017()
        texts, labels = preprocess_cicids2017(df, data_fraction, cache_path)
    elif dataset_name == "NSL-KDD":
        from Fine_tune import load_nsl_kdd, preprocess_nsl_kdd
        df = load_nsl_kdd()
        texts, labels = preprocess_nsl_kdd(df, data_fraction, cache_path)
    elif dataset_name == "UNSW":
        from Fine_tune import load_unsw, preprocess_unsw
        df = load_unsw()
        texts, labels = preprocess_unsw(df, data_fraction, cache_path)
    else:
        logging.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return texts, labels


# Main execution
if __name__ == "__main__":
    logging.info("Starting RAG integration execution")
    print("Starting RAG integration execution")
    log_config_parameters()

    all_metrics = []
    all_confusion_matrices = []
    for config in MODEL_DATASET_CONFIGS:
        model_name = config["model_name"]
        dataset_name = config["dataset_name"]
        data_fraction = config["data_fraction"]
        cache_path = DATA_PATH / f"cached_texts_{dataset_name}.pkl"
        model_path = Path(f"models/finetuned_model/{model_name}_{dataset_name}")
        kb_dir = Path(f"data/knowledge_base/{dataset_name.lower()}")

        logging.info(f"Processing {model_name} on {dataset_name}")
        print(f"Processing {model_name} on {dataset_name}")

        # Create FAISS index if not exists
        index_path = kb_dir / f"{dataset_name.lower()}_faiss_index.bin"
        if not index_path.exists():
            success = create_vector_index(kb_dir, dataset_name, index_path)
            if not success:
                logging.error(f"Skipping {model_name} on {dataset_name} due to missing knowledge base")
                continue

        # Load and preprocess dataset
        texts, labels = load_and_preprocess_dataset(dataset_name, data_fraction, cache_path)
        _, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

        # Evaluate with RAG
        metrics, cm = evaluate_with_rag(X_test, y_test, model_path, kb_dir, dataset_name, RAG_RESULTS_PATH)

        # Collect metrics for Excel
        all_metrics.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1-Score': metrics['F1-Score'],
            'FPR': metrics['FPR']
        })

        # Collect confusion matrix
        all_confusion_matrices.append((model_name, dataset_name, cm))

    # Save all metrics to Excel
    save_metrics_to_excel(all_metrics, RAG_RESULTS_PATH)

    # Create and save combined confusion matrices
    create_combined_confusion_matrices(all_confusion_matrices, RAG_RESULTS_PATH)

    logging.info("RAG integration execution finished")
    print("RAG integration execution finished")
