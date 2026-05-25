import torch
import torch.nn as nn
import numpy as np
import mlflow
import sys
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer

mlflow.is_tracking_uri_set = lambda: True

class MovieDataset(Dataset):
    """
    Custom PyTorch Dataset for handling movie content and multi-label genres.
    """
    def __init__(self, texts, labels, tokenizer, max_len=256): 
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

def compute_metrics(eval_pred):
    """
    Calculate F1 micro, precision, and recall for multi-label classification.
    """
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    predictions = np.where(probs > 0.5, 1, 0)
    
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    precision = precision_score(labels, predictions, average='micro', zero_division=0)
    recall = recall_score(labels, predictions, average='micro', zero_division=0)
    
    return {
        'f1_micro': f1_micro,
        'precision': precision,
        'recall': recall
    }

# ---------------------------------------------------------------------
# CUSTOM TRAINER: Apply higher penalization to minority genres
# ---------------------------------------------------------------------
class WeightedLossTrainer(Trainer):
    def __init__(self, pos_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Transfer the weights tensor to the same device as the model (e.g., GPU)
        weights = self.pos_weights.to(logits.device)
        
        # Execute BCEWithLogitsLoss utilizing the calculated positive weights
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=weights)
        loss = loss_fct(logits, labels.float())
        
        return (loss, outputs) if return_outputs else loss

def train_and_evaluate(X_train, X_test, y_train, y_test, num_labels):
    """
    Configure MLflow, enforce CUDA availability, and execute the model training process.
    """
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA is not available. Training requires a GPU.")
        sys.exit(1)

    print("CUDA is ready. Setting up Model & Trainer...")
    device = torch.device('cuda')

    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    train_dataset = MovieDataset(X_train, y_train, tokenizer)
    test_dataset = MovieDataset(X_test, y_test, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        problem_type="multi_label_classification"
    )
    model.to(device)

    # ---------------------------------------------------------------------
    # CALCULATE CLASS WEIGHTS BASED ON TRAINING DATA DISTRIBUTION
    # ---------------------------------------------------------------------
    print("Calculating class weights to address minority genres...")
    num_positives = np.sum(y_train, axis=0)
    num_negatives = len(y_train) - num_positives
    pos_weights = num_negatives / (num_positives + 1e-5)
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32)

    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        num_train_epochs=3, 
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        logging_dir='./logs',
        logging_steps=50,
        report_to="mlflow" 
    )

    # Inject the custom WeightedLossTrainer
    trainer = WeightedLossTrainer(
        pos_weights=pos_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    mlflow.set_experiment("DistilBERT_Movie_Recommendation_System")
    with mlflow.start_run():
        print("Initiating training process with weighted loss for minority classes...")
        trainer.train()

        print("Evaluating the best model...")
        eval_results = trainer.evaluate()
        print(eval_results)
        
        mlflow.log_metrics({
            "eval_f1_micro": eval_results.get("eval_f1_micro", 0),
            "eval_precision": eval_results.get("eval_precision", 0),
            "eval_recall": eval_results.get("eval_recall", 0)
        })

        # Save to a distinct directory to prevent overwriting previous versions
        save_path = '../models/best_model_v2'
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Optimal model v2 and tokenizer successfully saved to: {save_path}")