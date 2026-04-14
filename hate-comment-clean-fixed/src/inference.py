import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class HateCommentClassifier:
    def __init__(self, model_path="models/hate-detection-balanced"):
        print(f"Loading model from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Mapping based on training (0: Hate Speech, 1: Offensive Language, 2: Neither)
        # Note: Verify specific mapping from your dataset/training script
        self.id2label = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
        print("Model loaded successfully.")

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits
            probs = torch.nn.functional.softmax(scores, dim=1)
            
        probs = probs.cpu().numpy()[0]
        pred_id = np.argmax(probs)
        label = self.id2label[pred_id]
        confidence = float(probs[pred_id])
        
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                self.id2label[i]: float(probs[i]) for i in range(len(probs))
            }
        }

    def predict_batch(self, texts, batch_size=16):
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits
                probs = torch.nn.functional.softmax(scores, dim=1)
            
            probs = probs.cpu().numpy()
            pred_ids = np.argmax(probs, axis=1)
            
            for j, pred_id in enumerate(pred_ids):
                label = self.id2label[pred_id]
                confidence = float(probs[j][pred_id])
                results.append({
                    "text": batch_texts[j], 
                    "label": label,
                    "confidence": confidence,
                    "probabilities": {
                        self.id2label[k]: float(probs[j][k]) for k in range(len(probs[j]))
                    }
                })
        return results

if __name__ == "__main__":
    # Test
    classifier = HateCommentClassifier()
    test_texts = [
        "I hate you and your people.",
        "That was a stupid move.",
        "Have a nice day!"
    ]
    for t in test_texts:
        print(f"Input: {t}")
        print(f"Output: {classifier.predict(t)}\n")
