import os
try:
    import hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from src.youtube_service import YouTubeService

class HateCommentClassifier:
    def __init__(self, model_name="unitary/multilingual-toxic-xlm-roberta"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Tier 0: Check if local fine-tuned model exists (Best & Fastest)
        local_model = "models/hate-detection-balanced"
        if os.path.exists(local_model):
            try:
                print(f"Loading local fine-tuned model from {local_model}...")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model)
                self.model = AutoModelForSequenceClassification.from_pretrained(local_model)
                
                if self.device.type == "cuda":
                    print("Optimizing local model for GPU (FP16)...")
                    self.model = self.model.half()
                    
                self.id2label = {0: "Abusive", 1: "Not Abusive", 2: "Neither"}
                self.fallback_mode = False
                print("Local fine-tuned model loaded successfully.")
                return
            except Exception as e:
                print(f"Warning: Failed to load local model from {local_model}: {e}")
                print("Falling back to Hugging Face Hub models...")
        
        # Tier 1: Multilingual Toxic XLM-RoBERTa (Primary - 1.11 GB)
        # Attempt to load from local cache first to avoid download hangs
        try:
            print(f"Checking if primary model {model_name} is already cached...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
            
            if self.device.type == "cuda":
                print("Optimizing model for GPU (FP16)...")
                self.model = self.model.half()
                
            self.labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
            self.fallback_mode = True 
            print(f"Loaded primary model from local cache successfully.")
        except Exception as e_cache:
            # If not in cache, try downloading the primary model (with fast parallel download if hf-transfer is installed)
            try:
                print(f"Downloading primary model {model_name} (1.11 GB)...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                if self.device.type == "cuda":
                    print("Optimizing model for GPU (FP16)...")
                    self.model = self.model.half()
                    
                self.labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
                self.fallback_mode = True 
                print(f"Primary model downloaded and loaded successfully.")
            except Exception as e1:
                print(f"Warning: Failed to download primary model: {e1}")
                self._load_fallback_bert()


    def _load_fallback_bert(self):
        # Tier 2: Lightweight Toxic BERT (Fallback - 268 MB)
        fallback_model = "unitary/toxic-bert"
        try:
            print(f"Loading Lightweight Fallback Model ({fallback_model})...")
            try:
                # Try cache first
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(fallback_model, local_files_only=True)
                print("Loaded fallback model from local cache.")
            except Exception:
                # Download if not cached
                print(f"Downloading lightweight fallback model ({fallback_model} - 268 MB)...")
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self.model = AutoModelForSequenceClassification.from_pretrained(fallback_model)
                print("Fallback model downloaded successfully.")
                
            if self.device.type == "cuda":
                print("Optimizing fallback model for GPU (FP16)...")
                self.model = self.model.half()
                
            self.labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
            self.fallback_mode = True
            print(f"Lightweight fallback model loaded successfully.")
        except Exception as e2:
            print(f"Warning: Failed to load fallback model {fallback_model}: {e2}")
            self._load_local_model(e2)

    def _load_local_model(self, last_error):
        # Tier 3: Local fine-tuned model
        local_model = "models/hate-detection-balanced"
        try:
            print(f"Loading Local Model ({local_model})...")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(local_model)
            self.id2label = {0: "Abusive", 1: "Not Abusive", 2: "Neither"}
            self.fallback_mode = False
            print(f"Local model loaded successfully.")
        except Exception as e3:
            print("Critical: All models failed to load!")
            print(f"Last Error: {last_error}")
            print(f"Local Model Error: {e3}")
            raise e3

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits
            
        if self.fallback_mode:
            # Multilingual model uses sigmoid for multi-label classification
            probs = torch.sigmoid(scores).cpu().numpy()[0]
            
            # Identify the specific toxicity types
            tox_types = {self.labels[i]: float(probs[i]) for i in range(len(probs))}
            
            # Higher sensitivity: If ANY toxicity label is > 0.4, classify as Abusive
            # We use 0.4 instead of 0.5 to catch more subtle or borderline cases in speech
            max_tox_prob = float(np.max(probs))
            is_toxic = max_tox_prob > 0.4
            
            label = "Abusive" if is_toxic else "Not Abusive"
            confidence = max_tox_prob if is_toxic else (1.0 - max_tox_prob)
            
            return {
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    "Abusive": max_tox_prob,
                    "Not Abusive": 1.0 - max_tox_prob,
                    "Neither": 0.0,
                    "details": tox_types
                }
            }
        else:
            probs = torch.nn.functional.softmax(scores, dim=1).cpu().numpy()[0]
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
                
            if self.fallback_mode:
                # Sigmoid for multi-label multilingual model
                probs = torch.sigmoid(scores).cpu().numpy()
                for j in range(len(batch_texts)):
                    max_tox_prob = float(np.max(probs[j]))
                    is_toxic = max_tox_prob > 0.4
                    label = "Abusive" if is_toxic else "Not Abusive"
                    confidence = max_tox_prob if is_toxic else (1.0 - max_tox_prob)
                    results.append({
                        "text": batch_texts[j], 
                        "label": label,
                        "confidence": confidence,
                        "probabilities": {
                            "Abusive": max_tox_prob,
                            "Not Abusive": 1.0 - max_tox_prob,
                            "Neither": 0.0
                        }
                    })
            else:
                probs = torch.nn.functional.softmax(scores, dim=1).cpu().numpy()
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

    def classify_transcription(self, transcription):
        """
        Classify a transcription and return abusive segments with timestamps.
        
        Args:
            transcription (list): List of transcription segments with 'text', 'start_time', and 'end_time'.

        Returns:
            list: Abusive segments with timestamps.
        """
        abusive_segments = []
        for segment in transcription:
            text = segment['text']
            start_time = segment['start_time']
            end_time = segment['end_time']

            classification = self.predict(text)
            if classification['label'] == 'Abusive':
                abusive_segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                })
        return abusive_segments

if __name__ == "__main__":
    # Initialize services
    yt_service = YouTubeService()
    classifier = HateCommentClassifier()

    # Example YouTube video URL
    video_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"

    # Extract and classify video content
    print("Extracting video content...")
    transcript = yt_service.extract_video_content(video_url)
    print("Transcript:", transcript)

    print("Classifying content...")
    result = classifier.predict(transcript)
    print("Classification Result:", result)
