import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from src.youtube_service import YouTubeService

class HateCommentClassifier:
    def __init__(self, model_path="models/hate-detection-balanced"):
        print(f"Loading model from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Mapping based on training (0: Hate Speech, 1: Offensive Language, 2: Neither)
        # Modified to match requested labels
        self.id2label = {0: "Abusive", 1: "Not Abusive", 2: "Neither"}
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
