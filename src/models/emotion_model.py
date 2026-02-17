from transformers import pipeline

_classifier = None

def load_emotion_classifier(device: int = -1):
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            task="text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            device=device
        )
    return _classifier
