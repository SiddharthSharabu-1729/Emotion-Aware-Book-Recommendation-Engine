mood_categories = {
    "positive": ["admiration", "approval", "excitement", "joy", "love",
                 "optimism", "gratitude", "pride", "relief", "caring"],
    "negative": ["sadness", "grief", "disappointment", "anger", "disgust",
                 "fear", "remorse", "annoyance", "disapproval", "embarrassment"],
    "cheer_up": ["amusement", "surprise", "optimism", "joy", "relief"],
    "cognitive": ["curiosity", "confusion", "realization", "neutral",
                  "desire", "nervousness"],
    "love": ["love", "desire", "caring"]
}

recommendation_ratios = {
    "positive": {"positive": 0.6, "negative": 0.1, "cheer_up": 0.2, "cognitive": 0.1},
    "negative": {"positive": 0.4, "cheer_up": 0.4, "cognitive": 0.2},
    "love": {"love": 0.6, "positive": 0.2, "cognitive": 0.2},
    "cognitive": {"cognitive": 0.7, "positive": 0.3},
    "neutral": {"cognitive": 0.5, "positive": 0.5}
}

def get_main_category(emotion: str) -> str:
    for cat, emotions in mood_categories.items():
        if emotion in emotions:
            return cat
    return "neutral"
