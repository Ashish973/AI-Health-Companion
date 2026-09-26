import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download lexicon on initial load
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

EMOTION_KEYWORDS = {
    "Anxious/Stressed": ["stress", "stressed", "exam", "deadlines", "panic", "overwhelmed", "nervous", "anxiety", "pressure"],
    "Sadness/Depressed": ["sad", "lonely", "hopeless", "crying", "miserable", "empty", "grief", "heartbroken"],
    "Anger/Frustrated": ["angry", "furious", "annoyed", "irritated", "mad", "unfair", "hate"],
    "Joy/Grateful": ["happy", "relieved", "excited", "grateful", "peaceful", "proud", "content", "calm"]
}

def analyze_emotion(text: str) -> dict:
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    lowered = text.lower()
    
    detected = None
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            detected = emotion
            break
            
    if not detected:
        if compound >= 0.35:
            detected = "Joy/Grateful"
        elif compound <= -0.35:
            detected = "Sadness/Depressed"
        elif compound < 0.05 and scores['neg'] > 0.2:
            detected = "Anxious/Stressed"
        else:
            detected = "Neutral/Reflective"

    # Normalize stress proxy (1 to 10 scale)
    stress_proxy = int(min(max((1.0 - compound) * 4.5 + 1.0, 1), 10))

    return {
        "primary_emotion": detected,
        "sentiment_score": compound,
        "stress_level": stress_proxy,
        "raw_scores": scores
    }