import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import sklearn


class TFIDFVectorizerScratch:
    """A pure Python TF-IDF vectorizer with manual math documentation."""

    def __init__(self, documents: List[str]):
        self.documents = documents
        self.vocabulary = {}
        self.idf = {}
        self.document_vectors = []
        self._fit()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        return tokens

    @staticmethod
    def _term_frequency(tokens: List[str]) -> dict:
        counts = Counter(tokens)
        total = sum(counts.values())
        return {term: freq / total for term, freq in counts.items()} if total else {}

    def _fit(self) -> None:
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        all_terms = sorted({term for tokens in tokenized_docs for term in tokens})
        self.vocabulary = {term: idx for idx, term in enumerate(all_terms)}

        document_count = len(tokenized_docs)
        for term in self.vocabulary:
            df = sum(1 for tokens in tokenized_docs if term in tokens)
            self.idf[term] = math.log((document_count + 1) / (df + 1)) + 1

        self.document_vectors = [self._vectorize(tokens) for tokens in tokenized_docs]

    def _vectorize(self, tokens: List[str]) -> List[float]:
        tf = self._term_frequency(tokens)
        vector = [0.0] * len(self.vocabulary)
        for term, term_tf in tf.items():
            if term in self.vocabulary:
                vector[self.vocabulary[term]] = term_tf * self.idf[term]
        return vector

    def transform(self, texts: List[str]) -> List[List[float]]:
        return [self._vectorize(self._tokenize(text)) for text in texts]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity from scratch.

    CosineSimilarity(a, b) = (a . b) / (||a|| * ||b||)
    where ||a|| = sqrt(sum(a_i^2))
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def map_similarity_to_score(similarity: float) -> float:
    """Map a similarity score [0.0, 1.0] to a 10-point grading scale."""
    return round(max(0.0, min(1.0, similarity)) * 10, 2)


class HandwritingScorer:
    """ML-powered scorer that uses a trained Random Forest model for grading.
    
    This class demonstrates SUPERVISED LEARNING:
    - Extracts multiple features from student text (similarity, length, keywords)
    - Loads a pre-trained Random Forest model from disk
    - Uses the model to predict the final grade (0-10)
    
    This replaces rule-based scoring (if/else thresholds) with learned patterns.
    """
    
    def __init__(self, reference_texts: List[str]):
        self.reference_texts = reference_texts
        self.vectorizer = TFIDFVectorizerScratch(reference_texts)
        self.reference_vectors = self.vectorizer.document_vectors
        self.ml_model = self._load_ml_model()
    
    def _load_ml_model(self):
        """Load the trained Random Forest model from disk."""
        model_path = Path(__file__).parent.parent / "models" / "grading_model.pkl"
        
        if not model_path.exists():
            print(f"⚠️  Warning: Model file not found at {model_path}")
            print("   Run: python src/train_model.py")
            print("   Falling back to rule-based scoring.")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def extract_features(self, student_text: str, reference_text: str) -> List[float]:
        """Extract ML features from student text.
        
        Features:
        1. Cosine Similarity: Semantic match with reference answer (0.0-1.0)
        2. Word Count Ratio: Student answer length relative to reference
        3. Keyword Match Count: Number of reference keywords found in student text
        """
        # Feature 1: Cosine Similarity
        student_vector = self.vectorizer.transform([student_text])[0]
        ref_vector = self.vectorizer.transform([reference_text])[0]
        similarity = cosine_similarity(student_vector, ref_vector)
        
        # Feature 2: Word Count Ratio
        student_words = len(re.findall(r"[a-zA-Z]+", student_text.lower()))
        ref_words = len(re.findall(r"[a-zA-Z]+", reference_text.lower()))
        word_count_ratio = student_words / ref_words if ref_words > 0 else 0
        
        # Feature 3: Keyword Match Count
        ref_keywords = set(re.findall(r"[a-zA-Z]{4,}", reference_text.lower()))
        student_keywords = set(re.findall(r"[a-zA-Z]{4,}", student_text.lower()))
        keyword_matches = len(ref_keywords & student_keywords)
        
        return [similarity, word_count_ratio, keyword_matches]
    
    def score(self, student_text: str, reference_text: str = None) -> dict:
        """Score a student answer using ML model inference.
        
        If ML model is available: Uses Random Forest prediction
        If ML model unavailable: Falls back to rule-based scoring
        """
        if reference_text is None:
            reference_text = self.reference_texts[0] if self.reference_texts else ""
        
        # Extract features for ML model
        features = self.extract_features(student_text, reference_text)
        
        # Get similarity for matched_reference
        student_vector = self.vectorizer.transform([student_text])[0]
        similarities = [cosine_similarity(student_vector, ref_vec) for ref_vec in self.reference_vectors]
        best_similarity = max(similarities) if similarities else 0.0
        matched_ref = self.reference_texts[similarities.index(best_similarity)] if similarities else ""
        
        # Use ML model to predict score
        if self.ml_model is not None:
            features_array = np.array([features])
            predicted_score = float(self.ml_model.predict(features_array)[0])
            predicted_score = max(0.0, min(10.0, predicted_score))  # Clamp to 0-10
            model_confidence = True
        else:
            # Fallback: Rule-based scoring
            similarity = features[0]
            predicted_score = map_similarity_to_score(similarity)
            model_confidence = False
        
        return {
            "similarity": best_similarity,
            "score": round(predicted_score, 2),
            "matched_reference": matched_ref,
            "ml_features": {
                "cosine_similarity": round(features[0], 4),
                "word_count_ratio": round(features[1], 2),
                "keyword_matches": int(features[2]),
            },
            "model_confidence": model_confidence,
        }


DEFAULT_REFERENCES = [
    "The mitochondria is the powerhouse of the cell.",
    "Mitochondria provide energy for the cell.",
    "The cell's powerhouse is the mitochondria.",
    "Mitochondria generate ATP for cellular energy.",
    "The mitochondrion is the energy factory of the cell.",
    "A cell uses mitochondria to convert nutrients into energy.",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score a student answer against reference answers.")
    parser.add_argument("student_text", help="Student OCR text")
    args = parser.parse_args()

    scorer = HandwritingScorer(DEFAULT_REFERENCES)
    result = scorer.score(args.student_text)
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Score (0-10): {result['score']}")
    print(f"Model Confidence: {result['model_confidence']}")
    print(f"ML Features: {result['ml_features']}")
    print(f"Matched Reference: {result['matched_reference']}")
