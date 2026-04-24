"""
Machine Learning Training Phase for the Handwritten Answer Grading System

This module demonstrates SUPERVISED LEARNING for the ML Hackathon:
1. FEATURE ENGINEERING: Extract multiple features (similarity, length, keywords)
2. SUPERVISED LEARNING: Train a Random Forest model on labeled synthetic data
3. MODEL INFERENCE: The app loads this trained model to predict grades

Why this approach?
- Traditional approach: "If similarity > 0.85 then 10/10" (rule-based)
- ML approach: Learn the marking rubric from examples (supervised learning)
- Better: Model can capture nuanced relationships between features and scores
"""

import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def generate_synthetic_training_data(num_samples: int = 500) -> tuple:
    """Generate synthetic training dataset with features and labels.
    
    Each sample represents a student answer with extracted features:
    - cosine_similarity: Semantic match with reference (0.0-1.0)
    - word_count_ratio: Student answer length vs reference (0.3-2.0)
    - keyword_match_count: Number of key terms matched (0-10)
    
    Labels are based on a marking rubric:
    - similarity > 0.85: Full marks (8-10)
    - similarity 0.70-0.85: Partial marks (6-8)
    - similarity 0.50-0.70: Minimal marks (3-6)
    - similarity < 0.50: Very low marks (0-3)
    """
    X_features = []
    y_scores = []
    
    for _ in range(num_samples):
        # Feature 1: Cosine similarity (0.0 to 1.0)
        similarity = random.uniform(0.0, 1.0)
        
        # Feature 2: Word count ratio (student vs reference)
        word_count_ratio = random.uniform(0.3, 2.0)
        
        # Feature 3: Keyword match count (0-10)
        keyword_matches = random.randint(0, 10)
        
        # Label: Score based on marking rubric (0-10 scale)
        if similarity > 0.85:
            # Full marks: 8-10
            base_score = random.uniform(8, 10)
        elif similarity > 0.70:
            # Partial marks: 6-8
            base_score = random.uniform(6, 8)
        elif similarity > 0.50:
            # Minimal marks: 3-6
            base_score = random.uniform(3, 6)
        else:
            # Very low marks: 0-3
            base_score = random.uniform(0, 3)
        
        # Adjust score based on word count and keyword matches
        # Deduct points if answer is too short or too long
        if word_count_ratio < 0.5 or word_count_ratio > 1.8:
            base_score *= 0.85
        
        # Bonus points for matching keywords
        base_score += (keyword_matches / 10.0) * 1.5
        
        # Clamp to 0-10 range
        final_score = max(0, min(10, base_score))
        
        X_features.append([similarity, word_count_ratio, keyword_matches])
        y_scores.append(final_score)
    
    return np.array(X_features), np.array(y_scores)


def train_grading_model(X_train, y_train) -> RandomForestRegressor:
    """Train a Random Forest Regressor on the synthetic labeled data.
    
    Random Forest is chosen because:
    - Handles non-linear relationships between features and scores
    - Robust to feature scaling
    - Provides feature importance insights
    - Works well with small to medium datasets
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def save_model(model: RandomForestRegressor, model_path: str) -> None:
    """Serialize the trained model to disk."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


def main():
    """Generate data, train model, and save to disk."""
    print("=" * 70)
    print("SUPERVISED LEARNING PHASE: Training Handwritten Answer Grader")
    print("=" * 70)
    
    # Step 1: Generate synthetic training data
    print("\n[1/4] Generating synthetic training dataset...")
    X_features, y_scores = generate_synthetic_training_data(num_samples=500)
    print(f"      Generated {len(X_features)} labeled training samples")
    print(f"      Features per sample: [similarity, word_count_ratio, keyword_matches]")
    print(f"      Score range: {y_scores.min():.2f} - {y_scores.max():.2f} (0-10 scale)")
    
    # Step 2: Split into train/test sets
    print("\n[2/4] Splitting data: 80% train, 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_scores, test_size=0.2, random_state=42
    )
    print(f"      Train set: {len(X_train)} samples")
    print(f"      Test set: {len(X_test)} samples")
    
    # Step 3: Train Random Forest model
    print("\n[3/4] Training Random Forest Regressor...")
    model = train_grading_model(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"      Model Performance:")
    print(f"      - Mean Squared Error: {mse:.4f}")
    print(f"      - R² Score: {r2:.4f}")
    print(f"      - Feature Importance:")
    
    feature_names = ["Cosine Similarity", "Word Count Ratio", "Keyword Matches"]
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f"        * {name}: {importance:.4f}")
    
    # Step 4: Save model
    print("\n[4/4] Saving trained model to disk...")
    model_path = Path(__file__).parent.parent / "models" / "grading_model.pkl"
    save_model(model, str(model_path))
    
    print("\n" + "=" * 70)
    print("✅ Training complete! The app will now use ML inference.")
    print("=" * 70)


if __name__ == "__main__":
    main()
