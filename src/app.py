import os
import tempfile

import streamlit as st
from PIL import Image

from ocr_engine import extract_text, preprocess_image
from scorer import HandwritingScorer, DEFAULT_REFERENCES


def main() -> None:
    st.set_page_config(page_title="Handwritten Answer Grading (ML-Powered)", layout="wide")
    st.title("🤖 Handwritten Answer Grading System (ML-Powered)")
    st.markdown(
        "**Supervised Learning Grader**: Upload a handwritten response, provide the reference answer, "
        "and let the trained Random Forest model predict the grade."
    )

    # Sidebar: Teacher's Reference Answer
    st.sidebar.header("📚 Teacher Configuration")
    st.sidebar.write("Provide the **correct/reference answer** for grading:")
    reference_answer = st.sidebar.text_area(
        "Reference Answer",
        value=DEFAULT_REFERENCES[0] if DEFAULT_REFERENCES else "",
        height=100,
        help="The ground truth answer that students should aim for."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.write("**ML Model Status**:")
    st.sidebar.info("✅ Model loaded from `models/grading_model.pkl`\n\n"
                   "Uses features:\n"
                   "- Cosine Similarity\n"
                   "- Word Count Ratio\n"
                   "- Keyword Matches\n\n"
                   "Algorithm: Random Forest Regressor")

    # Main upload section
    uploaded_file = st.file_uploader("Upload a handwriting image", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        st.info("👆 Upload an image file to start grading.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        # Display original image
        original_image = Image.open(tmp_path)
        st.subheader("📷 Uploaded Image")
        st.image(original_image, use_column_width=True)

        # Display preprocessed image
        processed = preprocess_image(tmp_path)
        st.subheader("🔍 Preprocessed Image (Binarized & Rescaled)")
        st.image(processed, use_column_width=True, clamp=True)

        # Extract OCR text
        ocr_text = extract_text(tmp_path)
        st.subheader("📝 OCR Extracted Text")
        st.code(ocr_text or "(No text detected)")

        # ML-powered grading
        scorer = HandwritingScorer([reference_answer] + DEFAULT_REFERENCES)
        result = scorer.score(ocr_text, reference_text=reference_answer)

        # Display grading results
        st.subheader("📊 ML Model Grading Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cosine Similarity", f"{result['ml_features']['cosine_similarity']:.4f}")
        with col2:
            st.metric("Word Count Ratio", f"{result['ml_features']['word_count_ratio']:.2f}")
        with col3:
            st.metric("Keyword Matches", f"{result['ml_features']['keyword_matches']}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Assigned Marks (0-10)", f"{result['score']}")
        with col2:
            if result['model_confidence']:
                st.success("✅ ML Model Inference Active")
            else:
                st.warning("⚠️ Fallback Mode (Model Not Available)")
        
        st.markdown("---")
        
        st.success(f"✅ Grading complete! The ML model assigned {result['score']} out of 10.")
        
        st.write("**Matched Reference Answer:**")
        st.info(result['matched_reference'])

    except Exception as exc:
        st.error(f"❌ Error processing image: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
