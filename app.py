import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


@st.cache_resource
def load_trained_model():
    """Load the pre-trained LSTM model from disk."""
    return load_model("next_word_lstm.h5")


@st.cache_resource
def load_tokenizer():
    """Load the fitted tokenizer from disk."""
    with open("tokenizer.pickle", "rb") as handle:
        return pickle.load(handle)


def predict_next_word(model, tokenizer, text, max_sequence_len):
    """Predict the next word given an input text sequence.

    Args:
        model: Trained Keras sequential model.
        tokenizer: Fitted Keras Tokenizer instance.
        text: Input string of seed words.
        max_sequence_len: Maximum sequence length the model expects (including target).

    Returns:
        The predicted next word as a string, or None if prediction fails.
    """
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len - 1, padding="pre"
    )
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


def main():
    st.set_page_config(page_title="Next Word Prediction", layout="centered")
    st.title("Next Word Prediction")
    st.write(
        "Enter a sequence of words below and the LSTM model will predict "
        "the most likely next word."
    )

    model = load_trained_model()
    tokenizer = load_tokenizer()
    max_sequence_len = model.input_shape[1] + 1

    input_text = st.text_input("Enter a sequence of words", "To be or not to")

    if st.button("Predict Next Word"):
        if input_text.strip():
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            if next_word:
                st.success(f"**Predicted next word:** {next_word}")
            else:
                st.warning("Could not predict the next word for the given input.")
        else:
            st.warning("Please enter at least one word.")


if __name__ == "__main__":
    main()
