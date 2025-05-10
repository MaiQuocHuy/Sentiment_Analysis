import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, DistilBertModel
import os

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .title {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .positive {
        font-size: 1.8rem;
        color: #4CAF50;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
        text-align: center;
        margin: 1rem 0;
    }
    .negative {
        font-size: 1.8rem;
        color: #F44336;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFEBEE;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Define the model architecture
class DistilBertForSentimentAnalysis(nn.Module):
    def __init__(self, freeze_bert=False):
        super(DistilBertForSentimentAnalysis, self).__init__()

        # Load pre-trained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Freeze BERT if needed
        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False

        # Classification layer
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 2)  # Binary classification (positive/negative)

    def forward(self, input_ids, attention_mask):
        # Pass input through DistilBERT
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # Get the hidden state of the [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and classification layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_dir='./model_save/'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_dir)

        # Initialize model
        model = DistilBertForSentimentAnalysis()

        # Load model state dict
        model_path = os.path.join(model_dir, 'bert_sentiment_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None, None

# Predict sentiment
def predict_sentiment(text, model, tokenizer, device, max_length=128):
    # Tokenize input
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_attention_mask=True
    )

    # Move to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_label = torch.max(probs, dim=1)

    # Return result
    return {
        "text": text,
        "predicted_label": predicted_label.item(),
        "confidence": confidence.item(),
        "positive_prob": probs[0][1].item(),
        "negative_prob": probs[0][0].item()
    }

# Streamlit app
def main():
    # Display header
    st.markdown("<h1 class='title'>Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Enter text to analyze its sentiment</p>", unsafe_allow_html=True)

    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer()

    if model is None or tokenizer is None or device is None:
        st.error("Failed to load model or tokenizer. Please check the error message above.")
        return

    # Text input
    user_input = st.text_area("Enter text:", height=150)

    # Predict button
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    # Get prediction
                    result = predict_sentiment(user_input, model, tokenizer, device)

                    # Display result
                    if result['predicted_label'] == 1:
                        st.markdown(f"<div class='positive'>Positive 😊</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='negative'>Negative 😞</div>", unsafe_allow_html=True)

                    # Display confidence
                    st.markdown(f"<div class='confidence'>Confidence: {result['confidence']*100:.2f}%</div>", unsafe_allow_html=True)

                    # Display probabilities
                    st.write("Probability Distribution:")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Positive:")
                        st.progress(result['positive_prob'])
                        st.write(f"{result['positive_prob']*100:.2f}%")

                    with col2:
                        st.write("Negative:")
                        st.progress(result['negative_prob'])
                        st.write(f"{result['negative_prob']*100:.2f}%")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please enter some text to analyze.")

    # Add information about the model
    with st.expander("About the Model"):
        st.write("""
        This application uses a fine-tuned DistilBERT model for sentiment analysis.

        The model was trained on IMDB movie reviews and can classify text as either positive or negative.

        - **Positive**: The text expresses a favorable or positive sentiment
        - **Negative**: The text expresses an unfavorable or negative sentiment

        The confidence score indicates how certain the model is about its prediction.
        """)

if __name__ == "__main__":
    main()