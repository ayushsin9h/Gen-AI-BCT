from wordcloud import WordCloud  # Fixed: Changed capital 'F' to lowercase 'from'
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import google.generativeai as genai
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Note: Removed unused imports (numpy, confusion_matrix) for cleaner code.

# Download required NLTK data for sentiment scoring
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Configure the LLM API 
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")) 

def predict_sentiment(text):
    """
    Analyzes raw text and predicts the sentiment polarity.
    """
    scores = sia.polarity_scores(str(text))
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def generate_insights(reviews_list, product_name):
    """
    Passes grouped reviews to the Generative AI model for business insights.
    """
    if not reviews_list:
        return "No critical reviews found."

    reviews_text = "\n- ".join(reviews_list)
    prompt = f"""
    Analyze the following negative feedback for '{product_name}':
    {reviews_text}
    
    Provide a 3-bullet-point summary of the core issues and suggest 1 actionable technical fix.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Prediction Error: {e}"
    
if __name__ == "__main__":

    print("Loading dataset...")
    df = pd.read_csv('product_reviews.csv')

    print("Running sentiment predictions on dataset...")
    df['Predicted_Sentiment'] = df['Review_Text'].apply(predict_sentiment)

    print("\n--- Prediction Results ---")
    print(df[['Review_ID', 'Predicted_Sentiment', 'Review_Text']])

    # -------------------------------
    # GRAPHICAL ANALYSIS
    # -------------------------------

    sns.set_theme(style="whitegrid") # Updated to set_theme (set is deprecated in newer Seaborn versions)

    # 1️⃣ Sentiment Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='Predicted_Sentiment')
    plt.title("Overall Sentiment Distribution")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,6))
    sentiment_counts = df['Predicted_Sentiment'].value_counts()
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
    plt.title("Sentiment Percentage Distribution")
    plt.show()

    # Fixed: Indented the remaining block so it executes inside the __main__ execution block
    negative_text = " ".join(df[df['Predicted_Sentiment'] == 'Negative']['Review_Text'])

    if negative_text.strip() != "":
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
        
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("WordCloud - Negative Reviews")
        plt.show()

    # 2️⃣ Product-wise Sentiment
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='Product_Name', hue='Predicted_Sentiment')
    plt.xticks(rotation=45)
    plt.title("Sentiment by Product")
    plt.tight_layout()
    plt.show()

    # 3️⃣ Rating Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='Rating')
    plt.title("Rating Distribution")
    plt.tight_layout()
    plt.show()

    # 4️⃣ Negative Reviews per Product
    negative_counts = df[df['Predicted_Sentiment'] == 'Negative']['Product_Name'].value_counts()

    plt.figure(figsize=(6,4))
    negative_counts.plot(kind='bar', color='red')
    plt.title("Negative Reviews per Product")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # AI INSIGHT GENERATION
    # -------------------------------

    target_product = "Noise-Canceling Headphones"

    negative_reviews = df[
        (df['Product_Name'] == target_product) &
        (df['Predicted_Sentiment'] == 'Negative')
    ]['Review_Text'].tolist()

    print(f"\n--- Generating AI Business Insights for {target_product} ---")
    insights = generate_insights(negative_reviews, target_product)
    print(insights)