import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
def time_series_analysis():
    np.random.seed(0)
    date_range = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = pd.Series(50 + np.random.normal(size=100).cumsum(), index=date_range)
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Original Time Series')
    plt.title('Original Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
    model = ARIMA(data, order=(5, 1, 0))
    fit_model = model.fit()
    forecast_steps = 10
    forecast = fit_model.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Original Time Series')
    plt.plot(forecast_index, forecast, label='Forecasted Values', color='red')
    plt.title('ARIMA Model Forecast')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
def sentiment_analysis():
    texts = [
        "I love this product! It's amazing.",
        "This is the worst service I've ever received.",
        "I'm quite satisfied with the quality.",
        "Could be better, but it's okay.",
        "Absolutely fantastic experience!"
    ]
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    for text in texts:
        sentiment = sia.polarity_scores(text)
        print(f"Text: {text}\nSentiment: {sentiment}\n")

def clustering_analysis():
    np.random.seed(0)
    data = np.random.randn(100, 2)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

if __name__ == "__main__":
    print("Running Time Series Analysis...")
    time_series_analysis()

    print("Running Sentiment Analysis...")
    sentiment_analysis()

    print("Running Clustering Analysis...")
    clustering_analysis()
