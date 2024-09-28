from datetime import datetime
from alpaca_trade_api import REST, Stream
from timedelta import Timedelta
from finbert_utils import estimate_sentiment
import asyncio

API_KEY = 'PK2EI3D956LK9CSHIODN'
API_SECRET = 'o4wHHZqWEjwWfRjMl0adCcI28h6B3eKa6qnhyAK7'
BASE_URL = "https://paper-api.alpaca.markets"
STREAM_URL = "wss://stream.data.alpaca.markets/v1beta2/news"

class NewsSentimentAnalyzer:
    def __init__(self, symbol: str = "AAPL"):
        self.symbol = symbol
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        self.stream = Stream(API_KEY, API_SECRET, base_url=STREAM_URL)

    def get_dates(self):
        today = datetime.now().date()
        return today, today

    def get_sentiment(self, headline):
        probability, sentiment = estimate_sentiment([headline])
        return probability, sentiment

    def fetch_and_print_historical_news(self):
        today, _ = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=today, end=today)
        for ev in news:
            headline = ev.headline
            date = ev.created_at
            print(f"Company: {self.symbol}, Date: {date}, News: {headline}")
            probability, sentiment = self.get_sentiment(headline)
            suggestion = self.get_suggestion(sentiment, probability)
            print(f"Sentiment: {sentiment} (Probability: {probability}), Suggestion: {suggestion}\n")

    def get_suggestion(self, sentiment, probability):
        if sentiment == "positive" and probability > 0.999:
            return "buy"
        elif sentiment == "negative" and probability > 0.999:
            return "sell"
        else:
            return "hold"

    async def on_news(self, news):
        headline = news.headline
        date = news.created_at
        print(f"Company: {self.symbol}, Date: {date}, Real-time News: {headline}")
        probability, sentiment = self.get_sentiment(headline)
        suggestion = self.get_suggestion(sentiment, probability)
        print(f"Real-time Sentiment: {sentiment} (Probability: {probability}), Suggestion: {suggestion}\n")

    def start_real_time_stream(self):
        self.stream.subscribe_news(self.on_news, self.symbol)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.stream._run_forever())

# Example usage
analyzer = NewsSentimentAnalyzer(symbol="AAPL")
analyzer.fetch_and_print_historical_news()
analyzer.start_real_time_stream()
