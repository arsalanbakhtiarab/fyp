from datetime import datetime
from alpaca_trade_api import REST, Stream
from timedelta import Timedelta
from finbert_utils import estimate_sentiment
from twilio.rest import Client
import asyncio

API_KEY = 'PK2EI3D956LK9CSHIODN'
API_SECRET = 'o4wHHZqWEjwWfRjMl0adCcI28h6B3eKa6qnhyAK7'
BASE_URL = "https://paper-api.alpaca.markets"
STREAM_URL = "wss://stream.data.alpaca.markets/v1beta2/news"
TWILIO_ACCOUNT_SID = 'your_twilio_account_sid'
TWILIO_AUTH_TOKEN = 'your_twilio_auth_token'

class NewsSentimentAnalyzer:
    def __init__(self, symbols: list, days: int = 3, whatsapp_group: str = None):
        self.symbols = symbols
        self.days = days
        self.whatsapp_group = whatsapp_group
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        self.stream = Stream(API_KEY, API_SECRET, base_url=STREAM_URL)
        self.twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    def get_dates(self):
        today = datetime.now()
        prior_date = today - Timedelta(days=self.days)
        return today.strftime('%Y-%m-%d'), prior_date.strftime('%Y-%m-%d')

    def get_sentiment(self, headline):
        probability, sentiment = estimate_sentiment([headline])
        return probability, sentiment

    def send_whatsapp_message(self, message):
        if self.whatsapp_group:
            self.twilio_client.messages.create(
                from_='whatsapp:+<your_twilio_number>',
                body=message,
                to=self.whatsapp_group
            )

    def fetch_and_print_historical_news(self):
        for symbol in self.symbols:
            today, prior_date = self.get_dates()
            news = self.api.get_news(symbol=symbol, start=prior_date, end=today)
            for ev in news:
                headline = ev.headline
                date = ev.created_at
                message = f"Company: {symbol}, Date: {date}, News: {headline}\n"
                probability, sentiment = self.get_sentiment(headline)
                suggestion = self.get_suggestion(sentiment, probability)
                message += f"Sentiment: {sentiment} (Probability: {probability}), Suggestion: {suggestion}\n"
                print(message)
                self.send_whatsapp_message(message)

    async def on_news(self, news):
        headline = news.headline
        date = news.created_at
        message = f"Date: {date}, Real-time News: {headline}\n"
        for symbol in self.symbols:
            probability, sentiment = self.get_sentiment(headline)
            suggestion = self.get_suggestion(sentiment, probability)
            message += f"Company: {symbol}, Real-time Sentiment: {sentiment} (Probability: {probability}), Suggestion: {suggestion}\n"
        print(message)
        self.send_whatsapp_message(message)

    def start_real_time_stream(self):
        for symbol in self.symbols:
            self.stream.subscribe_news(self.on_news, symbol)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.stream._run_forever())

    def get_suggestion(self, sentiment, probability):
        if sentiment == "positive" and probability > 0.999:
            return "buy"
        elif sentiment == "negative" and probability > 0.999:
            return "sell"
        else:
            return "hold"

# Example usage
symbols = ["AAPL", "GOOGL", "NFLX", "AMZN", "FB"]
whatsapp_group = "whatsapp:your_whatsapp_group_id_or_phone_number"
analyzer = NewsSentimentAnalyzer(symbols=symbols, days=3, whatsapp_group=whatsapp_group)
analyzer.fetch_and_print_historical_news()
analyzer.start_real_time_stream()
