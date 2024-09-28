from datetime import datetime, timedelta
from alpaca_trade_api import REST, Stream
from finbert_utils import estimate_sentiment
import pywhatkit
import asyncio
import time

API_KEY = 'PK2EI3D956LK9CSHIODN'
API_SECRET = 'o4wHHZqWEjwWfRjMl0adCcI28h6B3eKa6qnhyAK7'
BASE_URL = "https://paper-api.alpaca.markets"
STREAM_URL = "wss://stream.data.alpaca.markets/v1beta2/news"


class NewsSentimentAnalyzer:
    def __init__(self, symbol: str = "AAPL", days: int = 3, group_id: str = "your_group_id"):
        self.symbol = symbol
        self.days = days
        self.group_id = group_id
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        self.stream = Stream(API_KEY, API_SECRET, base_url=STREAM_URL)

    def get_dates(self):
        today = datetime.now().date()
        prior_date = today - timedelta(days=self.days)
        return today, prior_date

    def get_sentiment(self, headline):
        probability, sentiment = estimate_sentiment([headline])
        return probability, sentiment

    def fetch_and_send_historical_news(self):
        today, prior_date = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=prior_date.isoformat(), end=today.isoformat())
        for ev in news:
            headline = ev.headline
            date = ev.created_at
            sentiment_info = self.get_sentiment(headline)
            suggestion = self.get_suggestion(sentiment_info[1], sentiment_info[0])
            # Split date and time into two lines
            formatted_date = date.date()
            formatted_time = date.time()
            message = (f"*Company:* {self.symbol}\n*Date:* {formatted_date}\n*Time:* {formatted_time}\n"
                       f"*News:*\n{headline}\n"
                       f"*Sentiment:* {sentiment_info[1]} (*Probability:* {sentiment_info[0]})\n"
                       f"*Suggestion:* {suggestion}")
            self.send_whatsapp_message(message)

    def get_suggestion(self, sentiment, probability):
        if sentiment == "positive" and probability > 0.999:
            return "*Buy*"
        elif sentiment == "negative" and probability > 0.999:
            return "*Sell*"
        else:
            return "*Hold*"
        

    async def on_news(self, news):
        headline = news.headline
        date = news.created_at
        sentiment_info = self.get_sentiment(headline)
        suggestion = self.get_suggestion(sentiment_info[1], sentiment_info[0])
        # Split date and time into two lines
        formatted_date = date.date()
        formatted_time = date.time()
        message = (f"*Company:* {self.symbol}\n*Date:* {formatted_date}\n*Time:* {formatted_time}\n"
                   f"*Real-time News:*\n{headline}\n"
                   f"*Real-time Sentiment:* {sentiment_info[1]}*Company:* AAPL*D (*Probability:* {sentiment_info[0]})\n"
                   f"*Suggestion:* {suggestion}")
        self.send_whatsapp_message(message)

    def start_real_time_stream(self):
        self.stream.subscribe_news(self.on_news, self.symbol)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.stream._run_forever())

    def send_whatsapp_message(self, message):
        try:
            pywhatkit.sendwhatmsg_to_group_instantly(self.group_id, message, tab_close=True)
            print("Message sent successfully!")
            
            # Add a delay before closing WhatsApp
            time.sleep(5)  # Adjust the delay as needed
            
            print("WhatsApp closed.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Example usage
group_id = "JfVqbe4ZepL4N8X59LmtmO"  # Replace with your group's IDate:* 2024-05-20*T
analyzer = NewsSentimentAnalyzer(symbol="AAPL", days=3, group_id=group_id)
analyzer.fetch_and_send_historical_news()
analyzer.start_real_time_stream()
