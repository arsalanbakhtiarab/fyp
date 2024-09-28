from alpaca_trade_api import REST
from datetime import datetime, timedelta

API_KEY = 'PK2EI3D956LK9CSHIODN'
API_SECRET = 'o4wHHZqWEjwWfRjMl0adCcI28h6B3eKa6qnhyAK7'
BASE_URL = "https://paper-api.alpaca.markets"

def fetch_news(start_date, end_date, symbol="SPY"):
    api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    news = api.get_news(symbol=symbol, start=start_str, end=end_str)
    headlines = [ev._raw["headline"] for ev in news]
    return headlines

if __name__ == "__main__":
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    news_headlines = fetch_news(start_date, end_date)
    for headline in news_headlines:
        print(headline,"\n")
