# from alpaca_trade_api import REST

API_KEY = 'PK2EI3D956LK9CSHIODN'
API_SECRET = 'o4wHHZqWEjwWfRjMl0adCcI28h6B3eKa6qnhyAK7'
# BASE_URL = "https://paper-api.alpaca.markets"

# # Initialize the REST client
# rest_client = REST(API_KEY, API_SECRET)

# def print_news_with_company_and_date(company_symbol, start_date, end_date):
#     # Fetch historical news data for the specified company and date range
#     news = rest_client.get_news(company_symbol, start_date, end_date)
    
#     # Print news along with company name and date
#     for article in news:
#         print(f"Company: {company_symbol}")
#         print(f"Date: {article.created_at}")
#         print(f"Headline: {article.headline}")
#         print(f"Summary: {article.summary}")
#         print("")

# # Example usage
# company_symbol = "AAPL"  # Replace with the desired company symbol
# start_date = "2020-01-01"  # Replace with the start date
# end_date = "2022-01-31"  # Replace with the end date

# print_news_with_company_and_date(company_symbol, start_date, end_date)

from alpaca_trade_api import Stream, REST
import asyncio


# Initialize the REST client
rest_client = REST(API_KEY, API_SECRET)

# Initialize the streaming client
stream_client = Stream(API_KEY, API_SECRET)

# Replace 'AAPL' with the desired company symbol
company_symbol = 'AAPL'

# Define a news data handler function
async def news_data_handler(news):
    print(f"Company: {company_symbol}")
    print(f"Date: {news.created_at}")
    print(f"Headline: {news.headline}")
    print(f"Summary: {news.summary}")
    print("")

# Subscribe to live news data for the specified company
stream_client.subscribe_news(news_data_handler, company_symbol)

# Run the streaming client
stream_client.run()

# Wait indefinitely
asyncio.get_event_loop().run_forever()
