from alpaca_trade_api import Stream
import asyncio
from datetime import datetime

# Alpaca API credentials
API_KEY = 'PK2EI3D956LK9CSHIODN'
API_SECRET = 'o4wHHZqWEjwWfRjMl0adCcI28h6B3eKa6qnhyAK7'

# Initialize the streaming client
stream_client = Stream(API_KEY, API_SECRET)

# List of company symbols
company_symbols = ["AAPL", "GOOGL", "NFLX", "AMZN", "FB"]

# Define a news data handler function
async def news_data_handler(news):
    print(f"Company: {news.symbol}")
    print(f"Date: {news.created_at}")
    print(f"Headline: {news.headline}")
    print(f"Summary: {news.summary}")
    print("")

# Function to print a message every 5 seconds with a timestamp
async def print_live_message():
    while True:
        await asyncio.sleep(5)
        print(f"{datetime.now().isoformat()} - The code is live...")

# Function to run the streaming client
async def run_stream():
    try:
        # Subscribe to live news data for each company in the list
        for symbol in company_symbols:
            stream_client.subscribe_news(news_data_handler, symbol)
        
        # Run the streaming client
        await stream_client._run_forever()
    except Exception as e:
        print(f"An error occurred: {e}. Restarting stream...")
        await asyncio.sleep(5)
        await run_stream()

# Main function to start the stream and print the live message
async def main():
    # Start the live message printer
    asyncio.create_task(print_live_message())
    
    # Run the streaming client
    await run_stream()

# Run the main function
asyncio.run(main())
