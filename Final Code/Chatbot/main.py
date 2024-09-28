import time
import random
import streamlit as st
from finbert_utils import estimate_sentiment  # Importing function from custom module

st.title('Trade :blue[Genie] :robot_face:')  # Setting the title of the Streamlit app

# Function to generate a random response
def response_generator():
    response = random.choice(  # Randomly select a response
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():  # Iterate through each word in the response
        yield word + " "  # Yield each word with a trailing space
        time.sleep(0.05)  # Introduce a slight delay for simulated typing

# Function to analyze a headline and provide a trading suggestion
def analyze_headline(headline):
    try:
        probability, sentiment = estimate_sentiment([headline])  # Estimate sentiment of the headline
        probability_float = float(probability)  # Convert probability tensor to float
        probability_rounded = round(probability_float, 4)  # Round probability to 4 decimal places
        
        suggestion = ""
        if sentiment.lower() == "positive" and probability_rounded > 0.80:
            suggestion = "Buy"  # Suggest "Buy" if sentiment is positive and confidence is high
        elif sentiment.lower() == "negative" and probability_rounded > 0.80:
            suggestion = "Sell"  # Suggest "Sell" if sentiment is negative and confidence is high
        else:
            suggestion = "Hold"  # Otherwise, suggest "Hold"
        
        # Prepare formatted output with headline, probability, sentiment, and suggestion
        output = f"**Headline:** {headline}\n**Probability:** {probability_rounded:.4f}\n**Sentiment:** {sentiment}\n**Suggestion:** {suggestion}"
        
        return output  # Return the formatted output string
    
    except Exception as e:
        st.error(f"Error analyzing headline: {str(e)}")  # Display error if sentiment analysis fails
        return None  # Return None if analysis fails

# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # Display each message based on its role
        st.markdown(message["content"])  # Show the content of each message

# Get user input message
prompt = st.chat_input("How can I assist you? ")

if not prompt:  # If prompt is empty, generate and display a random response
    st.chat_message("assistant").write(response_generator())
else:
    st.session_state.messages.append({"role": "user", "content": prompt})  # Store user message in session state
    with st.chat_message("user"):  # Display user message in the chat interface
        st.markdown(prompt)

    response = analyze_headline(prompt)  # Analyze user's input headline

    if response:  # If analysis is successful, display the response
        st.session_state.messages.append({"role": "assistant", "content": response})  # Store assistant response
        with st.chat_message("assistant"):  # Display assistant's response in the chat interface
            st.markdown(response)
