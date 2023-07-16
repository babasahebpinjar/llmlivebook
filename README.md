# llmlivebook

# Chatbot with Flask and Slack

This repository contains a chatbot implementation using Flask and Slack. The chatbot leverages the Langchain library for question-answering and text processing tasks.

## Features

- Receives events from Slack using SlackEventAdapter.
- Handles app mentions and responds with a message.
- Implements a conversational retrieval chain for question-answering.
- Integrates with OpenAI GPT-3.5 Turbo for language modeling.
- Utilizes Flask and FastAPI frameworks for building the web server.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>

    Configuration
    Before running the code, you need to configure the following environment variables:

    SLACK_SIGNING_SECRET: Slack app's signing secret.
    SLACK_BOT_TOKEN: Slack bot token for authentication.
    VERIFICATION_TOKEN: Slack verification token.
    OPENAI_API_KEY: OpenAI API key for language modeling.

2.  pip install -r requirements.txt



3.  Usage
    Run the Flask server:

    python bot.py

    Expose the server to the internet using a tool like ngrok.

    Set up the Slack app's Event Subscriptions and provide the ngrok URL as the Request URL.

    Install the Slack app to your workspace.

    Start interacting with the chatbot by mentioning the app in a Slack channel.


4.  Run ngrok for development
5.  To create AWS lambda layer https://medium.com/sopmac-labs/langchain-aws-lambda-serverless-q-a-chatbot-203470b9906f
