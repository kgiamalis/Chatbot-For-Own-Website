
# README for Streamlit Chatbot Interface with OpenAI

## Overview

This Streamlit application provides an interactive interface for a chatbot powered by OpenAI. The chatbot is designed to retrieve and provide responses based on the content loaded from a CSV file named `personal_posts.csv`. The application uses OpenAI's embeddings and a vector store to efficiently fetch relevant responses.

## Features

- **CSV Data Loading**: Loads conversational data from `personal_posts.csv`.
- **Text Chunking**: Splits large textual data into manageable chunks for efficient processing.
- **OpenAI Integration**: Uses OpenAI's API for embeddings and conversation modeling.
- **Interactive UI**: Provides a web interface for users to interact with the chatbot.

## Setup

### Prerequisites

- Python 3.x
- Streamlit
- OpenAI Python SDK

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key. You can do this by setting up Streamlit secrets or exporting it as an environment variable:

```bash
export OPENAI_API_KEY=your_openai_key
```

### Running the App

To run the Streamlit app, execute the following command:

```bash
streamlit run streamlitapp.py
```

## Usage

1. Open the provided link in your browser after running the app.
2. Click on the "Start" button to initiate the chatbot.
3. Type your question or statement in the input box and press "Enter" or click "Submit".
4. The chatbot will provide a response based on the content from `personal_posts.csv`.
5. You can continue the conversation with the bot, and the chat history will be displayed on the screen.

## Contributing

Feel free to fork the repository, make changes, and submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for details.
