# Text to Math Problem Solver and Data Search Assistant

This is a Streamlit application that acts as a **math problem-solving assistant**. It uses a **LangChain agent powered by Groq's Gemma2-9b-It model** to answer complex questions by leveraging specialized tools for calculation, reasoning, and data searching.

The application is built with a clean, object-oriented structure, making it easy to understand and extend.

---

## 🚀 Features

- **Intelligent Agent**: Utilizes a LangChain agent to break down user queries.
- **Multi-tool Capability**: The agent can use the following tools:
  - **Calculator**: Solves complex mathematical expressions.
  - **Wikipedia**: Searches for factual information on the web.
  - **Reasoning Tool**: Provides logical, point-wise explanations for math problems.
- **Groq Integration**: Uses the fast and efficient **Gemma2-9b-It** language model from Groq.
- **Streamlit UI**: Provides a simple and interactive chat interface for users to ask questions.

---

## 📦 Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.8+
- A valid **Groq API key**

---

## 🛠️ Installation

Clone this repository (if applicable) or save the Python code to a file named `app.py`.

Install the required Python packages using pip:

```bash
pip install streamlit langchain langchain-groq langchain_community
```

#💡 Usage
- On the sidebar, enter your Groq API Key.

- In the text area, type your math question or any query that requires calculation or data search.

- Click the "Find my answer" button.

- The application will process your request and display a detailed, point-wise response in the chat window.

# Code Structure
The application logic is encapsulated in the MathAssistantApp class using an Object-Oriented Programming (OOP) approach:

- __init__(): Initializes the Streamlit UI, gets the API key, and sets up the LangChain agent.

- _setup_tools(): A private method that defines and returns the list of tools (Wikipedia, Calculator, Reasoning Tool).

- _setup_agent(): A private method that initializes and returns the LangChain agent with the configured tools and LLM.

- _display_chat_history(): Renders all messages stored in the st.session_state.

- _handle_user_input(): Manages user input, calls the agent, and updates the chat history with the response.

- run(): The main entry point that runs the application, calling all display and input methods.
