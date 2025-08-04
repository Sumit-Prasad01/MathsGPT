# 🧮 Math Problem Solver & Research Assistant

This is a powerful Streamlit-based AI assistant that can solve complex mathematical problems, provide logical reasoning, search the web, analyze data, and fetch factual information using Wikipedia and DuckDuckGo.

It integrates Groq's LLMs with LangChain agents and tools to offer real-time interactive problem-solving with step-by-step explanations.

---

## 🚀 Features

- **Advanced Calculations** – Solve arithmetic, algebra, calculus, and other math problems.
- **Step-by-Step Reasoning** – Get detailed, logical explanations to problems.
- **Web Search** – Real-time information using DuckDuckGo.
- **Wikipedia Integration** – Quick access to encyclopedic knowledge.
- **Data Analysis** – Framework to process and suggest data insights (requires `pandas`).
- **Enhanced Calculator** – Supports `numexpr` for secure and complex math evaluation.
- **Interactive Chat Interface** – Built with Streamlit and stateful memory.

---

## 📦 Prerequisites

- Python 3.8+
- A valid **Groq API key** ([Get it here](https://console.groq.com/keys))

---

## 🛠️ Installation

```bash
pip install streamlit langchain langchain-groq langchain_community pandas numexpr validators
```

---

## ▶️ Running the App

```bash
streamlit run math_assistant_fixed.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Code Structure

- `initialize_session_state()` – Sets initial app memory.
- `setup_ui()` – Builds the Streamlit layout and sidebars.
- `get_groq_api_key()` – Configures API credentials and model settings.
- `create_enhanced_calculator_tool()` – Safe evaluation of expressions.
- `create_data_analysis_tool()` – Simulated data insights using pandas.
- `create_tools()` – Registers all LangChain tools.
- `initialize_agent_and_llm()` – Loads model and tools as a LangChain agent.
- `validate_input()` – Checks user queries for safety and completeness.
- `main()` – App entry point. Manages flow and UI interactions.

---

## 💬 Usage Tips

- Use clear and structured queries for best results.
- Ask for "step-by-step" to get detailed reasoning.
- Example prompts:
  - "Calculate the integral of x^2"
  - "What is the capital of France?"
  - "Analyze data trends in sales by month"

---

