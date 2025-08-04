import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

class MathAssistantApp:
    """
    A Streamlit application for a math problem-solving assistant using LangChain and Groq.

    This class encapsulates all the logic for setting up the LLM, tools, and agent,
    as well as rendering the Streamlit UI and handling user interactions.
    """

    def __init__(self):
        """Initializes the Streamlit app, API key, LLM, and agent."""
        st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant", page_icon="🧮")
        st.title("Text To Math Problem Solver Using Google Gemma 2")

        # Get the Groq API key from the sidebar
        self.groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

        # Stop the app if the API key is not provided
        if not self.groq_api_key:
            st.info("Please add your Groq API key to continue.")
            st.stop()

        # Initialize the language model
        self.llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=self.groq_api_key)

        # Initialize session state for chat messages
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your maths questions."}
            ]
        
        # Setup the agent with its tools
        self.agent = self._setup_agent()

    def _setup_tools(self):
        """Sets up and returns a list of tools for the agent."""
        
        # Initialize the Wikipedia tool
        wikipedia_wrapper = WikipediaAPIWrapper()
        wikipedia_tool = Tool(
            name="Wikipedia",
            func=wikipedia_wrapper.run,
            description="A tool for searching the Internet to find various information on the topics mentioned."
        )

        # Initialize the math tool (calculator)
        math_chain = LLMMathChain.from_llm(llm=self.llm)
        calculator = Tool(
            name="Calculator",
            func=math_chain.run,
            description="A tool for answering math-related questions. Only input mathematical expressions need to be provided."
        )

        # Define the reasoning tool
        prompt = """
        You're an agent tasked with solving users' mathematical questions. Logically arrive at the solution, provide a detailed explanation,
        and display it point-wise for the question below.
        Question: {question}
        Answer:
        """
        prompt_template = PromptTemplate(input_variables=["question"], template=prompt)
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        reasoning_tool = Tool(
            name="Reasoning tool",
            func=chain.run,
            description="A tool for answering logic-based and reasoning questions."
        )
        
        return [wikipedia_tool, calculator, reasoning_tool]

    def _setup_agent(self):
        """Initializes and returns the LangChain agent."""
        tools = self._setup_tools()
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )
        return agent

    def _display_chat_history(self):
        """Renders the chat messages from the session state."""
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg['content'])

    def _handle_user_input(self):
        """Handles the user's input, agent response, and updates the chat history."""
        question = st.text_area(
            "Enter your question:",
            "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?"
        )

        if st.button("Find my answer"):
            if question:
                with st.spinner("Generating response..."):
                    # Add user message to state and display it
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.chat_message("user").write(question)

                    # Get response from the agent
                    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                    response = self.agent.run(st.session_state.messages, callbacks=[st_cb])
                    
                    # Add assistant message to state and display it
                    st.session_state.messages.append({'role': 'assistant', "content": response})
                    st.write('### Response:')
                    st.success(response)
            else:
                st.warning("Please enter a question.")

    def run(self):
        """Main method to run the application."""
        self._display_chat_history()
        self._handle_user_input()

# Main execution block
if __name__ == "__main__":
    app = MathAssistantApp()
    app.run()
