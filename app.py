import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain, LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler


class MathAssistantApp:
    def __init__(self):
        self._setup_ui()
        self.groq_api_key = self._get_api_key()
        if not self.groq_api_key:
            st.info("Please add your Groq API key to continue.")
            st.stop()

        self.llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=self.groq_api_key)
        self.agent = self._initialize_agent()
        self._initialize_chat_history()

    def _setup_ui(self):
        st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant", page_icon="🧮")
        st.title("Text To Math Problem Solver Using Google Gemma 2")

    def _get_api_key(self):
        return st.sidebar.text_area("Groq API Key", type='password')

    def _initialize_agent(self):
        wikipedia_tool = Tool(
            name='Wikipedia',
            func=WikipediaAPIWrapper().run,
            description="A tool for searching Wikipedia to find information on a topic."
        )

        math_chain = LLMMathChain.from_llms(llm=self.llm)
        calculator_tool = Tool(
            name="Calculator",
            func=math_chain.run,
            description="A tool for answering math-related questions. Only mathematical expressions need to be provided."
        )

        prompt = """
        You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and provide a detailed explanation point-wise.
        Question: {question}
        Answer:
        """
        reasoning_prompt = PromptTemplate(input_variables=['question'], template=prompt)
        reasoning_chain = LLMChain(llm=self.llm, prompt=reasoning_prompt)

        reasoning_tool = Tool(
            name="Reasoning tool",
            func=reasoning_chain.run,
            description="A tool for answering logic-based and reasoning questions."
        )

        return initialize_agent(
            tools=[wikipedia_tool, calculator_tool, reasoning_tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

    def _initialize_chat_history(self):
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {'role': "assistant", "content": "Hi, I'm a Math chatbot who can answer all your maths questions."}
            ]

    def _render_chat_history(self):
        for msg in st.session_state["messages"]:
            st.chat_message(msg['role']).write(msg['content'])

    def run(self):
        self._render_chat_history()

        question = st.text_area("Enter your question:", 
                                "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

        if st.button("Find my answer"):
            if question:
                with st.spinner("Generating response..."):
                    st.session_state["messages"].append({"role": "user", "content": question})
                    st.chat_message("user").write(question)

                    streamlit_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                    response = self.agent.run(st.session_state["messages"], callbacks=[streamlit_callback])

                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    st.write("### Response:")
                    st.success(response)
            else:
                st.warning("Please enter the question.")


if __name__ == "__main__":
    app = MathAssistantApp()
    app.run()
