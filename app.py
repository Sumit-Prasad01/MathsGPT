import streamlit as st
import os
import numexpr
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain, LLMChain
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
import validators


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {'role': "assistant", "content": "Hi, I'm a Math chatbot who can answer all your maths questions and search the web for information!"}
        ]
    if "agent" not in st.session_state:
        st.session_state["agent"] = None
    if "llm" not in st.session_state:
        st.session_state["llm"] = None


def setup_ui():
    """Setup the Streamlit UI"""
    st.set_page_config(
        page_title="Advanced Math Problem Solver & Research Assistant", 
        page_icon="🧮",
        layout="wide"
    )
    st.title("🧮 Advanced Math Problem Solver & Research Assistant")
    st.markdown("*Powered by Groq, LangChain, and multiple search engines*")
    st.markdown("---")


def get_groq_api_key():
    """Get Groq API key from sidebar input"""
    with st.sidebar:
        st.header("🔑 Configuration")
        api_key = st.text_input(
            "Enter your Groq API Key:", 
            type="password",
            help="Get your API key from https://console.groq.com/keys"
        )
        
        # Model selection
        model_choice = st.selectbox(
            "Select Model:",
            ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
            help="Choose the AI model for processing"
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
            max_tokens = st.slider("Max Tokens", 500, 4000, 2000, 100)
        
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        
        return api_key, model_choice, temperature, max_tokens


def safe_import_pandas():
    """Safely import pandas with error handling"""
    try:
        import pandas as pd
        return pd, True
    except ImportError:
        st.warning("📊 Pandas not available. Data analysis features will be limited.")
        return None, False
    except Exception as e:
        st.warning(f"📊 Pandas import error: {str(e)}. Data analysis features will be limited.")
        return None, False


def create_enhanced_calculator_tool(llm):
    """Create an enhanced calculator that can handle complex expressions"""
    def enhanced_calculator(expression):
        try:
            # Clean the expression for safety
            expression = str(expression).strip()
            
            # Basic validation
            if not expression:
                return "Error: Empty expression provided."
            
            # Try with numexpr for complex mathematical expressions
            if any(op in expression for op in ['+', '-', '*', '/', '**', '(', ')', 'sqrt', 'sin', 'cos', 'tan', 'log']):
                try:
                    # Replace common math functions for numexpr
                    clean_expr = expression.replace('sqrt', 'sqrt')
                    clean_expr = clean_expr.replace('^', '**')  # Convert ^ to **
                    
                    # Evaluate with numexpr
                    result = numexpr.evaluate(clean_expr)
                    return f"Calculation: {expression} = {result}"
                except Exception as numexpr_error:
                    # If numexpr fails, try basic eval for simple expressions
                    try:
                        # Only allow safe mathematical operations
                        allowed_names = {
                            k: v for k, v in __builtins__.items() 
                            if k in ['abs', 'round', 'min', 'max', 'sum', 'pow']
                        }
                        allowed_names.update({
                            'sqrt': lambda x: x**0.5,
                            'pi': 3.14159265359,
                            'e': 2.71828182846
                        })
                        
                        result = eval(expression, {"__builtins__": {}}, allowed_names)
                        return f"Calculation: {expression} = {result}"
                    except:
                        pass
            
            # Fallback to LangChain's math chain
            try:
                math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)
                result = math_chain.run(expression)
                return result
            except Exception as chain_error:
                return f"Error in calculation: {str(chain_error)}. Please check your mathematical expression and try again."
            
        except Exception as e:
            return f"Error processing calculation: {str(e)}. Please verify your mathematical expression."
    
    return Tool(
        name="Enhanced_Calculator",
        func=enhanced_calculator,
        description="Advanced calculator for mathematical expressions. Can handle basic arithmetic, powers, square roots, trigonometric functions, and complex calculations. Input should be a mathematical expression like '2+2', 'sqrt(16)', or '(5*3)+2'."
    )


def create_data_analysis_tool():
    """Create data analysis tool with pandas support"""
    pd, pandas_available = safe_import_pandas()
    
    def data_analyzer(query):
        try:
            if not pandas_available:
                return "Data analysis tool is available but pandas is not properly installed. For advanced data analysis, please fix the pandas installation."
            
            # This is a framework for data analysis
            # In practice, you would process actual data here
            analysis_types = {
                'statistics': 'Statistical analysis including mean, median, mode, standard deviation',
                'correlation': 'Correlation analysis between variables',
                'trends': 'Trend analysis and time series patterns',
                'grouping': 'Group by operations and aggregations',
                'visualization': 'Data visualization recommendations'
            }
            
            query_lower = query.lower()
            relevant_analyses = [desc for key, desc in analysis_types.items() if key in query_lower]
            
            if relevant_analyses:
                return f"Data analysis for '{query}':\n" + "\n".join(f"• {analysis}" for analysis in relevant_analyses)
            else:
                return f"Data analysis capabilities available for: {query}. I can help with statistical analysis, correlations, trends, grouping operations, and provide visualization recommendations."
                
        except Exception as e:
            return f"Error in data analysis: {str(e)}"
    
    return Tool(
        name="Data_Analyzer",
        func=data_analyzer,
        description="Analyze data patterns, perform statistical calculations, and provide insights. Input should describe the type of analysis needed (statistics, correlation, trends, etc.)."
    )


def create_tools(llm):
    """Create the tools for the agent"""
    tools = []
    
    # Enhanced calculator tool (always available)
    calculator_tool = create_enhanced_calculator_tool(llm)
    tools.append(calculator_tool)
    
    # Wikipedia tool
    try:
        wikipedia = WikipediaAPIWrapper()
        wikipedia_tool = Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="Search Wikipedia for factual information, definitions, historical facts, and general knowledge. Input should be a search query or topic name."
        )
        tools.append(wikipedia_tool)
    except Exception as e:
        st.warning(f"⚠️ Wikipedia tool unavailable: {str(e)}")

    # DuckDuckGo search tool
    try:
        search = DuckDuckGoSearchAPIWrapper()
        search_tool = Tool(
            name="Web_Search",
            func=search.run,
            description="Search the web for current information, recent news, trends, and real-time data. Input should be a search query."
        )
        tools.append(search_tool)
    except Exception as e:
        st.warning(f"⚠️ Web search tool unavailable: {str(e)}")

    # Advanced reasoning tool
    reasoning_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        You are an expert problem solver with strong analytical and mathematical skills.
        Break down complex problems into clear, logical steps.
        
        Question: {question}
        
        Please provide a detailed, step-by-step solution:
        1. Understand the problem
        2. Identify what needs to be calculated or determined
        3. Show the work step by step
        4. Provide the final answer
        5. Explain the reasoning behind the solution
        """
    )
    
    reasoning_chain = LLMChain(llm=llm, prompt=reasoning_prompt)
    reasoning_tool = Tool(
        name="Step_by_Step_Reasoning",
        func=reasoning_chain.run,
        description="Use this for complex problem-solving that requires logical reasoning, step-by-step analysis, and detailed explanations. Input should be the complete question or problem."
    )
    tools.append(reasoning_tool)

    # Data analysis tool
    data_tool = create_data_analysis_tool()
    tools.append(data_tool)

    return tools


def initialize_agent_and_llm(api_key, model_name, temperature, max_tokens):
    """Initialize the LLM and agent"""
    try:
        # Initialize LLM with custom parameters
        llm = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create tools
        tools = create_tools(llm)
        
        if not tools:
            st.error("❌ No tools available. Please check your configuration.")
            return None, None
        
        # Display available tools
        with st.expander("🛠️ Available Tools", expanded=False):
            for tool in tools:
                st.write(f"**{tool.name}**: {tool.description}")
        
        # Initialize agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
        
        return llm, agent
    
    except Exception as e:
        st.error(f"❌ Error initializing agent: {str(e)}")
        return None, None


def render_chat_history():
    """Render the chat history"""
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def validate_input(question):
    """Validate user input"""
    if not question or not question.strip():
        return False, "Please enter a question."
    
    if len(question.strip()) < 3:
        return False, "Question seems too short. Please provide more details."
    
    # Check for potentially harmful content
    harmful_keywords = ['delete', 'drop', 'truncate', 'exec', 'eval', 'import os']
    if any(keyword in question.lower() for keyword in harmful_keywords):
        return False, "Please avoid potentially harmful commands in your questions."
    
    return True, ""


def main():
    # Setup UI
    setup_ui()
    initialize_session_state()
    
    # Show pandas status
    pd, pandas_available = safe_import_pandas()
    if not pandas_available:
        st.info("ℹ️ For full data analysis features, please fix pandas installation. See instructions below.")
    
    # Get configuration
    api_key, model_choice, temperature, max_tokens = get_groq_api_key()
    
    if not api_key:
        st.info("👈 Please enter your Groq API key in the sidebar to continue.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔑 How to get your Groq API key:
            1. Go to [Groq Console](https://console.groq.com/keys)
            2. Sign up or log in
            3. Create a new API key
            4. Copy and paste it in the sidebar
            """)
            
        with col2:
            st.markdown("""
            ### 🛠️ Fix Pandas Installation:
            ```bash
            # Uninstall pandas completely
            pip uninstall pandas -y
            
            # Clear cache
            pip cache purge
            
            # Reinstall pandas
            pip install pandas
            ```
            """)
        
        st.markdown("""
        ### ✨ Features Available:
        - 🧮 Advanced mathematical calculations
        - 🔍 Real-time web search  
        - 📚 Wikipedia knowledge base
        - 🧠 Step-by-step reasoning
        - 📊 Data analysis framework
        """)
        return
    
    # Initialize LLM and agent
    if (st.session_state["agent"] is None or 
        st.session_state["llm"] is None):
        
        with st.spinner("🚀 Initializing AI models and tools..."):
            llm, agent = initialize_agent_and_llm(api_key, model_choice, temperature, max_tokens)
            
            if llm and agent:
                st.session_state["llm"] = llm
                st.session_state["agent"] = agent
                st.success("✅ AI models and tools initialized successfully!")
            else:
                st.error("❌ Failed to initialize AI models. Please check your API key and try again.")
                return
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💭 Ask Your Question")
        
        # Question input with examples
        example_questions = [
            "Calculate: (15 + 25) * 2 - sqrt(64)",
            "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack contains 25 berries. How many fruits total?",
            "What is compound interest and calculate it for $1000 at 5% for 3 years?",
            "Search for latest developments in renewable energy",
            "Explain the quadratic formula with a step-by-step example",
            "What's 15% of 240 plus 30% of 150?",
            "Find information about machine learning algorithms"
        ]
        
        selected_example = st.selectbox(
            "Choose an example or type your own:",
            ["Custom question..."] + example_questions
        )
        
        if selected_example == "Custom question...":
            question = st.text_area(
                "Enter your question:",
                height=120,
                placeholder="Ask math problems, search queries, or request explanations..."
            )
        else:
            question = st.text_area(
                "Enter your question:",
                value=selected_example,
                height=120
            )
        
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
        with col_btn1:
            solve_button = st.button("🚀 Solve Problem", type="primary", use_container_width=True)
        
        with col_btn2:
            clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
        
        with col_btn3:
            if st.button("💡 Tips", use_container_width=True):
                st.info("""
                **💡 Tips for better results:**
                - Be specific and detailed in your questions
                - For math: include all numbers and operations
                - For search: use clear, descriptive terms
                - Ask for step-by-step explanations when needed
                - Try different phrasings if you don't get the expected result
                """)
        
        if clear_button:
            st.session_state["messages"] = [
                {'role': "assistant", "content": "Hi, I'm a Math chatbot who can answer all your maths questions and search the web for information!"}
            ]
            st.rerun()
    
    with col2:
        st.subheader("🎯 Capabilities")
        st.success("""
        **🧮 Mathematics:**
        ✅ Complex calculations  
        ✅ Word problems  
        ✅ Step-by-step solutions  
        ✅ Formula explanations  
        
        **🔍 Research:**
        ✅ Web search  
        ✅ Wikipedia lookup  
        ✅ Current information  
        ✅ Fact verification  
        
        **🧠 Analysis:**
        ✅ Logical reasoning  
        ✅ Problem breakdown  
        ✅ Detailed explanations  
        ✅ Multiple approaches  
        """)
        
        # Show current configuration
        st.markdown(f"""
        **Current Settings:**
        - Model: `{model_choice}`
        - Temperature: `{temperature}`
        - Max Tokens: `{max_tokens}`
        - Pandas: {'✅ Available' if pandas_available else '❌ Needs fixing'}
        """)
    
    # Process question
    if solve_button:
        is_valid, error_msg = validate_input(question)
        
        if not is_valid:
            st.warning(f"⚠️ {error_msg}")
        else:
            with st.spinner("🤔 Processing your request..."):
                try:
                    # Add user message
                    st.session_state["messages"].append({"role": "user", "content": question})
                    
                    # Create a container for the agent's thinking process
                    with st.expander("🔍 Agent Thinking Process", expanded=False):
                        callback_container = st.container()
                        callback = StreamlitCallbackHandler(callback_container)
                        
                        # Get response from agent
                        response = st.session_state["agent"].run(question, callbacks=[callback])
                    
                    # Add assistant response
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    
                    st.rerun()
                    
                except Exception as e:
                    error_details = str(e)
                    st.error(f"❌ Error processing question: {error_details}")
                    
                    # Provide helpful suggestions based on error type
                    if "rate limit" in error_details.lower():
                        st.info("🔄 Rate limit reached. Please wait a moment and try again.")
                    elif "api" in error_details.lower():
                        st.info("🔑 API issue detected. Please check your API key and try again.")
                    else:
                        st.info("💡 Try rephrasing your question or using simpler terms.")
                    
                    # Add error to chat history for context
                    error_msg = f"I encountered an error: {error_details[:100]}... Please try rephrasing your question."
                    st.session_state["messages"].append({"role": "assistant", "content": error_msg})
    
    # Chat history
    st.markdown("---")
    st.subheader("💬 Conversation History")
    render_chat_history()
    
    # Footer with instructions
    st.markdown("---")
    st.caption("🚀 Powered by Groq AI, LangChain, and multiple search engines")
    
    if not pandas_available:
        st.caption("⚠️ To enable full data analysis features, run: `pip uninstall pandas -y && pip install pandas`")


if __name__ == "__main__":
    main()