import streamlit as st
import time
from datetime import datetime
from config import Config
from auth import Authenticator
from data_loader import PolicyDataLoader
from rag_system import PolicyRAGSystem
from query_handler import QueryHandler
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="Bank Policy Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f4e79;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0;
        margin-left: 20%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        margin-right: 20%;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
    }
    
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 1px solid #e0e0e0;
    }
    
    .welcome-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 10px 15px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 10px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

class BankPolicyWebApp:
    def __init__(self):
        self.config = Config()
        
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        if 'authenticator' not in st.session_state:
            st.session_state.authenticator = None
        if 'query_handler' not in st.session_state:
            st.session_state.query_handler = None

    def initialize_system(self):
        """Initialize the RAG system components"""
        if st.session_state.system_initialized:
            return True
            
        try:
            with st.spinner("🔄 Initializing Bank Policy Assistant..."):
                # Configure Gemini API
                genai.configure(api_key=self.config.GEMINI_API_KEY)
                
                # Initialize document loader
                document_loader = PolicyDataLoader(self.config.POLICY_PDF_PATH)
                document_loader.load_all_documents()
                
                # Initialize RAG system
                rag_system = PolicyRAGSystem(
                    retriever=document_loader.get_retriever(),
                    config=self.config,
                    data_loader=document_loader
                )
                rag_system.initialize_llm()
                rag_system.setup_qa_chain()
                
                # Store in session state
                st.session_state.rag_system = rag_system
                st.session_state.system_initialized = True
                
                st.success("✅ System initialized successfully!")
                time.sleep(1)
                return True
                
        except Exception as e:
            st.error(f"❌ System initialization failed: {str(e)}")
            return False

    def show_login_page(self):
        """Display the login interface"""
        st.markdown('<h1 class="main-header">🏦 Bank Policy Assistant</h1>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            
            st.markdown("### 🔐 Employee Login")
            st.markdown("---")
            
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("👤 Username", placeholder="Enter your username")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    submit_button = st.form_submit_button("🚀 Login", use_container_width=True)
                
                if submit_button:
                    if username and password:
                        try:
                            authenticator = Authenticator(self.config)
                            if authenticator.authenticate(username, password):
                                user = authenticator.get_authenticated_user()
                                st.session_state.authenticated = True
                                st.session_state.user = user
                                st.session_state.authenticator = authenticator
                                st.success("✅ Authentication successful!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Invalid credentials. Please try again.")
                        except Exception as e:
                            st.error(f"⚠️ Authentication error: {str(e)}")
                    else:
                        st.warning("⚠️ Please enter both username and password.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    def show_chat_interface(self):
        """Display the main chat interface"""
        # Initialize system if not already done
        if not self.initialize_system():
            return
            
        # Setup query handler
        if not st.session_state.query_handler:
            st.session_state.query_handler = QueryHandler(
                st.session_state.user, 
                st.session_state.rag_system, 
                st.session_state.authenticator
            )
        
        # Header with user info and logout
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<h1 class="main-header">🏦 Bank Policy Assistant</h1>', unsafe_allow_html=True)
        with col2:
            st.markdown("---")
            st.write(f"👤 **{st.session_state.user['username']}**")
            st.write(f"🏷️ **Grade:** {st.session_state.user.get('grade', 'N/A')}")
            if st.button("🚪 Logout"):
                self.logout()

        # Welcome banner
        st.markdown(f"""
        <div class="welcome-banner">
            <h3>Welcome back, {st.session_state.user['username']}! 👋</h3>
            <p>Ask me anything about bank policies, leave applications, or HR procedures.</p>
        </div>
        """, unsafe_allow_html=True)

        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Query input
        st.markdown("---")
        
        # Quick action buttons



        # Main query input
        with st.form("query_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_query = st.text_input(
                    "💬 Ask your question:", 
                    placeholder="e.g., Ask about workplace harassment policy, Internal Committee roles, complaint process, or filing timelines",
                    label_visibility="collapsed"
                )
            with col2:
                submit = st.form_submit_button("📤 Send", use_container_width=True)
            
            if submit and user_query:
                self.process_query(user_query)

    def process_query(self, query):
        """Process user query and update chat"""
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        try:
            # Get response from query handler
            with st.spinner("Thinking..."):
                response = st.session_state.query_handler.handle_query(query)
            
            # Add assistant response to chat
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"⚠️ Sorry, there was an error processing your request: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        # Rerun to update the interface
        st.rerun()

    def logout(self):
        """Handle user logout"""
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("👋 Logged out successfully!")
        time.sleep(1)
        st.rerun()

    def run(self):
        """Main application runner"""
        load_css()
        
        if st.session_state.authenticated:
            self.show_chat_interface()
        else:
            self.show_login_page()

# Run the application
if __name__ == "__main__":
    app = BankPolicyWebApp()
    app.run()