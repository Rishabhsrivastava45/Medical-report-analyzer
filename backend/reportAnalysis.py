from dotenv import load_dotenv
load_dotenv()

import os
import gradio as gr
from typing import Dict, List, Optional, TypedDict, Annotated
import json
from datetime import datetime
import tempfile
import base64

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from openai import OpenAI

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("[!] GEMINI_API_KEY not found in environment variables!")
else:
    print("[+] Gemini API key loaded successfully!")

MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-flash")
print(f"[+] Using model: {MODEL_NAME}")


# ===== STATE DEFINITION =====
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_query: str
    report_analysis: str  # Hidden analysis for AI context
    chat_history: List[Dict[str, str]]
    is_report_analyzed: bool
    session_id: str


class OpenAIClient:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_report(self, file_path: str) -> str:
        """Analyze uploaded medical report using OpenAI Vision API."""
        try:
            if not os.path.exists(file_path):
                return "Error: File not found"
            
            # Check file size (20MB limit)
            file_size = os.path.getsize(file_path)
            if file_size > 20 * 1024 * 1024:
                return "Error: File too large (max 20MB)"
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            analysis_prompt = """
            Analyze this medical report thoroughly. Extract all key information including:
            1. Patient demographics and basic info
            2. Test results with normal/abnormal values
            3. Diagnosed conditions or findings
            4. Medications mentioned
            5. Recommendations from doctors
            6. Any concerning findings or red flags
            7. Follow-up instructions
            8. Overall health status assessment
            
            Provide a comprehensive analysis that I can use to answer patient questions.
            Include specific values, ranges, and medical context.
            """
            
            # Handle PDF files
            if file_extension == '.pdf':
                # For PDFs, we'll use a text extraction approach
                # Note: OpenAI's vision API doesn't directly support PDFs
                # You might need to convert PDF to images or use a PDF parser
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text_content = ""
                        for page in pdf_reader.pages:
                            text_content += page.extract_text()
                    
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": f"{analysis_prompt}\n\nMedical Report Text:\n{text_content}"
                            }
                        ],
                        max_tokens=2000
                    )
                    return response.choices[0].message.content
                    
                except ImportError:
                    return "Error: PyPDF2 not installed. Please install it with: pip install PyPDF2"
                except Exception as e:
                    return f"Error processing PDF: {str(e)}"
            
            # Handle image files (jpg, jpeg, png)
            elif file_extension in ['.jpg', '.jpeg', '.png']:
                base64_image = self.encode_image(file_path)
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": analysis_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2000
                )
                return response.choices[0].message.content
            
            else:
                return "Error: Unsupported file format. Please upload PDF, JPG, JPEG, or PNG files."
            
        except Exception as e:
            return f"Error analyzing report: {str(e)}"

    def generate_chat_response(self, query: str, report_analysis: str, chat_history: List[Dict[str, str]]) -> str:
        """Generate simple, easy-to-understand chat response."""
        
        # Build conversation history
        history_messages = []
        for msg in chat_history[-5:]:  # Last 5 messages for context
            history_messages.append({"role": "user", "content": msg['user']})
            history_messages.append({"role": "assistant", "content": msg['assistant']})
        
        system_prompt = f"""
        You are a friendly medical assistant helping someone understand their medical report.
        
        IMPORTANT RULES:
        1. Keep responses under 120 words
        2. Use simple, everyday language - no complex medical terms
        3. If you must use medical terms, immediately explain them in simple words
        4. Be encouraging and supportive
        5. Always remind them to consult their doctor for medical decisions
        
        MEDICAL REPORT ANALYSIS (for your reference only):
        {report_analysis}
        """
        
        messages = [
            {"role": "system", "content": system_prompt}
        ] + history_messages + [
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I'm sorry, I had trouble understanding that. Could you please ask again?"


openai_client = OpenAIClient()

# ===== LANGGRAPH NODES =====
def report_analysis_node(state: ChatState) -> ChatState:
    """Node to analyze the uploaded report (hidden from user)."""
    return state

def chat_response_node(state: ChatState) -> ChatState:
    """Generate chat response based on user query and report analysis."""
    current_query = state["current_query"]
    report_analysis = state["report_analysis"]
    chat_history = state["chat_history"]
    
    response = openai_client.generate_chat_response(
        current_query, 
        report_analysis, 
        chat_history
    )
    
    chat_history.append({
        "user": current_query,
        "assistant": response
    })
    
    state["messages"].append(HumanMessage(content=current_query))
    state["messages"].append(AIMessage(content=response))
    
    return state


def create_chat_graph():
    workflow = StateGraph(ChatState)
    
    workflow.add_node("chat_response", chat_response_node)
    
    workflow.set_entry_point("chat_response")
    workflow.add_edge("chat_response", END)
    
    return workflow.compile()

chat_graph = create_chat_graph()

# ===== SESSION MANAGEMENT =====
active_sessions: Dict[str, ChatState] = {}

def get_or_create_session(session_id: str) -> ChatState:
    if session_id not in active_sessions:
        active_sessions[session_id] = ChatState(
            messages=[],
            current_query="",
            report_analysis="",
            chat_history=[],
            is_report_analyzed=False,
            session_id=session_id
        )
    return active_sessions[session_id]

# ===== GRADIO INTERFACE FUNCTIONS =====
def upload_and_analyze_report(file, session_id):
    """Handle file upload and analysis."""
    if file is None:
        return "Please upload a medical report first.", gr.update(visible=False)
    
    try:
        state = get_or_create_session(session_id)
        
        # In this Gradio version, file_upload has type="filepath", so `file` is a string path
        file_path = file if isinstance(file, str) else getattr(file, "name", None)
        if not file_path:
            return "Error: Could not read the uploaded file path.", gr.update(visible=False)
        
        analysis = openai_client.analyze_report(file_path)
        
        if analysis.startswith("Error"):
            return analysis, gr.update(visible=False)
        
        state["report_analysis"] = analysis
        state["is_report_analyzed"] = True
        state["chat_history"] = []  # reset chat history when a new report is analyzed
        
        success_msg = """✅ Your medical report has been analyzed successfully!

Now you can ask me questions about your report. I'll explain everything in simple terms.

Some questions you might ask:
• What do my test results mean?
• Are there any concerning findings?
• What should I do next?
• Can you explain this in simple terms?"""
        
        # Make chatbot visible and clear previous messages
        return success_msg, gr.update(visible=True, value=[])
        
    except Exception as e:
        return f"Sorry, I couldn't analyze your report. Please try again. Error: {str(e)}", gr.update(visible=False)


def chat_with_bot(message, session_id, history):
    """Handle chat interactions (messages format: list of {'role', 'content'})."""

    # Ensure history is always a list
    if history is None:
        history = []

    # Ignore empty input but still return a valid messages list
    if not message.strip():
        return history, ""

    # Get / create session state
    state = get_or_create_session(session_id)

    # If report not analyzed yet, block questions
    if not state["is_report_analyzed"]:
        response = "Please upload and analyze your medical report first before asking questions."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""

    # Normal flow: call the graph
    state["current_query"] = message

    try:
        result = chat_graph.invoke(state)
        active_sessions[session_id] = result

        # Last assistant reply from your internal chat_history
        response = result["chat_history"][-1]["assistant"]

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        # First output → chatbot history, second → clear textbox
        return history, ""

    except Exception as e:
        error_response = "I'm sorry, I had trouble understanding that. Could you please ask again?"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_response})
        return history, ""




def create_interface():
    with gr.Blocks(title="🏥 Medical Report Chat Assistant") as demo:
        gr.Markdown("""
        # 🏥 Medical Report Chat Assistant
        
        **Simple AI assistant to help you understand your medical reports**
        
        ### How it works:
        1. Upload your medical report (PDF, image)
        2. Click "Analyze Report" 
        3. Chat with me about your report in simple terms!
        """)
        
        with gr.Row():
            session_id = gr.Textbox(
                label="Session ID", 
                placeholder="Enter a unique session ID", 
                value="default",
                scale=1
            )
        
        # Upload Section
        with gr.Group():
            gr.Markdown("### 📄 Step 1: Upload Your Medical Report")
            with gr.Row():
                file_upload = gr.File(
                    label="Choose your medical report",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png"],
                    type="filepath"
                )
                analyze_btn = gr.Button("🔍 Analyze Report", variant="primary")
            
            upload_output = gr.Textbox(
                label="Status", 
                lines=6,
                interactive=False
            )
        
        # Chat Section
        with gr.Group():
            gr.Markdown("### 💬 Step 2: Ask Questions About Your Report")
            
            chatbot = gr.Chatbot(
                label="Chat with Medical Assistant",
                height=400,
                visible=False,
                value=[]  # empty list is fine for messages mode
            )


            
            with gr.Row():
                chat_input = gr.Textbox(
                    label="Ask me about your report",
                    placeholder="What do my test results mean?",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Event handlers
        analyze_btn.click(
            upload_and_analyze_report,
            inputs=[file_upload, session_id],
            outputs=[upload_output, chatbot]
        )
        
        chat_handler = lambda msg, sid, hist: chat_with_bot(msg, sid, hist)
        
        send_btn.click(
            chat_handler,
            inputs=[chat_input, session_id, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        chat_input.submit(
            chat_handler,
            inputs=[chat_input, session_id, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        gr.Markdown("""
        ### 📋 Tips for Better Conversations:
        - Ask specific questions about your results
        - Request explanations in simple terms
        - Ask about next steps or recommendations
        - Feel free to ask follow-up questions
        
        ### ⚠️ Important:
        - This is for information only - always consult your doctor
        - For urgent health concerns, contact your healthcare provider immediately
        - Keep your medical information private and secure
        """)
    
    return demo

# ===== MAIN =====
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",  # Local access only
        server_port=7860,          # Default Gradio port
        share=False                # Set to True if you want a public URL
    )