import os
import base64
from typing import Dict, List, Optional, TypedDict, Annotated
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-flash")

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_query: str
    report_analysis: str
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
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_report(self, file_path: str) -> str:
        try:
            if not os.path.exists(file_path):
                return "Error: File not found"
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            analysis_prompt = """
            Analyze this medical report thoroughly. Extract and format the information as a JSON-like structure (but as text) with the following headers:
            1. PATIENT INFO: [Name, Age, Gender, etc.]
            2. TEST RESULTS: [List each test, value, and range. Mark clearly as NORMAL or ABNORMAL]
            3. KEY FINDINGS: [Main diagnoses or observations]
            4. MEDICATIONS: [Any prescribed meds]
            5. RECOMMENDATIONS: [Next steps]
            6. SUMMARY: [Brief 2-sentence health status]
            
            Provide specific values. Be precise but also explain what they mean in simple terms.
            """
            
            if file_extension == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text_content = ""
                        for page in pdf_reader.pages:
                            text_content += page.extract_text()
                    
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": f"{analysis_prompt}\n\nMedical Report Text:\n{text_content}"}],
                        max_tokens=2000
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    return f"Error processing PDF: {str(e)}"
            
            elif file_extension in ['.jpg', '.jpeg', '.png']:
                base64_image = self.encode_image(file_path)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    max_tokens=2000
                )
                return response.choices[0].message.content
            else:
                return "Error: Unsupported file format."
        except Exception as e:
            return f"Error analyzing report: {str(e)}"

    def generate_chat_response(self, query: str, report_analysis: str, chat_history: List[Dict[str, str]]) -> str:
        history_messages = []
        for msg in chat_history[-5:]:
            history_messages.append({"role": "user", "content": msg['user']})
            history_messages.append({"role": "assistant", "content": msg['assistant']})
        
        system_prompt = f"""
        You are a friendly medical assistant helping someone understand their medical report.
        Keep responses professional, encouraging, and under 150 words.
        Explain medical terms in simple language. Always remind them to consult their doctor.
        
        REPORT ANALYSIS:
        {report_analysis}
        """
        
        messages = [{"role": "system", "content": system_prompt}] + history_messages + [{"role": "user", "content": query}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}"

# LangGraph Setup
def create_chat_graph():
    def chat_response_node(state: ChatState) -> ChatState:
        client = OpenAIClient()
        response = client.generate_chat_response(
            state["current_query"], 
            state["report_analysis"], 
            state["chat_history"]
        )
        state["chat_history"].append({"user": state["current_query"], "assistant": response})
        state["messages"].append(HumanMessage(content=state["current_query"]))
        state["messages"].append(AIMessage(content=response))
        return state

    workflow = StateGraph(ChatState)
    workflow.add_node("chat_response", chat_response_node)
    workflow.set_entry_point("chat_response")
    workflow.add_edge("chat_response", END)
    return workflow.compile()

chat_graph = create_chat_graph()
