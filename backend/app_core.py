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
            You are a Professional AI Clinical Assistant specializing in laboratory medicine and diagnostic analysis. 
            Your goal is to transform raw medical report data into a highly intelligent, structured, and clinically useful analysis.

            DO NOT just repeat the text. Use clinical reasoning and your extensive medical knowledge to interpret the findings.

            STRICT OUTPUT STRUCTURE (Markdown):

            # 📋 PROFESSIONAL MEDICAL ANALYSIS REPORT

            ## 1. 👤 Patient Information
            [Extract Name, Age, Gender, and Report Date. If missing, state "Not specified in report"]

            ## 2. 🔍 Test Overview
            [Briefly explain what tests were performed and what they generally measure.]

            ## 3. 📊 Detailed Test Analysis
            | Parameter | Result | Reference Range | Status | Interpretation |
            |-----------|--------|-----------------|--------|----------------|
            [Fill table. Status: Normal / High / Low / Reactive / Non-Reactive. Interpretation: Short clinical meaning.]

            ## 4. 💡 Key Findings
            - [List the most critical or abnormal findings as bullet points.]

            ## 5. 🏥 Clinical Interpretation
            [Provide a deeper explanation of what these results mean for the patient's overall health. Connect related findings (e.g., if both Glucose and HbA1c are high, discuss insulin resistance).]

            ## 6. 🚦 Risk Level
            [RISK_LEVEL: LOW / MODERATE / HIGH]
            [Explain the reasoning behind this risk level.]

            ## 7. ⚠️ Possible Health Implications
            [Discuss potential conditions or health risks associated with these findings, even if data is limited. Use general medical knowledge.]

            ## 8. 💊 Medication Guidance (General Advice)
            - [Provide general info about medications typically used for such results. DO NOT prescribe. Add a strong note: "Generic information only, not a prescription."]

            ## 9. 🍎 Lifestyle & Prevention Advice
            - [Specific diet, exercise, or habit changes to improve these specific metrics.]

            ## 10. 👨‍⚕️ When to Consult a Doctor
            - [Specific "Red Flag" symptoms or thresholds where immediate medical attention is needed.]

            ## 11. 📝 Summary
            [A 3-sentence professional summary of the health status.]

            ## ⚖️ Medical Disclaimer
            [EXTREME IMPORTANCE: State that this is an AI analysis for informational purposes only. It is NOT a medical diagnosis. The user MUST consult a qualified healthcare professional before taking any action.]

            RULES:
            - CLINICAL REASONING: Don't just summarize. Connect dots between different parameters.
            - BEYOND REPORT: Use your medical knowledge to provide implications and lifestyle advice.
            - TONE: Professional, supportive, and clear clinical assistant.
            - NO REPETITION: Avoid copying sentences directly from the source report.
            - LIMITED DATA: If the report is small, still provide general medical insights based on the test type found.
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
        Use Markdown formatting (like bolding, lists) to make your responses easy to read.
        
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
