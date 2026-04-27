from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from datetime import datetime
from bson.objectid import ObjectId
import logging

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "medanalyzer_db")

# In-memory storage for fallback
MOCK_DB = {
    "users": [],
    "sessions": {},
    "messages": []
}

class Database:
    def __init__(self):
        self.use_mock = False
        try:
            self.client = AsyncIOMotorClient(MONGODB_URI, tlsAllowInvalidCertificates=True, connectTimeoutMS=2000, serverSelectionTimeoutMS=2000)
            self.db = self.client[DATABASE_NAME]
            # Collections
            self.sessions_collection = self.db["sessions"]
            self.messages_collection = self.db["messages"]
            self.users_collection = self.db["users"]
        except Exception as e:
            print(f"MongoDB connection failed, using in-memory mock: {e}")
            self.use_mock = True

    async def check_connection(self):
        if self.use_mock: return False
        try:
            await self.client.admin.command('ping')
            return True
        except Exception:
            self.use_mock = True
            return False

db_wrapper = Database()

async def get_user_by_email(email: str):
    if db_wrapper.use_mock or not await db_wrapper.check_connection():
        return next((u for u in MOCK_DB["users"] if u["email"] == email), None)
    return await db_wrapper.users_collection.find_one({"email": email})

async def create_user(user_data: dict):
    user_data["created_at"] = datetime.utcnow()
    if db_wrapper.use_mock or not await db_wrapper.check_connection():
        user_data["_id"] = str(ObjectId())
        MOCK_DB["users"].append(user_data)
        return user_data["_id"]
    result = await db_wrapper.users_collection.insert_one(user_data)
    return str(result.inserted_id)

async def get_session_data(session_id: str, user_id: str = None):
    if db_wrapper.use_mock or not await db_wrapper.check_connection():
        session = MOCK_DB["sessions"].get(session_id)
        if not session:
            return {
                "messages": [], "current_query": "", "report_analysis": "",
                "chat_history": [], "is_report_analyzed": False,
                "session_id": session_id, "user_id": user_id
            }
        # In mock, we don't store messages separately for simplicity in this fallback
        return session

    query = {"session_id": session_id}
    if user_id: query["user_id"] = user_id
        
    session = await db_wrapper.sessions_collection.find_one(query)
    if not session:
        return {
            "messages": [], "current_query": "", "report_analysis": "",
            "chat_history": [], "is_report_analyzed": False,
            "session_id": session_id, "user_id": user_id
        }
    
    db_messages = await get_chat_history_from_db(session_id)
    lc_messages = []
    chat_history = []
    
    for msg in db_messages:
        content = msg["content"]
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=content))
            if lc_messages and isinstance(lc_messages[-2], HumanMessage):
                chat_history.append({"user": lc_messages[-2].content, "assistant": content})

    return {
        "messages": lc_messages,
        "current_query": "",
        "report_analysis": session.get("report_analysis", ""),
        "chat_history": chat_history,
        "is_report_analyzed": session.get("is_report_analyzed", False),
        "session_id": session_id,
        "user_id": user_id
    }

async def save_session_data(session_id: str, state: dict, user_id: str = None):
    update_data = {
        "session_id": session_id,
        "report_analysis": state.get("report_analysis", ""),
        "is_report_analyzed": state.get("is_report_analyzed", False),
        "updated_at": datetime.utcnow()
    }
    if user_id: update_data["user_id"] = user_id
        
    if db_wrapper.use_mock or not await db_wrapper.check_connection():
        MOCK_DB["sessions"][session_id] = state
        return

    await db_wrapper.sessions_collection.update_one(
        {"session_id": session_id},
        {"$set": update_data},
        upsert=True
    )

async def add_message_to_db(session_id: str, role: str, content: str, user_id: str = None):
    doc = {
        "session_id": session_id, "role": role, "content": content, "timestamp": datetime.utcnow()
    }
    if user_id: doc["user_id"] = user_id
    
    if db_wrapper.use_mock or not await db_wrapper.check_connection():
        MOCK_DB["messages"].append(doc)
        return

    await db_wrapper.messages_collection.insert_one(doc)

async def get_chat_history_from_db(session_id: str):
    if db_wrapper.use_mock or not await db_wrapper.check_connection():
        return [m for m in MOCK_DB["messages"] if m["session_id"] == session_id]
        
    cursor = db_wrapper.messages_collection.find({"session_id": session_id}).sort("timestamp", 1)
    return await cursor.to_list(length=100)

async def get_all_sessions(user_id: str = None):
    if db_wrapper.use_mock or not await db_wrapper.check_connection():
        sessions = [s for s in MOCK_DB["sessions"].values() if s.get("is_report_analyzed") and (not user_id or s.get("user_id") == user_id)]
        results = []
        for s in sessions:
            analysis = s.get("report_analysis", "")
            snippet = analysis[:200].replace("\n", " ").strip() + ("..." if len(analysis) > 200 else "")
            results.append({
                "session_id": s.get("session_id", ""),
                "snippet": snippet,
                "updated_at": datetime.utcnow().isoformat()
            })
        return results

    query = {"is_report_analyzed": True}
    if user_id: query["user_id"] = user_id
        
    cursor = db_wrapper.sessions_collection.find(
        query, {"session_id": 1, "report_analysis": 1, "updated_at": 1, "_id": 0}
    ).sort("updated_at", -1)
    sessions = await cursor.to_list(length=50)
    results = []
    for s in sessions:
        analysis = s.get("report_analysis", "")
        snippet = analysis[:200].replace("\n", " ").strip() + ("..." if len(analysis) > 200 else "")
        results.append({
            "session_id": s.get("session_id", ""),
            "snippet": snippet,
            "updated_at": s.get("updated_at", "").isoformat() if s.get("updated_at") else ""
        })
    return results
