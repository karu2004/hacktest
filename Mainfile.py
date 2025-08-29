# app.py
# Streamlit + LangChain RAG App (fixed)

import os
import re
import uuid
import time
import random
import tracemalloc
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# LangChain / OpenAI / embeddings / vector store
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.azuresearch import AzureSearch as AzureSearchStore

# Token counting
import tiktoken

# Optional libs referenced by your original code
import plotly.express as px  # noqa: F401  (import kept for parity with original)
import snowflake.connector   # noqa: F401  (import kept for parity with original)

# Your custom logger
from utils import log_bot_interaction

# OpenAI SDK for title generation
from openai import OpenAI

tracemalloc.start()

# --------------------------Loading Environment Variables--------------------------- #
load_dotenv()

INDEX_NAME = "ppp-sh-knowledgebase-test"
CONTENT_FIELD = "content"
VECTOR_FIELD = "content_vector"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

TENANT_ID = "68a6d291-775d-4c44-9e16-65a366eda69d"

SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")

# -------------------------- Streamlit cache helpers --------------------------- #
@st.cache_resource
def load_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

@st.cache_resource
def load_vector_store(_embeddings):
    return AzureSearchStore(
        azure_search_endpoint=AZURE_SEARCH_SERVICE,
        azure_search_key=AZURE_API_KEY,
        index_name=INDEX_NAME,
        embedding_function=_embeddings.embed_query,
        content_field=CONTENT_FIELD,
        vector_field=VECTOR_FIELD,
    )

# Initialize embeddings and vector store
word_embeddings = load_embeddings()
try:
    vector_store = load_vector_store(word_embeddings)
except Exception:
    st.markdown(
        """
        <div style='display: flex; height: 60vh; align-items: center; justify-content: center;'>
          <div style='text-align: center; background-color: #fff3cd; padding: 20px 30px; border: 1px solid #ffeeba; border-radius: 8px; color: #856404; max-width: 600px;'>
            <strong>⚠️ Search backend unavailable</strong><br><br>
            Azure Cognitive Search initialization failed. Verify endpoint, API key, and index settings.<br><br>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Primary LLM for refinement, retrieval QA, and final answer
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# -------------------- Chat Title Helpers ------------------------- #
def generate_chat_title(messages):
    if not messages or len(messages) < 2:
        return f"New Chat {datetime.now().strftime('%I:%M %p')}"
    first_user_content = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            first_user_content = msg.content
            break
    if not first_user_content:
        return f"New Chat {datetime.now().strftime('%I:%M %p')}"
    return first_user_content[:27] + "..." if len(first_user_content) > 30 else first_user_content

def generate_title_with_llm(messages, api_key):
    try:
        client = OpenAI(api_key=api_key)
        user_message = ""
        for msg in messages:
            if isinstance(msg, str):
                user_message = msg
                break
            elif hasattr(msg, "content"):
                user_message = msg.content
                break
            elif isinstance(msg, dict) and msg.get("type") == "human":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return None

        prompt = (
            'Generate a short, descriptive title (max 5 words) for the following user query:\n'
            f'User question: "{user_message}"\n'
            "Return ONLY the title — no quotes or explanations."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.3,
        )
        title = response.choices[0].message.content.strip()
        st.session_state.response_record["title_generation_tokens"] = response.usage.total_tokens
        st.session_state.title_generation_tokens = response.usage.total_tokens
        return title
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

# -------------------- Session helpers --------------------- #
def load_current_chat():
    current_id = st.session_state.current_chat_id
    return st.session_state.chat_sessions[current_id]["messages"]

def save_to_current_chat(item):
    current_id = st.session_state.current_chat_id
    st.session_state.chat_sessions[current_id]["messages"].append(item)

def get_current_chat_messages():
    return st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"]

def switch_chat(chat_id):
    if st.session_state.get("processing_new_question", False):
        st.session_state.processing_new_question = False
    for key in ["current_response_data", "response_chat_id"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.current_chat_id = chat_id

def rename_chat(chat_id, new_title):
    st.session_state.chat_sessions[chat_id]["title"] = new_title

def create_new_chat():
    new_chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_chat_id] = {
        "title": f"New Chat {datetime.now().strftime('%I:%M %p')}",
        "messages": [],
        "created_at": int(time.time()),
    }
    return new_chat_id

def clean_up_unused_chats():
    current_id = st.session_state.get("current_chat_id")
    if (
        current_id in st.session_state.chat_sessions
        and not st.session_state.chat_sessions[current_id].get("messages")
        and not st.session_state.chat_sessions[current_id].get("history")
    ):
        del st.session_state.chat_sessions[current_id]
        print(f"Chat: {current_id} Deleted Successfully !!")

def initialize_session_state():
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chat_sessions:
        initial_chat_id = str(uuid.uuid4())
        st.session_state.chat_sessions[initial_chat_id] = {
            "messages": [],
            "title": f"New Chat {datetime.now().strftime('%I:%M %p')}",
            "created_at": time.time(),
            "bot": "knowledge",
        }
        st.session_state.current_chat_id = initial_chat_id

    # normalize chat entries
    for chat_id, chat in st.session_state.chat_sessions.items():
        chat.setdefault("messages", [])
        chat.setdefault("history", [])
        chat.setdefault("title", f"Chat {chat_id[:6]}")
        chat.setdefault("created_at", time.time())

    # misc state defaults
    defaults = {
        "show_delete_warning": False,
        "chat_to_delete": None,
        "show_rename": False,
        "chat_to_rename": None,
        "processing_new_question": False,
        "current_response_data": False,
        "response_chat_id": False,
        "selected_question": None,
        "user_name": None,
        "user_email": None,
        "defer_rerun_after_title": False,
        "active_bot": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    if "response_record" not in st.session_state:
        st.session_state.response_record = {
            "chatid": st.session_state.current_chat_id,
            "username": st.session_state.user_email,
            "chat_title": None,
            "question_id": None,
            "question": None,
            "refined_question": None,
            "refining_tokens": None,
            "title_generation_tokens": None,
            "answer": None,
            "retrieval_tokens": None,
            "answer_summary": None,
            "answer_summary_tokens": None,
            "is_correct_response": None,
            "comment": None,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

# ----------------------- Retrieval & QA ------------------------ #
def is_greeting_or_acknowledgment(query):
    query_clean = query.strip()
    query_lower = query_clean.lower()
    words = query_clean.split()

    if len(words) > 5:
        return False

    exact_matches = {
        "hi", "hello", "hey", "hi there", "hello there",
        "good morning", "good afternoon", "good evening",
        "good", "thank you", "thanks", "thank you so much", "thanks a lot",
        "ok", "okay", "got it", "sure", "alright", "cool",
    }
    if query_lower in exact_matches:
        return True

    greeting_patterns = [
        r"^(hi|hello|hey)[\s!.]*$",
        r"^good\s+(morning|afternoon|evening)[\s!.]*$",
        r"^(thank|thanks)[\s!.]*$",
        r"^(ok|okay)[\s!.]*$",
    ]
    for pattern in greeting_patterns:
        if re.match(pattern, query_lower):
            return True

    if len(words) <= 4:
        first_word = words[0].lower()
        if first_word in ["hi", "hello", "hey"]:
            return True
        elif len(words) >= 2 and f"{words[0]} {words[1]}".lower() in [
            "good morning", "good afternoon", "good evening"
        ]:
            return True
    return False

def get_greeting_response(query: str) -> str:
    query_lower = query.lower().strip()
    if any(p in query_lower for p in ["who are you", "what are you", "what is this", "what can you do", "tell me about yourself", "are you a bot"]):
        return ("I'm an AI assistant developed by the Bridgehorn Team, "
                "designed to support queries across Snowflake data models, "
                "the Knowledge Management Portal, and hybrid scenarios—delivering smart support where it matters most.")
    if "morning" in query_lower:
        return "Good morning! How can I assist you today?"
    if "afternoon" in query_lower:
        return "Good afternoon! What can I help you with?"
    if "evening" in query_lower:
        return "Good evening! How may I help you?"
    if any(word in query_lower for word in ["hi", "hello", "hey"]):
        return "Hello! How can I help you today?"
    if any(word in query_lower for word in ["thank", "thanks"]):
        return "You're welcome! Is there anything else I can help you with?"
    if any(word in query_lower for word in ["ok", "okay", "got it", "sure", "alright", "cool"]):
        return "Great! What else would you like to know?"
    return "Hello! How can I help you today?"

def count_tokens(text, model="gpt-4o-mini"):
    if not text:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        # Reasonable fallback for modern OpenAI chat models
        try:
            enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def calculate_rag_tokens(user_query, retrieved_documents, llm_response, chat_history_tuples, system_prompt, model):
    embedding_tokens = count_tokens(user_query, model)
    docs_text = "\n".join([doc.page_content for doc in retrieved_documents]) if retrieved_documents else ""
    docs_tokens = count_tokens(docs_text, model)
    history_tokens = 0
    if chat_history_tuples:
        history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history_tuples])
        history_tokens = count_tokens(history_text, model)
    system_tokens = count_tokens(system_prompt, model)
    query_tokens = count_tokens(user_query, model)
    response_tokens = count_tokens(llm_response, model) if llm_response else 0
    total_prompt_tokens = system_tokens + history_tokens + docs_tokens + query_tokens
    total_llm_tokens = total_prompt_tokens + response_tokens
    total_pipeline_tokens = embedding_tokens + total_llm_tokens
    return {
        "embedding_tokens": embedding_tokens,
        "llm_prompt_tokens": total_prompt_tokens,
        "llm_completion_tokens": response_tokens,
        "total_pipeline_tokens": total_pipeline_tokens,
    }

def get_queries_response_retrievalqa(query: str, chat_history=[]):
    # Handle greetings / acknowledgements
    if is_greeting_or_acknowledgment(query):
        greeting_response = get_greeting_response(query)
        st.session_state.response_record["question"] = query
        st.session_state.response_record["refined_question"] = ""
        st.session_state.response_record["refining_tokens"] = 0
        st.session_state.response_record["answer"] = greeting_response
        st.session_state.response_record["answer_summary"] = greeting_response
        st.session_state.response_record["answer_summary_tokens"] = 0
        return greeting_response, "_No sources required_"

    # Build concise history tuples (limit to last 3 Q/A pairs)
    history_tuples = []
    for i in range(0, len(chat_history) - 1, 2):
        q_msg = chat_history[i]
        a_msg = chat_history[i + 1]
        q = q_msg.get("content") if isinstance(q_msg, dict) else getattr(q_msg, "content", None)
        a = a_msg.get("content") if isinstance(a_msg, dict) else getattr(a_msg, "content", None)
        if not q or not a:
            continue
        if "**📂 Sources:**" in a:
            a_clean = a.split("**📂 Sources:**")[0].strip()
        else:
            a_clean = a.strip()
        history_tuples.append((q.strip(), a_clean))
    history_tuples = history_tuples[-6:]  # keep up to 3 pairs (6 messages)

    # Track question id
    question_id = str(uuid.uuid4())
    st.session_state.response_record["question_id"] = question_id

    # Refine vague queries
    refine_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rephrase vague user queries into structured, vector DB-friendly questions using available chat history. "
         "Unless specified PPP should be referred to as Portage Point Partners. "
         "Do NOT add extra information—just clarify using key nouns or entities. "
         "Return an empty string ONLY for: greetings (hi, hello), thanks/acknowledgments (thank you, thanks), "
         "or purely conversational queries with no information need (how are you, nice weather). "
         "Do NOT try to improvise questions which are outside the purview of the Vector DB. "
         "Be cooperative but brief. Do not try to improve already clear questions."
         ),
        ("human", "Chat history:\n{chat_history}\n\nUser's new question:\n{user_question}\n\nRefined question:")
    ])
    refinement_chain = refine_prompt | llm
    chat_history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history_tuples])

    refined_result = refinement_chain.invoke({
        "chat_history": chat_history_str,
        "user_question": query
    })
    refining_tokens = getattr(refined_result, "usage_metadata", {}).get("total_tokens", 0)
    refined_query = refined_result.content.strip()
    print(f"🔍 Refined query: {refined_query}")

    # Session variables
    st.session_state.response_record["question"] = query
    st.session_state.response_record["refined_question"] = refined_query
    st.session_state.response_record["refining_tokens"] = refining_tokens
    st.session_state.refining_tokens = refining_tokens

    if not refined_query:
        msg = "I don't have enough context to answer that question."
        st.session_state.response_record["answer"] = msg
        st.session_state.response_record["answer_summary"] = msg
        st.session_state.response_record["answer_summary_tokens"] = 0
        return msg, "_No sources returned_"

    # Retrieval
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        k=10,
        search_kwargs={"score_threshold": 0.6},
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    retrieval_result = qa_chain.invoke({
        "question": refined_query,
        "chat_history": history_tuples
    })
    source_docs = retrieval_result.get("source_documents", []) or []

    if not source_docs:
        print("⚠️ No relevant documents found for the refined query.")
        msg = "I don't have enough context to answer that question."
        st.session_state.response_record["answer"] = msg
        st.session_state.response_record["answer_summary"] = msg
        st.session_state.response_record["answer_summary_tokens"] = 0
        return msg, "_No sources returned_"

    # Draft answer from ConversationalRetrievalChain
    answer_text = (retrieval_result.get("answer") or "").strip()
    print(answer_text)
    st.session_state.response_record["answer"] = answer_text

    # Build final prompt (policy / redirect behavior preserved)
    DA_REDIRECT_TOPICS = [
        "Data Diagnostics", "Data Cleaning", "Data Engineering", "Data Management", "Data Traceability",
        "Data Models", "Data Modeling", "Data Visualization", "Dashboards", "Power BI Dashboard", "MS Power BI",
        "Data Retrieval", "ELT", "ETL", "Data Monitoring", "Airflow", "Matillion", "Snowflake Data Warehouse",
        "Data Integration", "Data Transformation", "Stored Procedures", "Sprocs", "dbt", "dbt core", "docker",
        "Data Architecture", "Datamart", "Reporting", "Schema", "Database", "Data Source", "API", "Rest API",
        "Data Governance", "Analytics", "Data Analytics", "AI", "AI Agent", "Chatbot", "AI Chatbot", "AI Solution",
        "LLM", "Data Security", "Data Science", "ML", "Machine Learning", "AI/ML", "Data Mining", "Web Scraping",
        "GitHub", "CI/CD"
    ]
    topics_str = ", ".join(DA_REDIRECT_TOPICS)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant who answers strictly based on the retrieved context and chat history. "
         "Use bold text or bullet points only when necessary, and keep your tone professional and friendly. "
         "If asked about who created you (AI Chatbot/Agent) tell it was developed by Data and Analytics Team at PPP. "
         "You can answer any question from the Snowflake Database and the Knowledge Management Portal. "
         f"If the user asks a 'who' or 'which' question related to any of the following topics: {topics_str} — "
         "politely redirect them to the Data & Analytics (DA) team. "
         "Note: For more information on the subject, please reach out to the DA team at **da_ppp@pppllc.com**. "
         "Otherwise, answer only using the provided context and history, and politely decline anything beyond that."
         ),
        ("human", "Chat history:\n{chat_history}\n\nUser's original question:\n{user_question}\n\nRetrieved context:\n{context}\n\n")
    ])
    final_answer_chain = answer_prompt | llm

    # Combine answer + docs for the final prompt context
    doc_texts = "\n\n".join(doc.page_content for doc in source_docs)
    context_text = f"{answer_text}\n\n{doc_texts}"

    # Build Sources HTML + reference list (both)
    references = []
    sources_html = ""
    seen_sources = set()
    for doc in source_docs:
        title = doc.metadata.get("page_title", "Unknown Page")
        src = doc.metadata.get("source", "Unknown Source")
        if src in seen_sources:
            continue
        seen_sources.add(src)
        wrapped_src_display = src.replace("/", "/\u200b")
        sources_html += (
            f"<p style='word-break: break-word;'>"
            f"<strong>{title}</strong>: "
            f"<a href='{src}' target='_blank'>{wrapped_src_display}</a></p>"
        )
        ref = f"- **{title}** | _{src}_"
        if ref not in references:
            references.append(ref)

    # Generate final answer summary
    answer = final_answer_chain.invoke({
        "chat_history": chat_history_str,
        "user_question": query,
        "context": context_text
    })
    summary_tokens = getattr(answer, "usage_metadata", {}).get("total_tokens", 0)
    final_answer = answer.content.strip() if hasattr(answer, "content") else str(answer)

    # Build display payloads
    refs_md = "\n".join(set(references)) if references else "_No sources returned_"
    # Show HTML sources by default
    sources_block = sources_html if sources_html else "<p><em>No sources returned</em></p>"

    st.session_state.response_record["answer_summary"] = final_answer
    st.session_state.response_record["answer_summary_tokens"] = summary_tokens
    st.session_state.summary_tokens = summary_tokens

    # Token accounting after we have the answer
    rag_tokens = calculate_rag_tokens(
        user_query=refined_query,
        retrieved_documents=source_docs,
        llm_response=final_answer,
        chat_history_tuples=history_tuples,
        system_prompt="You are a helpful assistant who answers only based on retrieved context and chat history",
        model="gpt-4o-mini",
    )
    st.session_state.response_record["retrieval_tokens"] = rag_tokens["total_pipeline_tokens"]
    st.session_state.total_retrieval_tokens = rag_tokens["total_pipeline_tokens"]

    return final_answer, sources_block

# ----------------------- Main Streamlit app ------------------------ #
def main_km(user_query):
    initialize_session_state()

    question_id = str(uuid.uuid4())
    st.session_state.question_id = question_id

    response_start = time.time()
    current_chat_id = st.session_state.current_chat_id
    current_chat_messages = get_current_chat_messages()
    current_messages = current_chat_messages.copy()

    if user_query:
        st.session_state.processing_new_question = True
        for key in ["current_response_data", "response_chat_id", "show_comment_box"]:
            if key in st.session_state:
                del st.session_state[key]

        current_messages = get_current_chat_messages()
        current_messages.append(HumanMessage(content=user_query))

        with st.chat_message("AI"):
            response_container = st.container()
            with response_container:
                with st.spinner("Generating Response..."):
                    ai_response, sources = get_queries_response_retrievalqa(user_query, current_messages)
                    print(f"Summary: {ai_response}")
                    print(f"Sources: {sources}")

            with response_container:
                st.markdown(ai_response, unsafe_allow_html=True)
                if sources and sources.strip() and sources.strip() != "_No sources returned_":
                    with st.expander("📂 Sources"):
                        st.markdown(sources, unsafe_allow_html=True)

            st.caption(f"🤖 Answered Using: Knowledge Management Data, ⏱️ Response Time: {time.time() - response_start:.2f} seconds")

        # Save the AI message
        full_response = ai_response
        if sources and sources.strip() and sources.strip() != "_No sources returned_":
            full_response += f"\n\n**📂 Sources:**\n{sources}"

        ai_message = AIMessage(
            content=full_response,
            additional_kwargs={
                "bot": "Knowledge Management Data",
                "response_time": time.time() - response_start,
                "sources": sources if sources and sources.strip() and sources.strip() != "_No sources returned_" else None,
            },
        )
        current_messages.append(ai_message)
        st.session_state.chat_sessions[current_chat_id]["messages"] = current_messages
        st.session_state.processing_new_question = False

        # Title generation if still "New Chat ..."
        current_title = st.session_state.chat_sessions[current_chat_id]["title"]
        if ("New Chat" in current_title and len(current_messages) >= 1 and OPENAI_API_KEY and not hasattr(st.session_state, "title_generation_in_progress")):
            try:
                st.session_state.title_generation_in_progress = True
                first_message = current_messages[0].content.lower()
                greeting_words = ["good morning", "good afternoon", "good evening"]
                if not any(word in first_message for word in greeting_words):
                    print(f"Attempting to generate title for chat {current_chat_id}")
                    message_contents = [msg.content for msg in current_messages]
                    new_title = generate_title_with_llm(message_contents, OPENAI_API_KEY)
                    print(f"Generated title: {new_title}")
                    if new_title and new_title.strip() and not new_title.startswith("New Chat"):
                        new_title = new_title.strip().strip('"').strip("'")
                        st.session_state.chat_sessions[current_chat_id]["title"] = new_title
                        st.session_state.title_updated = True
                    else:
                        fallback_title = generate_chat_title(current_messages)
                        st.session_state.chat_sessions[current_chat_id]["title"] = fallback_title
                        st.session_state.title_updated = True
                        st.session_state.response_record["chat_title"] = fallback_title
                else:
                    # It's a greeting—use fallback
                    fallback_title = generate_chat_title(current_messages)
                    st.session_state.chat_sessions[current_chat_id]["title"] = fallback_title
                    st.session_state.title_updated = True
                    st.session_state.response_record["chat_title"] = fallback_title
            except Exception as e:
                print(f"Error generating title with LLM: {e}")
            finally:
                if "title_generation_in_progress" in st.session_state:
                    del st.session_state.title_generation_in_progress

        response_stop = time.time()
        print(f"Time taken for KM Bot to respond: {response_stop - response_start}")

        # -------------------- BACK-END CONFIGURATION & LOGGING -------------------- #
        routing_tokens = st.session_state.get("routing_tokens", 0)
        title_generation_tokens = st.session_state.get("title_generation_tokens", 0)
        sql_generation_tokens = st.session_state.get("sql_generation_tokens", 0)
        retrieval_tokens = st.session_state.get("total_retrieval_tokens", 0)
        refining_tokens = st.session_state.get("refining_tokens", 0)
        summary_tokens = st.session_state.get("summary_tokens", 0)
        q_id = st.session_state.get("question_id", 0)

        try:
            total_answer_tokens = (
                routing_tokens
                + summary_tokens
                + title_generation_tokens
                + refining_tokens
                + retrieval_tokens
            )
            print(f"Answer Tokens: {total_answer_tokens}")
            log_bot_interaction(
                bot_type="Knowledge Management Bot",
                question=user_query,
                answer=ai_response,
                chat_id=current_chat_id,
                question_id=q_id,
                username=st.session_state.get("user_email"),
                chat_title=st.session_state.chat_sessions[current_chat_id]["title"],
                qc_tokens=routing_tokens,
                sources=sources if sources and sources.strip() and sources.strip() != "_No sources returned_" else None,
                response_time=response_stop - response_start,
                answer_tokens=total_answer_tokens,
                sql_query=None,
                title_generation_tokens=title_generation_tokens,
                is_correct_response=None,
                comment=None,
            )
            print(
                f"Token breakdown - Query Classifier: {routing_tokens}, SQL: {sql_generation_tokens}, "
                f"Summary: {summary_tokens}, Title: {title_generation_tokens}, Refining: {refining_tokens}, "
                f"Retrieval Tokens: {retrieval_tokens}, Total: {total_answer_tokens}"
            )
        except Exception as e:
            print(f"Error logging bot interaction: {e}")

        if st.session_state.get("title_updated", False):
            del st.session_state["title_updated"]
            st.rerun()

# ---------------------------- Streamlit entrypoint ---------------------------- #
st.set_page_config(page_title="KM RAG Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Knowledge Management Assistant")
user_input = st.chat_input("Ask me anything…")
if user_input is not None:
    main_km(user_input)
