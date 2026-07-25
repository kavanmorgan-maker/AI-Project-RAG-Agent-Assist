import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="OVO Agent Assist & Policy Engine", layout="wide")

# Custom CSS for input field sizing
st.markdown("""
    <style>
    div[data-baseweb="textarea"] {
        width: 100% !important;
    }
    textarea {
        font-size: 14px !important;
        line-height: 1.4 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎧 Enterprise CX Agent Assist Portal")


CHROMA_DIR = "./chroma_db"

@st.cache_resource
def setup_advanced_retriever():
    # 1. Load local vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    # Extract docs for BM25 sparse index
    raw_docs = vectorstore.get()
    from langchain_core.documents import Document
    docs = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(raw_docs["documents"], raw_docs["metadatas"])
    ]
    
    # 2. Base Retrievers (Fetch top 6 candidate chunks)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 6
    
    # 3. Hybrid Ensemble (Reciprocal Rank Fusion)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    # 4. Local FlashRank Re-Ranker (Selects top 3 most relevant chunks)
    compressor = FlashrankRerank(top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever

try:
    retriever = setup_advanced_retriever()
except Exception as e:
    st.error(f"Error initializing knowledge base retriever: {e}")
    st.stop()

# Layout: Two Columns
col_chat, col_assist = st.columns([1, 1])

with col_chat:
    st.subheader("💬 Live Customer Interaction")
    
    scenario = st.selectbox(
        "Select Test Scenario:",
        [
            "Custom Input",
            "Emergency Boiler Repair / Coverage Limit",
            "Medical Outage / Vulnerable Customer",
            "Failed DPA / Verification Issue",
            "Formal Escalation / Ombudsman Deadlock"
        ]
    )
    
    default_text = ""
    if scenario == "Emergency Boiler Repair / Coverage Limit":
        default_text = "My boiler broke down. How much is covered under my policy and what is the maximum claim cap?"
    elif scenario == "Medical Outage / Vulnerable Customer":
        default_text = "My electricity went off, and I have an oxygen machine running at home. What can you do right now?"
    elif scenario == "Failed DPA / Verification Issue":
        default_text = "I want to change my bank details, but I don't know my account number or my password."
    elif scenario == "Formal Escalation / Ombudsman Deadlock":
        default_text = "I've been waiting 2 months for a billing resolution. I want a deadlock letter for the Energy Ombudsman!"

    customer_query = st.text_area("Customer Input / Live Transcript:", value=default_text, height=68)
    
    # DPA Verification Toggle State
    dpa_verified = st.checkbox("✅ Customer Identity Verified (DPA Passed)", value=True)
    
    search_btn = st.button("Fetch Agent Guidance ⚡", type="primary")

with col_assist:
    st.subheader("💡 Real-Time Action & Policy Guidance")

    if search_btn and customer_query:
        with st.spinner("Searching KB & Re-Ranking context..."):
            results = retriever.invoke(customer_query)

        with st.expander("🔍 Re-Ranked Policy Context (FlashRank Top 3)", expanded=False):
            for i, doc in enumerate(results, 1):
                source_file = doc.metadata.get("source", "Unknown")
                st.markdown(f"**Rank {i} | Source:** `{os.path.basename(source_file)}`")
                st.info(doc.page_content)

        context_text = "\n\n".join([doc.page_content for doc in results])

        prompt_template = ChatPromptTemplate.from_template("""
        You are an AI Agent Assistant for OVO Energy contact center.
        Analyze the context and query below, then return a valid JSON object ONLY (no markdown code blocks or wrapper text around the JSON).

        Customer DPA Verification Status: {dpa_status}

        Context:
        {context}

        Customer Query:
        {query}

        STRICT SYSTEM RULES (FORBIDDEN BEHAVIORS):
        1. IF Customer DPA Status is "VERIFIED", YOU ARE STRICTLY FORBIDDEN FROM INSTRUCTING THE AGENT TO ASK FOR ACCOUNT NUMBERS, PASSWORDS, POSTCODES, OR ANY VERIFICATION DETAILS.
        2. Do NOT say "Can I take your account details to look into this?". Assume the agent already has the customer's account open on their screen right now.
        3. Jump DIRECTLY into giving actionable advice, troubleshooting steps, policy limits, or goodwill authorizations.

        Provide JSON matching this exact key structure:
        {{
            "urgency": "CRITICAL" | "HIGH" | "STANDARD",
            "summary": "Short 1-sentence issue summary",
            "recommended_script": "Direct, empathetic response answering the customer's query directly without asking for verification details",
            "action_checklist": [
                "Step 1 action",
                "Step 2 action"
            ],
            "auth_limit": "Financial or policy limit (e.g. £2,000 cap, £30 goodwill, or N/A)"
        }}
        """)

        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
        chain = prompt_template | llm

        dpa_status_str = "VERIFIED (Do NOT ask for verification)" if dpa_verified else "UNVERIFIED (Verification required)"

        with st.spinner("Generating structured agent workflows..."):
            raw_response = chain.invoke({
                "context": context_text, 
                "query": customer_query,
                "dpa_status": dpa_status_str
            }).content
            
            # Sanitize possible markdown formatting around JSON
            clean_json = raw_response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)

        # FRONTEND UI COMPONENTS
        
        # 1. Urgency Alert Banner
        urgency = data.get("urgency", "STANDARD")
        if urgency == "CRITICAL":
            st.error(f"🚨 **PRIORITY LEVEL: CRITICAL EMERGENCY** — {data.get('summary')}")
        elif urgency == "HIGH":
            st.warning(f"⚠️ **PRIORITY LEVEL: HIGH RISK** — {data.get('summary')}")
        else:
            st.success(f"🟢 **PRIORITY LEVEL: STANDARD ENQUIRY** — {data.get('summary')}")

        st.divider()

        # 2. Recommended Call Script Card (Auto-wrapping text callout)
        st.markdown("### 🗣️ Call Script (Copy to Customer Chat)")
        st.info(f"💬 {data.get('recommended_script')}")

        # 3. Interactive CRM Step Checklist
        st.markdown("### ☑️ Mandatory CRM Action Steps")
        for idx, step in enumerate(data.get("action_checklist", [])):
            st.checkbox(step, key=f"step_{idx}")

        # 4. Authorized Policy Limit Callout
        st.markdown(f"**💰 Authorized Policy Limit:** `{data.get('auth_limit', 'N/A')}`")