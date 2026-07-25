# AI-Project-RAG-Agent-Assist
# AI Agent Assist for Contact Centers

I built this project to test how AI and retrieval systems can support contact center agents during live customer calls. The main goal is to give agents instant, policy-compliant guidance, scripts, and next steps to help resolve queries faster and lower Average Handling Time (AHT).

---

## 🛠️ How It Works

Instead of making agents search through endless knowledge articles while on a call, this tool listens to the query (or scenario) and pulls back exact policy answers automatically.

Here’s what I built into the system:

* **Hybrid Search & Re-Ranking:** It searches policy documents using both keyword search (BM25) and vector embeddings (ChromaDB), then uses FlashRank to re-rank the results so only the most accurate context reaches the model.
* **Smart DPA State Logic:** One common problem with AI call assistants is that they keep telling the agent to verify the account details over and over. I added a simple DPA toggle and strict system rules so that if the customer is already verified, the AI skips security checks and gives the resolution steps immediately.
* **Structured Agent UI:** Built using Streamlit to give frontline agents a clear layout:
  * Emergency level alert banner
  * Ready-to-use call script that wraps cleanly on screen
  * Interactive checklist of CRM steps to follow
  * Policy limits (like goodwill caps or maximum claim allowances)

---

## 💻 Tech Used

* **UI:** Streamlit
* **LLM:** Groq (Llama 3 70B)
* **Framework:** LangChain
* **Vector Database:** ChromaDB
* **Search & Re-Ranking:** BM25 + FlashRank

---

## 🚀 How to Run It

1. **Clone this repo:**
   ```bash
   git clone [https://github.com/kavanmorgan-maker/AI-Project-Enterprise-Knowledge-Article-Sync-RAG-Agent-Assist.git](https://github.com/kavanmorgan-maker/AI-Project-Enterprise-Knowledge-Article-Sync-RAG-Agent-Assist.git)
   cd AI-Project-Enterprise-Knowledge-Article-Sync-RAG-Agent-Assist
