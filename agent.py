from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.documents import Document
from deep_translator import GoogleTranslator
from config import vector_store
import gradio as gr

PREMIUM = False

SYSTEM_PROMPT = """Your name is UC Atlas. You are an official virtual assistant for the University of Cebu (UC). \
You help students, faculty, applicants, and visitors find accurate information about the university.

## Tool Usage
- ALWAYS call retrieve_context before answering any UC-related question, no exceptions.
- Pass the user's question exactly as-is to retrieve_context.
- After retrieving, carefully read ALL content returned and use it to form your answer.
- Only say you don't have information if the retrieved content is truly empty or irrelevant.

## After Calling retrieve_context
- Read the full content returned by the tool carefully.
- The content contains the answer — extract and present it clearly.
- Do NOT say you lack information if the tool returned any content at all.
- If the context lacks sufficient information or has no information, respond with:
  "I'm sorry, I don't have that information available. Please contact the University of Cebu directly for assistance."

## Scope
You can ONLY assist with UC-related topics:
- Campus locations (Main, Banilad, Lapu-Lapu Mactan, Mandaue, and Naga)
- Courses, programs, and colleges
- Admission requirements and enrollment procedures
- Tuition fees and scholarships
- Offices, facilities, and contact information
- Student organizations and extracurricular activities

For off-topic questions, politely decline and remind the user of your purpose. Do not answer them.

## Response Format
- Be concise. ONLY include what is necessary to answer the question.
- Use bullet points or numbered lists only when presenting multiple items.
- If information differs per campus, clearly distinguish between them.
- Cite the source URL from the retrieved context at the end of your response.
- Never repeat the raw metadata or chunk structure in your response.

## Tone
- Professional, friendly, and patient.
- Use plain language, especially for applicants unfamiliar with the university.
"""

def normalize_query(query: str) -> str:
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(query)
        return translated
    except Exception:
        return query

@tool
def retrieve_context(query: str):
    """
    This is a tool that would help retrieve information related to University of Cebu to help answer a query.
    query argument accepts a literal string, not a dictionary or object. Just a string.
    """
    translated_query = normalize_query(query)
    top_docs = vector_store.max_marginal_relevance_search(translated_query, k=2)

    if not top_docs:
        return "No relevant information."
    
    all_docs = []
    seen_seq_num = set()
    for top_doc in top_docs:
        seq_num = top_doc.metadata.get("seq_num")
        if not seq_num or seq_num in seen_seq_num:
            continue
        seen_seq_num.add(seq_num)

        results = vector_store.get(where={ "seq_num": int(seq_num) })
        sibling_chunks = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        sorted_siblings = sorted(sibling_chunks, key=lambda d: d.metadata.get("start_index", -1))
        all_docs.extend(sorted_siblings)

    serialized = "\n\n".join(
        f"# Document {i + 1}\n"
        f"Source: {doc.metadata.get("url")}\n"
        f"Content:\n {doc.page_content}"
        for i, doc in enumerate(all_docs)
    )

    output = f"Retrieved {len(all_docs)} relevant documents:\n\n{serialized}"  
    return output 

def create_app(agent):
    def generate_tokens(query, history):
        response = ""
        for chunk in agent.stream(
            { "messages" : [{ "role" : "user", "content" : query }]},
            { "configurable" : { "thread_id" : "1" }},
            stream_mode="messages",
            version="v2"
        ):
            msg, _ = chunk["data"]
            msg.pretty_print()
            if isinstance(msg, AIMessageChunk) and msg.content:
                response += str(msg.content)
                yield response

    return gr.ChatInterface(fn=generate_tokens)
            
def main():
    model = (
        ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        if PREMIUM else
        ChatOllama(model="llama3.2:latest", temperature=0, num_ctx=8192)
    )

    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )

    print("Model being used:", model.get_name())
    app = create_app(agent)
    app.launch()

if __name__ == "__main__":
    main()
