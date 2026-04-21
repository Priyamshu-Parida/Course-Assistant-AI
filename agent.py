import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.get_or_create_collection(name="course_assistant")

docs = [
    "Retrieval Augmented Generation improves LLM accuracy by retrieving relevant documents.",
    "Overfitting occurs when a model performs well on training data but poorly on new data.",
    "Evaluation metrics include accuracy, precision, recall, and F1-score.",
    "Supervised learning uses labeled data while unsupervised learning does not.",
]

if collection.count() == 0:
    embeddings = embedder.encode(docs).tolist()
    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[f"id_{i}" for i in range(len(docs))]
    )

memory_store = {}

def memory_node(state):
    thread_id = state["thread_id"]
    msgs = memory_store.get(thread_id, [])

    msgs.append({
        "role": "user",
        "content": state["question"]
    })

    msgs = msgs[-6:]
    memory_store[thread_id] = msgs

    return {"messages": msgs}

def retrieval_node(state):
    question = state["question"]

    q_emb = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=2)

    docs = results["documents"][0]

    return {
        "retrieved": docs,
        "sources": docs
    }

def router_node(state):
    q = state["question"].lower()

    if any(x in q for x in ["remember", "my name", "what did i say"]):
        route = "memory"
    elif any(x in q for x in ["calculate", "+", "-", "*", "/"]):
        route = "tool"
    else:
        route = "retrieve"

    return {"route": route}

def tool_node(state):
    question = state["question"]

    try:
        result = eval(question.replace("calculate", ""))
    except:
        result = "Invalid calculation"

    return {"tool_result": str(result)}

def answer_node(state):
    question = state["question"]
    context = "\n".join(state.get("retrieved", []))

    prompt = f"""
You are a Course Assistant AI.

Answer based ONLY on the context below.
If answer is not found, say "Not in knowledge base".

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return {"answer": response.content.strip()}

def ask(question, thread_id="default"):
    state = {
        "question": question,
        "thread_id": thread_id
    }

    mem = memory_node(state)
    state.update(mem)

    route = router_node(state)["route"]

    if route == "memory":
        return {"answer": "Memory feature active."}

    elif route == "tool":
        tool = tool_node(state)
        return {"answer": tool["tool_result"]}

    else:
        ret = retrieval_node(state)
        state.update(ret)

        ans = answer_node(state)
        return {"answer": ans["answer"]}

if __name__ == "__main__":
    print("\n🎓 Course Assistant AI (type 'exit' to quit)\n")

    while True:
        q = input("You: ")

        if q.lower() == "exit":
            break

        res = ask(q, thread_id="user1")
        print("AI:", res["answer"])
