from langchain_community.vectorstores import FAISS
from models import model,llm
from langgraph.graph import StateGraph, START, END
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
import operator
load_dotenv()


#   State:
class ChatState(TypedDict):
    question:str
    history: Annotated[List[str],operator.add]
    answer:str



#   Question Node:
def question_node(state:ChatState):
    question=input('Human: ')
    return {'question':question}



#   Retriever:
store = FAISS.load_local(
    "Vector_Store",
    embeddings=model,
    allow_dangerous_deserialization=True
)

retriever = store.as_retriever(search_kwargs={"k": 8})



#   RAG_Node:
def RAG_node(state:ChatState):

    #   Fetch the question from the state:
    question=state['question']

    #   Load the history from state, keep only the last 6 entries:
    history = "\n".join(state['history'][-6:])

    #   Make the question standalone (only if history exists):
    search_question = question
    if history != "":
        rewrite_prompt = f'''
        Given the chat history and the latest user question, 
        rewrite the latest question as a complete standalone question that can
        be understood without the history. 
        Resolve words like "it", "its", "that" and phrases like "and for X?" using the history. 
        If the question is already complete, return it unchanged. 
        Output ONLY the rewritten question and nothing else.

        Chat history:
        {history}

        Latest question: {question}

        Standalone question:
        '''
        search_question = llm.invoke(rewrite_prompt).text.strip()
    print('STANDALONE:', search_question)

    #   Fetch the useful docs using retriever:
    docs = retriever.invoke(search_question)

    #   Convert docs into one string (with source file name):
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

    #   Prompt:
    prompt=f'''
        You are CareOps AI, an assistant for hospital administrators at CareOps General Hospital. You help with hospital operations: beds, patient flow, waiting times, staffing, equipment, capacity levels, policies and procedures.

        Rules:
        1. Answer ONLY using the context given below. Do not use outside knowledge and do not guess.
        2. If the context does not contain the answer, reply exactly: "This information is not available in the hospital documents."
        3. Never give medical diagnosis, treatment or medicine advice. If asked, say you only support hospital operations and the user should consult a qualified clinician.
        4. When the context has numbers, thresholds or levels, quote them exactly as written.
        5. Mention the source document name(s) you used at the end of the answer, in the form: Source: <file name>.
        6. Use the chat history only to understand follow-up questions. Facts must come from the context, not from the history.
        7. Keep the answer clear and short. Use numbered steps only when the user asks for a procedure or actions.

        Chat history:
        {history}

        Context:
        {context}

        Question: {search_question}

        Answer:
    '''

    #   Finding the answer:
    answer=llm.invoke(prompt).text

    print(answer)

    #   Return the answer and history:
    return {
        'history':[f'Human_message:{question}',f'AI_Message:{answer}'],

        'answer':answer
    }



#   Check Node:
def check_node(state:ChatState):
    question=state['question']

    if question.lower().strip() in ['bye','exit','quit'] :
        return 'exit'
    return 'continue'



#   Graph:
builder=StateGraph(ChatState)

builder.add_node('ask_question',question_node)
builder.add_node('RAG',RAG_node)

builder.add_edge(START,'ask_question')
builder.add_conditional_edges('ask_question',check_node,
                              {'exit': END,
                               'continue':'RAG'
                               })
builder.add_edge('RAG','ask_question')

graph=builder.compile()



#   Invoke The Graph:
graph.invoke({
    'question':'',
    'history':[],
    'answer':''
})
