from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from tools import Search_hospital_docs
from models import llm
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


tools=[Search_hospital_docs]
llm_with_tools=llm.bind_tools(tools)



SYSTEM_PROMPT = '''You are CareOps AI, an assistant for hospital administrators at CareOps General Hospital. You help with hospital operations: beds, patient flow, waiting times, staffing, equipment, capacity levels, policies and procedures.

Rules:
    1. For any question about hospital policies, procedures, fees, thresholds or rules, ALWAYS call the Search_hospital_docs tool first. Never answer from your own knowledge.
    2. When calling the tool, send a complete standalone question. Use the chat history to resolve words like "it", "its" and "and for X?" before calling. Example: write "What is the nurse to patient ratio in the emergency department?" and not "And in the emergency department?".
    3. Answer ONLY using the text returned by the tool. Do not guess.
    4. If the tool text does not contain the answer, reply exactly: "This information is not available in the hospital documents." Do not write a source in that case.
    5. Never give medical diagnosis, treatment or medicine advice. If asked, say you only support hospital operations and the user should consult a qualified clinician.
    6. Quote numbers, thresholds and levels exactly as written in the tool text.
    7. End the answer with the source file name(s) you used, in the form: Source: <file name>.
    8. For greetings or small talk, reply briefly without calling the tool.
    9.If the question is not about CareOps General Hospital operations, reply: I can only help with hospital operations questions.
    10. Keep the answer clear and short. Use numbered steps only when the user asks for a procedure or actions.'''



def agent_node(state:AgentState):

    question=[SystemMessage(content=SYSTEM_PROMPT)]+state['messages']

    answer=llm_with_tools.invoke(question)

    return{'messages':[answer]}



builder=StateGraph(AgentState)

builder.add_node('agent', agent_node)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, 'agent')
builder.add_conditional_edges('agent', tools_condition)
builder.add_edge('tools', 'agent')

graph = builder.compile()


if __name__ == "__main__":

    messages=[]

    while True:

        question=input('User: ')

        if question.lower().strip() in ['bye','exit','quit']:
            break

        result=graph.invoke({
            'messages': messages + [('user', question)]
        })

        messages=result['messages']

        print(f'\n\nAI: {messages[-1].content}\n\n')
        
