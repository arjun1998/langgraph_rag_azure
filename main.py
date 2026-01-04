from langgraph.graph import StateGraph, END, START
from nodes.ingest import ingest_node, GraphState
from nodes.retrieve import retrieve_node
from nodes.answer import answer_node

graph = StateGraph(GraphState)
graph.add_node("ingest", ingest_node)
graph.add_node("retrieve", retrieve_node)   
graph.add_node("answer", answer_node)    
graph.add_edge(START, "ingest")
graph.add_edge("ingest", "retrieve")
graph.add_edge("retrieve", "answer")    
graph.add_edge("answer", END)
app=graph.compile()

if __name__ == "__main__":
    result = app.invoke({"query":"Give the details of all Hyundai Creta"})
    print(result)