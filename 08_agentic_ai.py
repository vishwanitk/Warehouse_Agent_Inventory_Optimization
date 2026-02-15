#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()


# In[135]:


import numpy as np
import pandas as pd

np.random.seed(42)

skus = [f"SKU_{i}" for i in range(100)]
locations = [f"Store_{i}" for i in range(10)]

data = []

for sku in skus:
    demand_type = np.random.choice(["fast","medium","slow"], p=[0.2,0.5,0.3])
    
    if demand_type == "fast":
        base_demand = np.random.randint(20,40)
    elif demand_type == "medium":
        base_demand = np.random.randint(8,20)
    else:
        base_demand = np.random.randint(1,8)
        
    for loc in locations:
        daily_sales = max(1, int(np.random.normal(base_demand, 3)))
        stock = np.random.randint(0, daily_sales*60)
        
        data.append([sku, loc, stock, daily_sales])

df = pd.DataFrame(data, columns=["SKU","Location","Stock","Avg_Daily_Sales"])


# In[136]:


df


# In[137]:


warehouse_stock = np.random.randint(50, 1000)


# In[138]:


df["Location"].value_counts()


# In[139]:


df[df["Location"]=="Store_9"]


# In[140]:


from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from langgraph.constants import Send

DOI_LIMIT = 30

class State(TypedDict, total=False):
    sku: str
    warehouse_stock: int
    warehouse_daily_sales: float

    stores: Dict[str, Dict]

    warehouse_doi: float
    excess_qty: int

    eligible_stores: List[str]
    transfer_plan: Dict[str, int]

    final_output: Dict


# In[141]:


def load_store_data(state: State):
    sku_df = df[df["SKU"] == state["sku"]]

    stores = {}

    for _, row in sku_df.iterrows():
        stores[row["Location"]] = {
            "stock": row["Stock"],
            "daily_sales": row["Avg_Daily_Sales"]
        }

    return {"stores": stores}


# In[142]:


from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

from langchain_groq import ChatGroq

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)



def warehouse_reasoning(state: dict):
    analytics_output = state.get("transfer_plan", {})
    sku = state["sku"]

    prompt = f"""
    You are a Central Warehouse Optimization Agent.

    SKU: {sku}

    Transfer Plan:
    {analytics_output}

    Provide:
    1. Priority ranking of stores
    2. Risk analysis
    3. Strategic recommendation
    4. Whether warehouse stock is sufficient
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"llm_recommendation": response.content}


# In[143]:


def orchestrator(state: State):
    warehouse_doi = state["warehouse_stock"] / state["warehouse_daily_sales"]

    excess_qty = 0
    if warehouse_doi > DOI_LIMIT:
        target_stock = DOI_LIMIT * state["warehouse_daily_sales"]
        excess_qty = state["warehouse_stock"] - target_stock

    return {
        "warehouse_doi": warehouse_doi,
        "excess_qty": int(excess_qty)
    }


# In[144]:


def route_if_excess(state: State):
    if state["excess_qty"] > 0:
        return "identify_stores"
    return END


# In[145]:


def identify_stores(state: State):
    eligible = []

    for store, data in state["stores"].items():
        doi = data["stock"] / data["daily_sales"]
        if doi < DOI_LIMIT:
            eligible.append(store)

    return {"eligible_stores": eligible}


# In[146]:


from langgraph.constants import Send

def assign_workers(state: dict):
    # Create Send objects for each store
    sends = [
        Send("calculate_transfer", {"store": store})
        for store in state["eligible_stores"]
    ]
    
    # Wrap the list in a dict under "__send__"
    return {"__send__": sends}


# In[147]:


def calculate_transfer(state: dict):
    sku = state["sku"]
    warehouse_stock = state["warehouse_stock"]

    # Filter dataframe for SKU
    sku_df = df[df["SKU"] == sku]

    transfers = []
    total_required = 0

    for _, row in sku_df.iterrows():
        store = row["Location"]
        stock = row["Stock"]
        daily_sales = row["Avg_Daily_Sales"]

        doi = stock / daily_sales if daily_sales > 0 else 0
        required_stock = DOI_LIMIT * daily_sales
        needed = max(required_stock - stock, 0)

        if needed > 0:
            transfers.append({
                "store": store,
                "required_units": round(needed)
            })
            total_required += needed

    # Adjust if warehouse insufficient
    if total_required > warehouse_stock:
        scale = warehouse_stock / total_required
        for t in transfers:
            t["required_units"] = round(t["required_units"] * scale)

    return {
        "transfers": transfers,
        "warehouse_remaining": warehouse_stock - sum(
            t["required_units"] for t in transfers
        )
    }


# In[148]:


def aggregate(state: State):
    # Merge outputs from all workers
    transfer_plan = {}
    warehouse_stock = state["warehouse_stock"]

    # 'state' contains keys like 'worker_results' or '__merged__'
    worker_outputs = state.get("__merged__", [])
    for w in worker_outputs:
        transfer_plan.update(w.get("transfer_plan", {}))

    total_transfer = sum(transfer_plan.values())
    warehouse_after = warehouse_stock - total_transfer

    return {
        "final_output": {
            "sku": state["sku"],
            "warehouse_before": warehouse_stock,
            "warehouse_after": warehouse_after,
            "transfers": transfer_plan
        },
        "transfer_plan": transfer_plan  # make sure LLM can access it
    }


# In[149]:


from IPython.display import Image,display
from langgraph.graph import START,StateGraph,END


# In[150]:
# At the bottom of agentic_ai.py

def build_graph():
    builder = StateGraph(State)

    # REGISTER NODES
    builder.add_node("load_store_data", load_store_data)
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("identify_stores", identify_stores)
    builder.add_node("assign_workers", assign_workers)
    builder.add_node("calculate_transfer", calculate_transfer)
    builder.add_node("aggregate", aggregate)
    builder.add_node("llm_agent", warehouse_reasoning)

    builder.set_entry_point("load_store_data")

    builder.add_edge("load_store_data", "orchestrator")

    # Conditional Routing
    builder.add_conditional_edges(
    "orchestrator",
    route_if_excess,
    {"identify_stores": "identify_stores", END: END}
    )
    builder.add_edge("identify_stores", "assign_workers")
    # assign_workers → calculate_transfer is handled by Send()


    builder.add_edge("assign_workers", "calculate_transfer")
    builder.add_edge("calculate_transfer", "aggregate")
    builder.add_edge("aggregate", "llm_agent")
    builder.add_edge("llm_agent", END)


    return builder.compile()


# In[159]:


#graph_image=graph.get_graph().draw_mermaid_png()
#display(Image(graph_image))


# In[152]:




# In[114]:




# In[ ]:





# In[153]:


df


# In[154]:


df["DOI"]=df["Stock"]/df["Avg_Daily_Sales"]


# In[158]:


df[df["SKU"]=="SKU_1"]


# In[ ]:




