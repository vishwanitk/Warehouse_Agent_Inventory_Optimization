#!/usr/bin/env python
# coding: utf-8

# In[236]:


import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.types import Send
from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Dict, List


# In[237]:


load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)


# In[238]:


llm.invoke("Hi Vishwa here")


# In[239]:


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


# In[240]:


## Vendor Assignment
np.random.seed(42)
vendors = [f"VEN_{i:02d}" for i in range(1, 11)]


sku_list=df["SKU"].unique()
np.random.shuffle(sku_list)

vendor_assignments=[]

for i,sku in enumerate(sku_list):
    vendor_assignments.append({
        "SKU":sku,
        "Vendor_ID":vendors[i%len(vendors)]
    })
    vendor_df=pd.DataFrame(vendor_assignments)

df=df.merge(vendor_df,on="SKU",how="left")


# In[241]:


## Vendor return policy
vendor_policies = {}
np.random.seed(42)

return_conditions=[
    "DO_NOT_ACCEPT",
    "ACCEPT_POST_EXPIRY",
    "ACCEPT_3_MONTHS_BEFORE_EXPIRY"
]

for vendor in vendors:
    vendor_policies[vendor]={
        "return_policy_type":np.random.choice(return_conditions)

    }


# In[242]:


## Vector DB for Vendor policies

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma


# In[243]:


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents=[]

for vendor,policy in vendor_policies.items():
    policy_text=f""" 
    Vendor:{vendor}
    Return_Policy:{policy["return_policy_type"]}
    """ 

    documents.append(Document(page_content=policy_text.strip(),
                              metadata={"vendor_id":vendor}))
    



vectorstore=Chroma.from_documents(documents=documents,embedding=embedding_model,collection_name="vendor_policies",persist_directory="./chroma_vendor_db")

print(f"Vector DB Created Successfully with {len(documents)} documents.")
    


# In[244]:


## Create Retrival  tool

def get_vendor_policy(vendor_id: str):
    """
    Retrieve vendor policy deterministically using metadata.
    """
    docs = vectorstore._collection.get(include=["metadatas", "documents"])
    for doc_text, metadata in zip(docs["documents"], docs["metadatas"]):
        if metadata.get("vendor_id") == vendor_id:
            return doc_text
    return "Vendor policy not found"


## Add tool node
def vendor_policy_tool_node(state:dict):
    sku=state["sku"]

    vendor_id=df[df["SKU"]==sku]["Vendor_ID"].iloc[0]

    policy=get_vendor_policy(vendor_id)

    return {
        "vendor_policy":policy
    }


# In[245]:


## load the data

def load_store_data(state:dict):
    sku_df=df[df["SKU"]==state["sku"]]

    stores={}

    for _,row in sku_df.iterrows():
        stores[row["Location"]]={
            "stock":row["Stock"],
            "daily_sales":row["Avg_Daily_Sales"]
        }
    return{"stores":stores}


# In[246]:


class State(TypedDict,total=False):
    sku:str
    warehouse_stock:int
    warehouse_daily_sales:float

    stores:Dict[str,Dict]

    warehouse_doi:float
    excess_qty: int
    shortage_qty: int

    eligible_stores:List[str]
    transfer_plan:Dict[str,int]
    final_output: Dict
    llm_output:str
    llm_recommendation:str
    vendor_policy:str
    total_transfer_qty:int


# In[247]:


DOI_LIMIT_HIGH=30
DOI_LIMIT_LOW=30
DOI_LIMIT_AVG=15


# In[248]:


## Orchestrator
def orchestrator(state:dict):
    warehouse_doi=state["warehouse_stock"]/state["warehouse_daily_sales"]

    excess_qty=0
    shortage_qty=0

    max_stock=DOI_LIMIT_HIGH*state["warehouse_daily_sales"]

    if warehouse_doi>DOI_LIMIT_HIGH:
        excess_qty=state["warehouse_stock"]-max_stock
    
    elif warehouse_doi<DOI_LIMIT_LOW:
        shortage_qty=max_stock-state["warehouse_stock"]

    return {
        "warehouse_doi":warehouse_doi,
        "excess_qty":excess_qty,
        "shortage_qty":shortage_qty

    }


# In[249]:


def route(state:dict):
    if state["excess_qty"]>0:
        return "identify_stores_push"
    elif state["shortage_qty"]>0:
        return "identify_stores_pull"

    return END


# In[250]:


## Push logic
def assign_workers_push(state:dict):
    sends=[
        Send("calculate_transfer_push",{"store":store})
        for store in state["eligible_stores"]
    ]
    return{"__send__":sends}


# In[251]:


def assign_workers_pull(state: dict):
    sends = [
        Send("calculate_transfer_pull", {"store": store})
        for store in state["eligible_stores"]
    ]
    return {"__send__": sends}


# In[252]:


def calculate_transfer_push(state:dict):

    transfer_plan = {}
    excess_qty = state["excess_qty"]

    for store, store_data in state["stores"].items():

        store_doi = store_data["stock"] / store_data["daily_sales"]

        if store_doi < DOI_LIMIT_HIGH:

            required_stock = DOI_LIMIT_HIGH * store_data["daily_sales"]
            gap = required_stock - store_data["stock"]

            qty_to_transfer = min(gap, excess_qty)

            if qty_to_transfer > 0:
                transfer_plan[store] = int(qty_to_transfer)
                excess_qty -= qty_to_transfer

        if excess_qty <= 0:
            break
    total_transfer=sum(transfer_plan.values())
    return {"transfer_plan": transfer_plan,
            "total_transfer_qty":total_transfer}



# In[253]:


def identify_stores_push(state: dict):
    eligible = []
    for store, data in state["stores"].items():
        doi = data["stock"] / data["daily_sales"]
        if doi < DOI_LIMIT_HIGH:
            eligible.append(store)
    return {"eligible_stores": eligible}


# In[254]:


def identify_stores_pull(state: dict):
    eligible = []
    for store, data in state["stores"].items():
        doi = data["stock"] / data["daily_sales"]
        if doi > DOI_LIMIT_HIGH:
            eligible.append(store)
    return {"eligible_stores": eligible}


# In[255]:


def calculate_transfer_pull(state: dict):
    """
    Pull excess inventory from stores back to warehouse.
    Executed via Send() fan-out (one store per execution).
    """

    transfer_plan = {}
    warehouse_shortage = state.get("shortage_qty", 0)

    for store, store_data in state["stores"].items():
        # Compute DOI at store
        store_doi = store_data["stock"] / store_data["daily_sales"]

        # Only consider stores with excess stock
        if store_doi > DOI_LIMIT_HIGH:
            store_excess = store_data["stock"] - store_data["daily_sales"] * DOI_LIMIT_HIGH

            # Pull only what the warehouse needs
            pull_qty = min(store_excess, warehouse_shortage)

            if pull_qty > 0:
                transfer_plan[store] = round(pull_qty)
                # Reduce remaining warehouse shortage
                warehouse_shortage -= pull_qty

        # Stop if warehouse shortage is satisfied
        if warehouse_shortage <= 0:
            break

    if not transfer_plan:
        return {}

    total_transfer=sum(transfer_plan.values())
    return {"transfer_plan": transfer_plan,
            "total_transfer_qty":total_transfer}

   


# In[256]:


def aggregate(state: dict):
    transfers = state.get("transfer_plan", {})  # <-- empty if workers didn't merge
    total_transfer = sum(transfers.values())
    if state["excess_qty"]>0:
         warehouse_after = state["warehouse_stock"] - total_transfer
    elif state["shortage_qty"]>0:
        warehouse_after = state["warehouse_stock"] + total_transfer

    
    return {
        "final_output": {
            "sku": state["sku"],
            "excess_stock": state["excess_qty"],
            "shortage_qty":state["shortage_qty"],
            "warehouse_before": state["warehouse_stock"],
            "warehouse_after": warehouse_after,
            "transfers": transfers,
            "total_transfer_qty": total_transfer
        },
        "transfer_plan": transfers  # keep for LLM
    }


# In[257]:


def assign_workers_pull(state: dict):
    sends = [
        Send("calculate_transfer_pull", {"store": store})
        for store in state["eligible_stores"]
    ]
    return {"__send__": sends}


# In[258]:


from langchain_core.messages import HumanMessage,AIMessage,SystemMessage


# In[259]:


def warehouse_reasoning(state: dict):
    prompt = f"""
You are a Central Warehouse Optimization Agent.

SKU: {state['sku']}
Excess stock: {state['excess_qty']}
Shortage stock: {state['shortage_qty']}
Vendor Policy: {state['vendor_policy']}
Total Transfer Quantity: {state['total_transfer_qty']}

STRICT OUTPUT FORMAT RULES:

1. If Excess Stock > 0:
   Output exactly:
   Push Plan in Action

2. Else if Shortage Stock > 0:
   Output exactly:
   Pull Plan in Action

3. Leave ONE blank line.

4. Print Vendor Return Option exactly in this format:
   Return option to vendor <Vendor_ID>: <Policy_Type>

5. Leave ONE blank line.

6. Print:
   Transfer Plan:
   <exact transfer_plan dictionary>

7. Leave ONE blank line.

8. Final Summary:
   - If shortage case:
     In total we have Shortage of <shortage_qty> qty and we can pull <total_transfer_qty> qty from different locations.
   - If excess case:
     In total we have Excess of <excess_qty> qty and we can push <total_transfer_qty> qty to different locations.

DO NOT add explanations.
DO NOT add extra commentary.
DO NOT modify dictionary format.

Transfer Plan Dictionary:
{state['transfer_plan']}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"llm_recommendation": response.content}



# In[ ]:





# In[260]:


def build_graph():
    builder = StateGraph(State)

    builder.add_node("load_store_data", load_store_data)
    builder.add_node("orchestrator", orchestrator)

    # PUSH
    builder.add_node("identify_stores_push", identify_stores_push)
    builder.add_node("assign_workers_push", assign_workers_push)
    builder.add_node("calculate_transfer_push", calculate_transfer_push)

    # PULL
    builder.add_node("identify_stores_pull", identify_stores_pull)
    builder.add_node("assign_workers_pull", assign_workers_pull)
    builder.add_node("calculate_transfer_pull", calculate_transfer_pull)

    builder.add_node("aggregate", aggregate)
    builder.add_node("llm_agent", warehouse_reasoning)
    builder.add_node("vendor_policy_tool", vendor_policy_tool_node)

    builder.set_entry_point("load_store_data")

    builder.add_edge("load_store_data", "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        route,
        {
            "identify_stores_push": "identify_stores_push",
            "identify_stores_pull": "identify_stores_pull",
            END: END
        }
    )

    # PUSH FLOW
    builder.add_edge("identify_stores_push", "assign_workers_push")
    builder.add_edge("assign_workers_push", "calculate_transfer_push")

    # PULL FLOW
    builder.add_edge("identify_stores_pull", "assign_workers_pull")
    builder.add_edge("assign_workers_pull", "calculate_transfer_pull")

    # Merge
    builder.add_edge("calculate_transfer_push", "aggregate")
    builder.add_edge("calculate_transfer_pull", "aggregate")

    builder.add_edge("aggregate", "vendor_policy_tool")
    builder.add_edge("vendor_policy_tool", "llm_agent")
    builder.add_edge("llm_agent", END)

    return builder.compile()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




