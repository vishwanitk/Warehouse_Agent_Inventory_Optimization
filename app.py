# app.py
import streamlit as st
from agentic_ai import build_graph,StateGraph

# Set a heading
st.title("Centra Warehouse Inventory Optimization Agent")
graph = build_graph()


# Input fields
sku = st.selectbox("Select SKU", [f"SKU_{i}" for i in range(100)])
warehouse_stock = st.number_input("Warehouse Stock", min_value=0, value=100)
warehouse_daily_sales = st.number_input("Warehouse Daily Sales", min_value=1, value=10)
DOI=warehouse_stock/warehouse_daily_sales 
st.write(f"DOI: {DOI}")

# Run agent
# Run agent
if st.button("Run Agent"):
    try:
        result = graph.invoke({
            "sku": sku,
            "warehouse_stock": warehouse_stock,
            "warehouse_daily_sales": warehouse_daily_sales
        })

        st.subheader("Final Output")
        st.json(result.get("final_output", {}))

        st.subheader("LLM Recommendation")
        st.text(result.get("llm_recommendation", ""))

    except Exception as e:
        st.error(f"Error running agent: {e}")