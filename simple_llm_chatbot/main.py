import streamlit as st
from langchina_helper import create_vector_db, get_qa_chain

st.title("Course Q n' A ")
btn = st.button("Create Knowledgebase")
if btn:
    pass

question = st.text_input("Question: ")
if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer: ")
    st.write(response["result"])