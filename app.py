import streamlit as st
from llm import get_ai_messages
from dotenv import load_dotenv
import os


st.title("Stramlit 기본예제")
st.write("소득세에 관련된 모든 것을 답변해드립니다.")

load_dotenv()

if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀하세요"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("AI is thinking... please wait"):
        ai_response = get_ai_messages(user_question)
        with st.chat_message("ai"):
             ai_message = st.write_stream(ai_response)
    st.session_state.message_list.append({"role": "ai", "content": ai_message})

# print(f"after == {st.session_state.message_list}")