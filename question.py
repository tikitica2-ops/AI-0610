# pip install --upgrade streamlit langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv
# 실행: streamlit run app.py

import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ──────────────────────────────────────────────
# RAG 체인 구성 (업로드된 PDF 바이트 기준으로 캐싱 → 같은 파일은 1회만 처리)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="📄 PDF를 읽고 벡터DB를 만드는 중...")
def build_rag_chain(pdf_bytes: bytes, file_name: str):
    # 1. 업로드된 바이트를 임시 파일로 저장 (PyPDFLoader는 파일 경로가 필요)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # 2. PDF 로드 & 페이지 분할
    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()

    # 3. 텍스트 청크 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,            # 하나의 청크가 가질 최대 글자 수
        chunk_overlap=20,          # 청크 간 문맥 연결을 위해 겹칠 글자 수
        length_function=len,       # 길이 측정 기준
        is_separator_regex=False,  # 구분 기호의 정규표현식 해석 여부
    )
    texts = text_splitter.split_documents(pages)

    # 4. 임베딩 & 벡터DB 구축
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(texts, embeddings_model)

    # 5. LLM & 멀티 쿼리 리트리버
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm,
    )

    # 6. 프롬프트 & 체인
    system_prompt = (
        "너는 질문-답변을 돕는 유능한 비서야. "
        "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
        "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)
    return rag_chain


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="PDF 질문-답변 RAG", page_icon="📚")
st.title("📚 PDF 기반 질문-답변")
st.caption("PDF 문서 내용을 기반으로 질문에 답합니다.")

# 사이드바: PDF 업로드 & 옵션
with st.sidebar:
    st.header("설정")
    uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
    show_context = st.checkbox("참조 문서 함께 보기", value=False)
    if st.button("🗑️ 대화 기록 지우기"):
        st.session_state.messages = []
        st.rerun()

# PDF가 업로드되지 않았으면 안내 후 중단
if uploaded_file is None:
    st.info("👈 왼쪽 사이드바에서 PDF 파일을 업로드하면 질문할 수 있습니다.")
    st.stop()

# 업로드된 파일로 체인 준비 (같은 파일이면 캐시 재사용)
pdf_bytes = uploaded_file.getvalue()
rag_chain = build_rag_chain(pdf_bytes, uploaded_file.name)

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 기록을 화면에 다시 그리기 (누적 표시)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and show_context and msg.get("context"):
            with st.expander(f"📑 참조 문서 ({len(msg['context'])}개)"):
                for i, ctx in enumerate(msg["context"], start=1):
                    st.markdown(f"**문서 {i} (p.{ctx['page']})**")
                    st.write(ctx["content"])

# 질문 입력 (화면 하단 고정)
question = st.chat_input("질문을 입력하세요")

if question:
    # 사용자 질문을 기록에 추가하고 즉시 표시
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중..."):
            response = rag_chain.invoke({"input": question})
        answer = response["answer"]
        st.write(answer)

        docs = response.get("context", [])
        ctx_list = [
            {"page": d.metadata.get("page", "?"), "content": d.page_content}
            for d in docs
        ]
        if show_context and ctx_list:
            with st.expander(f"📑 참조 문서 ({len(ctx_list)}개)"):
                for i, ctx in enumerate(ctx_list, start=1):
                    st.markdown(f"**문서 {i} (p.{ctx['page']})**")
                    st.write(ctx["content"])

    # 답변을 기록에 추가 (참조 문서도 함께 저장)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": ctx_list,
    })