# pip install --upgrade streamlit langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv
# 실행: streamlit run app.py

import os
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
# RAG 체인 구성 (무거운 작업은 캐싱해서 1회만 실행)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="📄 PDF를 읽고 벡터DB를 만드는 중...")
def build_rag_chain(pdf_path: str):
    # 1. PDF 로드 & 페이지 분할
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    # 2. 텍스트 청크 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,            # 하나의 청크가 가질 최대 글자 수
        chunk_overlap=20,          # 청크 간 문맥 연결을 위해 겹칠 글자 수
        length_function=len,       # 길이 측정 기준
        is_separator_regex=False,  # 구분 기호의 정규표현식 해석 여부
    )
    texts = text_splitter.split_documents(pages)

    # 3. 임베딩 & 벡터DB 구축
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(texts, embeddings_model)

    # 4. LLM & 멀티 쿼리 리트리버
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm,
    )

    # 5. 프롬프트 & 체인
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

# 사이드바: PDF 경로 설정 (기본값 unsu.pdf)
with st.sidebar:
    st.header("설정")
    pdf_path = st.text_input("PDF 파일 경로", value="unsu.pdf")
    show_context = st.checkbox("참조 문서 함께 보기", value=False)

# PDF 존재 여부 확인 후 체인 준비
if not os.path.exists(pdf_path):
    st.warning(f"'{pdf_path}' 파일을 찾을 수 없습니다. 사이드바에서 경로를 확인해 주세요.")
    st.stop()

rag_chain = build_rag_chain(pdf_path)

# 질문 입력
question = st.text_input(
    "질문을 입력하세요",
    placeholder="예) 아내가 사다 달라고 했던 음식과, 문맥상 추정할 수 있는 아내가 좋아하는 음식은?",
)

if st.button("질문하기", type="primary") and question.strip():
    with st.spinner("답변을 생성하는 중..."):
        response = rag_chain.invoke({"input": question})

    st.subheader("💡 답변")
    st.write(response["answer"])

    if show_context:
        docs = response.get("context", [])
        st.subheader(f"📑 참조 문서 ({len(docs)}개)")
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page", "?")
            with st.expander(f"문서 {i} (p.{page})"):
                st.write(doc.page_content)