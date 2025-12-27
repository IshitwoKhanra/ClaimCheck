import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
import json
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

llm=ChatOpenAI(temperature=0)

#Functions
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        reader=PdfReader(pdf)
        for page in reader.pages:
           page_text=page.extract_text()
           if page_text:
                text+=page_text
    return text

def get_text_chunks(text):
    splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

def get_vectorstore(chunks):
    embeddings=OpenAIEmbeddings()
    return FAISS.from_texts(chunks,embeddings)

def run_rag(question,vectordb):
    retriever=vectordb.as_retriever()

    docs=retriever.invoke(question)

    if not docs:
        return "I don't know",[]

    context="\n\n".join([d.page_content for d in docs])

    messages=[
        {
            "role":"user",
            "content":f"""
            Answer the question using ONLY the context below.

            Rules:
            - Use short, explicit factual sentences.
            - Do NOT add explanations or examples.
            - Do NOT merge multiple facts into one sentence.
            - If the answer is not fully present in the context, say "I don't know".

            Context:
            {context}

            Question:
            {question}
            """
        }
    ]

    response=llm.invoke(messages)

    return response.content,docs

def extract_claims(answer):
    prompt=f"""You are extracting factual claims for verification.

    Rules:
    - Extract ONLY explicit factual statements.
    - Each claim must be atomic (one fact only).
    - Do NOT paraphrase or infer.
    - Do NOT include opinions, explanations, or examples.
    - Do NOT add new facts.
    - If the text contains no factual claims, return an empty list.
    Return a strict JSON list:
    [
        {{
            "claim":"..."
        }}
    ]

    Text:
    {answer}
    """

    response=llm.invoke(prompt)
    try:
        claims = json.loads(response.content)
        return claims
    except json.JSONDecodeError:
        return []

def retrieve_evidence_for_claim(claim,vectordb,k=6):
    retriever=vectordb.as_retriever(search_kwargs={"k":k})
    return retriever.invoke(claim)

def verify_claims(claim,evidence_doc):

    evidence_text="\n\n".join([d.page_content for d in evidence_doc])

    prompt=f"""
    You are a factual verifier.

    Your task is to determine whether the EVIDENCE supports the CLAIM.
    Use only the evidence provided. Do not use outside knowledge.

    CLAIM:
    {claim}

    EVIDENCE:
    {evidence_text}

    Decide the relationship between the claim and the evidence:

    SUPPORTED:
    - The evidence explicitly states the same fact about the same entity.
    - Definitions, numeric facts, or listed components match in meaning.

    PARTIALLY_SUPPORTED:
    - The claim is generally correct but incomplete.
    - The evidence lists additional components or stages not mentioned in the claim.
    - The claim summarizes a multi-stage process using different wording.
    - If a claim mixes core components with optional variants or extensions,
    the claim MUST be marked PARTIALLY_SUPPORTED, not NOT_SUPPORTED.

    NOT_SUPPORTED:
    - The evidence contradicts the claim.
    - The same result or metric is attributed to a different entity.
    - The evidence does not mention the claim at all.

    Rules:
    - Missing information ≠ contradiction.
    - Enumerated components or stages may be summarized as a multi-step approach.
    - Do not penalize paraphrasing if meaning is equivalent.
    - If unsure between NOT_SUPPORTED and PARTIALLY_SUPPORTED, choose PARTIALLY_SUPPORTED.
    - If a claim states that a method, technique, component, concept, or approach exists, is used, or is discussed,the claim MUST be marked SUPPORTED as long as the evidence mentions or describes it in any relevant capacity.
    Do NOT require the evidence to state that the item is:
    - necessary
    - sufficient
    - exclusive
    - optimal
    - complete
    unless the claim explicitly asserts such conditions.

    Return EXACTLY in this format:

    Answer EXACTLY with one of the following ONLY. DO NOT INCLUDE TAGS LIKE 'ANSWER'/'VERDICT' before SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED:
    SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED \n\n
    REASON: <one sentence grounded in evidence>
    """
    response=llm.invoke(prompt)
    return response.content

def compute_score(verdicts):
    if not verdicts:
        return 1.0

    supported = sum(v.strip().upper().startswith("SUPPORTED") for v in verdicts)
    partially_supported=sum(v.strip().upper().startswith("PARTIALLY_SUPPORTED") for v in verdicts) * 0.5
    return (supported+partially_supported) / len(verdicts)



def hallucination_guard(answer, vectordb):
    if answer.lower().strip() == "i don't know":
        return 1.0, []
    
    claims = extract_claims(answer)

    results = []

    for c in claims:
        claim_text = c["claim"]
        evidence = retrieve_evidence_for_claim(claim_text, vectordb)
        verdict = verify_claims(claim_text, evidence)

        results.append({
            "claim": claim_text,
            "verdict": verdict
        })

    score = compute_score([r["verdict"] for r in results])

    return score, results



st.set_page_config(page_title="AI Powered Hallucination Guard")
with st.container():
    st.markdown(
        """
        <h1 style="text-align:center;">ClaimCheck</h1>
        <p style="text-align:center; font-size:18px; opacity:0.85;">
        Verify LLM answers claim-by-claim using only your documents.
        </p>
        <p style="text-align:center; font-size:14px; opacity:0.65;">
        Upload PDFs → Ask questions → Get evidence-grounded correctness scores
        </p>
        """,
        unsafe_allow_html=True
    )
st.divider()
steps = [
    ("📄 Upload PDFs", "Add source documents"),
    ("⚙️ Process", "Index & chunk content"),
    ("❓ Ask", "Query the documents"),
    ("🧠 Verify", "Extract factual claims"),
    ("📊 Score", "Check grounded correctness"),
]

cols = st.columns(len(steps))
for col, (title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""<p style="text-align:center; font-size:16px; opacity:0.9;">{title}</p>""",unsafe_allow_html=True)
        st.markdown(f"""<p style="text-align:center; font-size:14px; opacity:0.65;">{desc}</p>""",unsafe_allow_html=True)
st.divider()

if "vectordb" not in st.session_state:
    st.session_state.vectordb=None

user_question=st.text_input("Ask a question")

if user_question and st.session_state.vectordb:
    answer,docs=run_rag(user_question,st.session_state.vectordb)

    st.write("### Answer")
    st.write(answer)
    score, guard_results = hallucination_guard(answer, st.session_state.vectordb)

    st.write("### Correctness Score")
    st.write(f"Score: {score}")
    st.progress(score)

    with st.expander("Claim Verification"):
        for r in guard_results:
            st.write(f"**Claim:** {r['claim']}")
            st.write(f"**Verdict:** {r['verdict']}")
            st.divider()
    if score < 0.75:
        st.warning("Potential Hallucination Detected")
    else:
        st.success("No Hallucination detected")


with st.sidebar:
    st.subheader("Upload your PDF:")
    pdf=st.file_uploader("Upload Files",accept_multiple_files=True)
    
    if pdf:
        st.success(f"""{len(pdf)} PDF(s) uploaded""")

    if st.button("Process"):
        if not pdf:
            st.warning("Uplaod atleast one pdf")
        else:
            with st.spinner("Processing...."):
                text=get_pdf_text(pdf)
                chunks=get_text_chunks(text)
                st.session_state.vectordb=get_vectorstore(chunks)

            st.success("Ready! Ask your questions")
    
    st.markdown(
    """
    <style>
    /* Sidebar container */
    [data-testid="stSidebar"] {
        position: relative;
    }

    /* Sidebar footer */
    .sidebar-footer {
        position: fixed;
        bottom: 10px;
        left: 20px;
        font-size: 0.8rem;
        color: #9ca3af;
        opacity: 0.8;
        z-index: 100;
    }
    </style>

    <div class="sidebar-footer">
        👨‍💻 Made by <b>Ishitwo Khanra</b>
    </div>
    """,
    unsafe_allow_html=True
    )