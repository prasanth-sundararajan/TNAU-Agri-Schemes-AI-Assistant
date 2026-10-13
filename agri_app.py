import base64
import io
import os
import re
import shutil
import urllib.parse
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pymupdf  # PyMuPDF for PDF processing
import streamlit as st

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# 0. CONFIGURATION & TRACING
# -----------------------------------------------------------------------------
load_dotenv()

PERSIST_DIRECTORY = "./chroma_tnau_agri_db"
COLLECTION_NAME = "tnau_agri_portal_schemes"
TNAU_SEED_URL = "https://agritech.tnau.ac.in/"

if os.getenv("LANGCHAIN_API_KEY"):
  os.environ["LANGCHAIN_TRACING_V2"] = "true"
  os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
  os.environ["LANGCHAIN_PROJECT"] = os.getenv(
      "LANGCHAIN_PROJECT", "tnau-agri-schemes-assistant"
  )

# -----------------------------------------------------------------------------
# STEP 1: SCHEME EXTRACTION & SANITIZATION HELPERS
# -----------------------------------------------------------------------------
SCHEME_KEYWORDS = [
    "scheme",
    "subsidy",
    "yojana",
    "mission",
    "assistance",
    "kisan",
    "component",
    "insurance",
    "grant",
    "திட்டம்",
    "மானியம்",
    "development programme",
    "procurement",
    "mechanisation",
    "mechanization",
    "nadp",
    "nais",
    "rkvy",
    "smam",
    "pmfby",
    "nfsm",
    "pmksy",
    "iamwarm",
]

EXCLUDE_WORDS = [
    "sl no",
    "name of the scheme",
    "scheme details",
    "how to apply",
    "botany",
    "fermented",
    "idli",
    "season and varieties",
    "table 1",
    "table 2",
]


def clean_scheme_name(title: str) -> str:
  """Removes table header prefixes, serial numbers, and linebreaks."""
  if not title:
    return ""
  t = title.strip()
  t = re.sub(
      r"^(Scheme Table:\s*|\d+[\.\)]\s*)+", "", t, flags=re.IGNORECASE
  ).strip()
  t = re.sub(
      r"^(Name of Scheme\s*:\s*|Details of\s+)", "", t, flags=re.IGNORECASE
  ).strip()
  t = re.sub(r"\s+", " ", t.split("\n")[0].split(";")[0]).strip()
  return t


def is_valid_scheme_button(title: str) -> bool:
  """Only display actual scheme titles as clickable sidebar buttons."""
  if not title or len(title) < 5:
    return False
  lower = title.lower()
  if any(ex in lower for ex in EXCLUDE_WORDS):
    return False
  return any(k in lower for k in SCHEME_KEYWORDS)


def clean_text(text: str) -> str:
  """Removes irregular whitespace, non-breaking spaces, and blank lines."""
  return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


# -----------------------------------------------------------------------------
# STEP 2: MULTIMODAL EXTRACTION & CHILD-URL CRAWLER FOR TNAU
# -----------------------------------------------------------------------------
def describe_image_with_vision(
    image_bytes: bytes, context_source: str, page_num: int
) -> str:
  """Uses GPT-4o-mini Vision to generate descriptions of agricultural scheme diagrams."""
  try:
    b64_img = base64.b64encode(image_bytes).decode("utf-8")
    vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    prompt = (
        "You are an agricultural scheme expert. Describe this image extracted"
        " from a Tamil Nadu government agricultural scheme document. Detail any"
        " machinery, tools, subsidy charts, process flow, or crop conditions"
        " shown."
    )
    msg = vision_llm.invoke([{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
            },
        ],
    }])
    return msg.content
  except Exception as e:
    return f"[Image description omitted: {e}]"


def extract_media_and_text_from_pdf(pdf_bytes: bytes, file_name: str):
  """Extracts text, hyperlinks, URLs, images, and audio/video references from PDFs."""
  documents = []
  discovered_urls = set()

  doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

  for page_idx, page in enumerate(doc):
    page_num = page_idx + 1
    page_text = clean_text(page.get_text() or "")

    for link in page.get_links():
      uri = link.get("uri")
      if uri and (uri.startswith("http://") or uri.startswith("https://")):
        discovered_urls.add(uri)

    for u in re.findall(r"https?://[^\s<>\"']+", page_text):
      discovered_urls.add(u.rstrip(".,;)"))

    if page_text:
      documents.append(
          Document(
              page_content=page_text,
              metadata={
                  "source": file_name,
                  "page": page_num,
                  "type": "pdf_text",
                  "scheme_title": file_name,
              },
          )
      )

    image_list = page.get_images(full=True)
    for img_idx, img_info in enumerate(image_list):
      xref = img_info[0]
      base_image = doc.extract_image(xref)
      image_bytes = base_image["image"]

      img_desc = describe_image_with_vision(image_bytes, file_name, page_num)
      documents.append(
          Document(
              page_content=(
                  f"Image Description (Page {page_num}, Img"
                  f" {img_idx+1}): {img_desc}"
              ),
              metadata={
                  "source": file_name,
                  "page": page_num,
                  "type": "pdf_image_caption",
                  "scheme_title": file_name,
              },
          )
      )

  doc.close()
  return documents, list(discovered_urls)


def parse_schemes_from_html(soup: BeautifulSoup, page_url: str):
  """Parses schemes from TNAU HTML without dropping contextual body data."""
  scheme_docs = []
  page_title = (
      soup.title.string.strip()
      if soup.title and soup.title.string
      else page_url
  )
  cleaned_page_title = clean_scheme_name(page_title)

  tables = soup.find_all("table")
  for tbl_idx, table in enumerate(tables):
    rows = table.find_all("tr")
    if len(rows) > 1:
      table_text = clean_text(table.get_text())
      header_text = clean_text(rows[0].get_text())[:120]
      tbl_title = clean_scheme_name(header_text) or cleaned_page_title

      scheme_docs.append(
          Document(
              page_content=(
                  f"Page: {page_title}\nTable Header:"
                  f" {header_text}\n\nFull Table Details:\n{table_text}"
              ),
              metadata={
                  "source": page_url,
                  "scheme_title": tbl_title,
                  "type": "scheme_table",
              },
          )
      )

  headings = soup.find_all(["h1", "h2", "h3", "h4", "b", "strong"])
  for h in headings:
    h_text = clean_text(h.get_text())
    cleaned_h = clean_scheme_name(h_text)
    if len(cleaned_h) > 5:
      content_parts = []
      for sib in h.next_siblings:
        if (
            getattr(sib, "name", None)
            in ["h1", "h2", "h3", "h4", "b", "strong"]
        ):
          break
        if hasattr(sib, "get_text"):
          content_parts.append(sib.get_text())

      body = clean_text(" ".join(content_parts))
      if len(body) > 40:
        scheme_docs.append(
            Document(
                page_content=(
                    f"Scheme Name / Section:"
                    f" {cleaned_h}\nDetails, Benefits & Eligibility:\n{body}"
                ),
                metadata={
                    "source": page_url,
                    "scheme_title": cleaned_h,
                    "type": "scheme_section",
                },
            )
        )

  for tag in soup(["script", "style", "nav", "footer", "aside"]):
    tag.decompose()
  page_body = clean_text(soup.get_text())
  if len(page_body) > 80:
    scheme_docs.append(
        Document(
            page_content=(
                f"TNAU Scheme Page: {page_title}\n\nContent:\n{page_body}"
            ),
            metadata={
                "source": page_url,
                "scheme_title": cleaned_page_title,
                "type": "web_page",
            },
        )
    )

  return scheme_docs


def deep_crawl_tnau_portal(seed_url=TNAU_SEED_URL, max_pages=40):
  """Deep Crawls TNAU Agritech portal, extracting nested schemes and PDF links."""
  import requests

  session = requests.Session()
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }

  discovered_docs = []
  visited_urls = set()
  queue = [
      seed_url,
      "https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html",
      (
          "https://agritech.tnau.ac.in/agricultural_engineering/agriengg_govt_schemes.html"
      ),
      (
          "https://agritech.tnau.ac.in/govt_schemes_services/govt_serv_schems_main.html"
      ),
  ]

  while queue and len(visited_urls) < max_pages:
    curr_url = queue.pop(0)
    if curr_url in visited_urls:
      continue
    visited_urls.add(curr_url)

    try:
      resp = session.get(curr_url, headers=headers, timeout=12)
      resp.encoding = resp.apparent_encoding or "utf-8"
      if resp.status_code != 200:
        continue

      if curr_url.lower().endswith(".pdf") or "application/pdf" in resp.headers.get(
          "Content-Type", ""
      ):
        file_title = curr_url.split("/")[-1]
        pdf_docs, _ = extract_media_and_text_from_pdf(resp.content, file_title)
        discovered_docs.extend(pdf_docs)
        continue

      soup = BeautifulSoup(resp.text, "html.parser")

      for frame in soup.find_all(["frame", "iframe"]):
        src = frame.get("src")
        if src:
          f_url = urljoin(curr_url, src)
          if f_url not in visited_urls and f_url not in queue:
            queue.append(f_url)

      for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "javascript:", "mailto:")):
          continue
        full_url = urljoin(curr_url, href)
        if "agritech.tnau.ac.in" in full_url or "tn.gov.in" in full_url:
          if full_url not in visited_urls and full_url not in queue:
            queue.append(full_url)

      page_schemes = parse_schemes_from_html(soup, curr_url)
      discovered_docs.extend(page_schemes)

    except Exception as e:
      print(f"Error crawling {curr_url}: {e}")

  return discovered_docs


# -----------------------------------------------------------------------------
# STEP 3: PERSISTENT VECTOR STORE & HYBRID SEARCH
# -----------------------------------------------------------------------------
def get_vector_store():
  """Returns persistent ChromaDB instance."""
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  return Chroma(
      collection_name=COLLECTION_NAME,
      embedding_function=embeddings,
      persist_directory=PERSIST_DIRECTORY,
  )


def ingest_documents(documents):
  """Splits documents and commits to Chroma vector store."""
  if not documents:
    return 0

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1400, chunk_overlap=250, separators=["\n\n", "\n", ". ", " ", ""]
  )
  split_docs = text_splitter.split_documents(documents)
  vector_store = get_vector_store()
  vector_store.add_documents(split_docs)
  return len(split_docs)


def hybrid_retrieve(query: str, vector_store, k=8) -> list[Document]:
  """Combines semantic vector search with BM25 keyword matching."""
  semantic_docs = vector_store.similarity_search(query, k=k)

  try:
    data = vector_store._collection.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]
    if all_docs:
      bm25_retriever = BM25Retriever.from_documents(all_docs)
      bm25_retriever.k = k
      bm25_docs = bm25_retriever.invoke(query)
    else:
      bm25_docs = []
  except Exception:
    bm25_docs = []

  seen = set()
  combined = []
  for d in semantic_docs + bm25_docs:
    if d.page_content not in seen:
      seen.add(d.page_content)
      combined.append(d)

  return combined[:k]


def rerank_documents(
    query: str, docs: list[Document], top_n: int = 5
) -> list[Document]:
  """Reranks retrieved context documents using LLM relevance scoring."""
  if not docs:
    return []

  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
  doc_snippets = "\n\n".join(
      [f"[{idx}] {d.page_content[:400]}" for idx, d in enumerate(docs)]
  )

  prompt = f"""Given the user query: "{query}"
Rank the following document snippets in order of relevance from most relevant to least relevant.
Return ONLY a comma-separated list of indices (e.g., 2,0,4,1) representing the top {min(top_n, len(docs))} best documents.

Snippets:
{doc_snippets}

Indices:"""

  try:
    response = llm.invoke(prompt).content.strip()
    indices = [
        int(i.strip())
        for i in response.replace(";", ",").split(",")
        if i.strip().isdigit()
    ]
    ranked = [docs[i] for i in indices if i < len(docs)]
    for d in docs:
      if d not in ranked and len(ranked) < top_n:
        ranked.append(d)
    return ranked[:top_n]
  except Exception:
    return docs[:top_n]


def format_docs(docs):
  formatted = []
  for d in docs:
    src = d.metadata.get("source", "TNAU Portal")
    sch_title = d.metadata.get("scheme_title", "")
    header = (
        f"--- Source: {src}"
        + (f" | Scheme: {sch_title}" if sch_title else "")
        + " ---"
    )
    formatted.append(f"{header}\n{d.page_content}")
  return "\n\n".join(formatted)


# -----------------------------------------------------------------------------
# STEP 4: PROMPT ENGINEERING (BILINGUAL ENGLISH 1ST, TAMIL 2ND)
# -----------------------------------------------------------------------------
MULTILINGUAL_RAG_PROMPT = """You are an official Tamil Nadu Agriculture Schemes Assistant (தமிழ்நாடு வேளாண் திட்டங்கள் உதவி மையம்) for TNAU (https://agritech.tnau.ac.in/).

Your task is to provide comprehensive, clear, and actionable information regarding Tamil Nadu agricultural schemes using the retrieved context below.

Bilingual Display Rules (English 1st, Tamil 2nd):
1. EVERY section, heading, explanation, and bullet point MUST be provided in BOTH English and Tamil.
2. For every detail, present the **English version first**, immediately followed by its **complete Tamil translation** (உள்ளடக்கங்கள் ஆங்கிலத்தில் முதலில், அதைத் தொடர்ந்து தமிழில் முழுமையாக வழங்கப்பட வேண்டும்).
3. Do not omit Tamil translations for bullet points or descriptions. Every bullet must have its English text and its Tamil translation side-by-side or stacked cleanly.

Answer Structure Guidelines:
1. Scheme Name & Objective | திட்டத்தின் பெயர் மற்றும் நோக்கம் (English details first, followed by full Tamil explanation)
2. Benefits & Subsidies | மானியம் மற்றும் சலுகைகள் (English details first, followed by full Tamil explanation)
3. Eligibility Criteria | தகுதி வரம்புகள் (English details first, followed by full Tamil explanation)
4. Required Documents & Application Process | தேவையான ஆவணங்கள் மற்றும் விண்ணப்பிக்கும் முறை (English details first, followed by full Tamil explanation)
5. Contact Officials | தொடர்பு கொள்ள வேண்டிய அலுவலர்கள் (English details first, followed by full Tamil explanation)

Retrieved Context:
{context}

User Question:
{question}

Comprehensive Bilingual Answer:"""

# -----------------------------------------------------------------------------
# STEP 5: STREAMLIT UI IMPLEMENTATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TNAU Agri Schemes AI Assistant", page_icon="🌾", layout="wide"
)

# Custom Styling
st.markdown(
    """
<style>
    .main { background-color: #f8fafc; }
    .header-banner {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-banner h1 { color: #ffffff !important; font-size: 2.2rem; margin-bottom: 6px; }
    .header-banner p { color: #a7f3d0 !important; font-size: 1.05rem; }
    div[data-testid="stSidebarUserContent"] button[kind="secondary"] {
        text-align: left;
        white-space: normal;
        height: auto;
        padding-top: 8px;
        padding-bottom: 8px;
        margin-bottom: 6px;
        font-size: 0.85rem;
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        background-color: #ffffff;
    }
    div[data-testid="stSidebarUserContent"] button[kind="secondary"]:hover {
        border-color: #10b981;
        color: #047857;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
  st.session_state.messages = []
if "pending_query" not in st.session_state:
  st.session_state.pending_query = None
if "portal_auto_ingested" not in st.session_state:
  st.session_state.portal_auto_ingested = False


def run_full_reingest():
  """Wipes old index and crawls TNAU schemes fresh."""
  if os.path.exists(PERSIST_DIRECTORY):
    try:
      shutil.rmtree(PERSIST_DIRECTORY)
    except Exception:
      pass

  with st.status(
      "🌾 Crawling TNAU Agritech Portal & Building Hybrid DB...", expanded=True
  ) as status:
    st.write(f"Initiating deep crawl on: `{TNAU_SEED_URL}`")
    crawled_docs = deep_crawl_tnau_portal(seed_url=TNAU_SEED_URL, max_pages=40)
    st.write(
        f"Extracted {len(crawled_docs)} scheme documents & sub-pages. Splitting"
        " & vectorizing..."
    )
    chunks_count = ingest_documents(crawled_docs)
    status.update(
        label=f"Done! Ingested {chunks_count} complete scheme chunks.",
        state="complete",
    )

  st.session_state.portal_auto_ingested = True
  st.success("TNAU Portal indexing completed!")
  st.rerun()


# Banner Header
st.markdown(
    """
    <div class="header-banner">
        <h1>🌾 TNAU Agri Schemes AI Assistant</h1>
        <p>தமிழ்நாடு வேளாண்மை பல்கலைக்கழக தகவல் முனை - English 1st & Tamil 2nd Bilingual Assistance</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Sidebar UI
with st.sidebar:
  st.markdown("### ⚙️ Portal Data Sync")

  try:
    vs_check = get_vector_store()
    initial_chunk_count = vs_check._collection.count()
  except Exception:
    initial_chunk_count = 0

  if initial_chunk_count == 0 and not st.session_state.portal_auto_ingested:
    run_full_reingest()

  if st.button(
      "🔄 Re-Index TNAU Portal (https://agritech.tnau.ac.in/)",
      type="primary",
      use_container_width=True,
  ):
    run_full_reingest()

  st.markdown("---")

  st.markdown("#### 📂 Custom Scheme PDF Ingestion")
  uploaded_files = st.file_uploader(
      "Upload Scheme PDF Notice", type=["pdf"], accept_multiple_files=True
  )

  if st.button("📥 Index Uploaded PDFs", use_container_width=True):
    if not uploaded_files:
      st.warning("Please upload a PDF file.")
    else:
      with st.status(
          "Processing PDF Text & Vision Media...", expanded=True
      ) as status:
        all_pdf_docs = []
        for f in uploaded_files:
          st.write(f"Parsing: `{f.name}`")
          bytes_data = f.read()
          docs, _ = extract_media_and_text_from_pdf(bytes_data, f.name)
          all_pdf_docs.extend(docs)

        chunks = ingest_documents(all_pdf_docs)
        status.update(
            label=f"Done! Added {chunks} new chunks.", state="complete"
        )
      st.success("PDF successfully indexed!")
      st.rerun()

  st.markdown("---")

  st.markdown("#### 🏛️ Indexed TNAU Schemes Explorer")
  try:
    vs = get_vector_store()
    col = vs._collection
    count = col.count()
    st.metric(label="Total Chunks in Vector DB", value=count)

    if count > 0:
      with st.expander("🌾 Clickable Scheme List", expanded=True):
        all_metas = col.get(include=["metadatas"])["metadatas"]

        unique_schemes = set()
        for m in all_metas:
          raw_title = m.get("scheme_title") or ""
          cleaned = clean_scheme_name(raw_title)
          if cleaned and is_valid_scheme_button(cleaned):
            unique_schemes.add(cleaned)

        sorted_schemes = sorted(list(unique_schemes))

        if sorted_schemes:
          for idx, sch in enumerate(sorted_schemes):
            btn_label = (sch[:40] + "...") if len(sch) > 42 else sch
            if st.button(
                f"📌 {btn_label}", key=f"sch_btn_{idx}", use_container_width=True
            ):
              st.session_state.pending_query = (
                  f"Provide all details for '{sch}' including subsidies,"
                  " eligibility, and application method in both English and"
                  " Tamil."
              )
              st.rerun()
        else:
          st.info("Click 'Re-Index TNAU Portal' above to fetch schemes.")
  except Exception:
    st.caption("Initializing vector database...")

# Main Chat Display with WhatsApp Sharing Option
for i, msg in enumerate(st.session_state.messages):
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])
    if msg.get("sources"):
      with st.expander("📂 Referenced Scheme Sources"):
        for src in msg["sources"]:
          st.markdown(f"- `{src}`")

    if msg["role"] == "assistant":
      wa_text = urllib.parse.quote(
          f"🌾 *Tamil Nadu Agri Scheme Info*:\n\n{msg['content'][:1500]}\n\nShared"
          " via TNAU Agri AI Assistant"
      )
      wa_url = f"https://api.whatsapp.com/send?text={wa_text}"
      st.markdown(
          f'<a href="{wa_url}" target="_blank"><button'
          ' style="background-color:#25D366; color:white; padding:6px 14px;'
          " border:none; border-radius:4px; cursor:pointer; font-size:13px;"
          f' margin-top:8px;"><b>📲 Share on WhatsApp</b></button></a>',
          unsafe_allow_html=True,
      )

# Quick Query Suggestions
st.markdown("##### 💡 Suggested Questions / மாதிரி கேள்விகள்:")
col1, col2, col3 = st.columns(3)
if col1.button("🌾 Paddy Transplanter Subsidy?"):
  st.session_state.pending_query = (
      "What is the subsidy provided for self propelled paddy transplanter"
      " machines in Tamil Nadu?"
  )
  st.rerun()
if col2.button("🚜 நெல் நடவு இயந்திர மானியம்?"):
  st.session_state.pending_query = (
      "நெல் நடவு இயந்திரம் மற்றும் விவசாய கருவிகளுக்கான மானியம் எவ்வளவு?"
  )
  st.rerun()
if col3.button("💧 Drip Irrigation Assistance?"):
  st.session_state.pending_query = (
      "What are the drip irrigation and micro irrigation scheme subsidies"
      " available for farmers?"
  )
  st.rerun()

# Handle User Input
active_query = None
chat_input_val = st.chat_input(
    "Ask any query in English or தமிழ் (e.g., How to apply for seed village"
    " subsidy? / விதை கிராம திட்டத்தில் பயன் பெற எப்படி விண்ணப்பிப்பது?)"
)

if st.session_state.pending_query:
  active_query = st.session_state.pending_query
  st.session_state.pending_query = None
elif chat_input_val:
  active_query = chat_input_val

# Execute RAG Chain
if active_query:
  st.session_state.messages.append({"role": "user", "content": active_query})
  with st.chat_message("user"):
    st.markdown(active_query)

  vector_store = get_vector_store()

  with st.chat_message("assistant"):
    with st.spinner(
        "Searching TNAU portal database (Hybrid Search + Reranking)..."
    ):
      retrieved_docs = hybrid_retrieve(active_query, vector_store, k=8)
      reranked_docs = rerank_documents(active_query, retrieved_docs, top_n=5)
      context_str = format_docs(reranked_docs)
      prompt = ChatPromptTemplate.from_template(MULTILINGUAL_RAG_PROMPT)
      llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

      chain = prompt | llm | StrOutputParser()
      answer = chain.invoke({"context": context_str, "question": active_query})

      st.markdown(answer)

      cited_sources = []
      for d in reranked_docs:
        title = d.metadata.get("scheme_title")
        src = d.metadata.get("source", "Unknown")
        label = f"{title} ({src})" if title else src
        if label not in cited_sources:
          cited_sources.append(label)

      if cited_sources:
        with st.expander("📂 Referenced TNAU Scheme Sources"):
          for s in cited_sources:
            st.markdown(f"- `{s}`")

      wa_text = urllib.parse.quote(
          f"🌾 *Tamil Nadu Agri Scheme Info*:\n\n{answer[:1500]}\n\nShared via"
          " TNAU Agri AI Assistant"
      )
      wa_url = f"https://api.whatsapp.com/send?text={wa_text}"
      st.markdown(
          f'<a href="{wa_url}" target="_blank"><button'
          ' style="background-color:#25D366; color:white; padding:6px 14px;'
          " border:none; border-radius:4px; cursor:pointer; font-size:13px;"
          f' margin-top:8px;"><b>📲 Share on WhatsApp</b></button></a>',
          unsafe_allow_html=True,
      )

      st.session_state.messages.append({
          "role": "assistant",
          "content": answer,
          "sources": cited_sources,
      })