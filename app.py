from __future__ import annotations

import hashlib
import io
import os
import time
from dataclasses import dataclass
from urllib.parse import quote

import requests
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class TextChunk:
    text: str
    source: str
    page: int


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 250) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def read_pdf(uploaded_file) -> list[TextChunk]:
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    chunks: list[TextChunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        chunks.extend(
            TextChunk(text=chunk, source=uploaded_file.name, page=page_number)
            for chunk in chunk_text(page_text)
        )
    return chunks


def file_signature(uploaded_files) -> str:
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def build_index(uploaded_files):
    chunks: list[TextChunk] = []
    for uploaded_file in uploaded_files:
        chunks.extend(read_pdf(uploaded_file))
    if not chunks:
        return [], None, None
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunk.text for chunk in chunks)
    return chunks, vectorizer, matrix


def retrieve(
    question: str,
    chunks: list[TextChunk],
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = 5,
) -> list[tuple[TextChunk, float]]:
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix).ravel()
    best_indices = scores.argsort()[::-1][:top_k]
    return [(chunks[index], float(scores[index])) for index in best_indices]


def answer_question(
    client: OpenAI,
    question: str,
    retrieved: list[tuple[TextChunk, float]],
    chat_model: str,
) -> str:
    context = "\n\n".join(
        f"[{index}] {chunk.source}, page {chunk.page}\n{chunk.text}"
        for index, (chunk, _) in enumerate(retrieved, start=1)
    )
    response = client.chat.completions.create(
        model=chat_model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using only the supplied document excerpts. "
                    "If the excerpts do not contain enough information, say so clearly. "
                    "Cite supporting excerpts inline using [1], [2], etc. Do not invent facts."
                ),
            },
            {
                "role": "user",
                "content": f"Document excerpts:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content or "I could not generate an answer."


def create_visual_brief(
    client: OpenAI,
    request: str,
    retrieved: list[tuple[TextChunk, float]],
    chat_model: str,
) -> str:
    source_text = "\n\n".join(
        f"{chunk.source}, page {chunk.page}: {chunk.text}" for chunk, _ in retrieved
    )
    response = client.chat.completions.create(
        model=chat_model,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an art director converting source material into an image prompt. "
                    "Use only facts present in the source. Create one concise visual brief with "
                    "subject, composition, visual hierarchy, color palette, and style. "
                    "Avoid invented numbers, logos, brand names, and long readable text. "
                    "Keep it under 900 characters."
                ),
            },
            {
                "role": "user",
                "content": f"Visual request: {request}\n\nSource material:\n{source_text[:6000]}",
            },
        ],
    )
    return response.choices[0].message.content or request


def generate_image(prompt: str, provider: str, hf_token: str = "") -> tuple[bytes, str]:
    if provider == "Hugging Face · FLUX.1-schnell":
        if not hf_token:
            raise ValueError("Add an HF_TOKEN to use Hugging Face image generation.")
        response = None
        for attempt in range(3):
            response = requests.post(
                "https://router.huggingface.co/hf-inference/models/"
                "black-forest-labs/FLUX.1-schnell",
                headers={"Authorization": f"Bearer {hf_token}"},
                json={"inputs": prompt},
                timeout=120,
            )
            if response.status_code != 503:
                break
            if attempt < 2:
                time.sleep(5)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise RuntimeError(f"Hugging Face returned {response.text[:500]}")
        return response.content, content_type

    # Pollinations receives the prompt in the URL, so keep it below common URL limits.
    compact_prompt = prompt[:1800]
    image_url = "https://image.pollinations.ai/prompt/" + quote(compact_prompt, safe="")
    response = requests.get(
        image_url,
        params={"model": "flux", "width": 1024, "height": 1024, "nologo": "true"},
        timeout=120,
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise RuntimeError(f"Pollinations returned {response.text[:500]}")
    return response.content, content_type


def main() -> None:
    st.set_page_config(page_title="Papertrail", page_icon="📚", layout="wide")
    st.title("Papertrail")
    st.caption("Ask grounded questions across your uploaded PDF library.")

    with st.sidebar:
        st.header("Library")
        api_key = st.text_input(
            "Groq API key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Used only for this Streamlit session unless you set GROQ_API_KEY.",
        )
        chat_model = st.selectbox(
            "Answer model",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "openai/gpt-oss-20b",
                "qwen/qwen3-32b",
            ],
        )
        st.divider()
        st.header("Image studio")
        image_provider = st.selectbox(
            "Image provider",
            ["Pollinations · no key", "Hugging Face · FLUX.1-schnell"],
        )
        hf_token = st.text_input(
            "Hugging Face token",
            value=os.getenv("HF_TOKEN", ""),
            type="password",
            help="Required only for the Hugging Face provider.",
        )
        uploaded_files = st.file_uploader(
            "Upload subject PDFs",
            type="pdf",
            accept_multiple_files=True,
            help="You can upload one or more PDFs. Text-based PDFs work best.",
        )
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} PDF(s) selected")

    if not api_key:
        st.info("Add a Groq API key in the sidebar to ask questions.")
        st.stop()
    if not uploaded_files:
        st.info("Upload one or more PDFs to begin.")
        st.stop()

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    signature = file_signature(uploaded_files)
    if st.session_state.get("index_signature") != signature:
        with st.spinner("Reading and indexing your PDFs..."):
            chunks, vectorizer, matrix = build_index(uploaded_files)
        st.session_state.index_signature = signature
        st.session_state.chunks = chunks
        st.session_state.vectorizer = vectorizer
        st.session_state.matrix = matrix
        st.session_state.messages = []

    chunks = st.session_state.get("chunks", [])
    vectorizer = st.session_state.get("vectorizer")
    matrix = st.session_state.get("matrix")
    if not chunks:
        st.error("No selectable text was found. Try a text-based PDF or add OCR first.")
        st.stop()

    st.subheader("Generate an image from your PDFs")
    st.caption("Use the indexed PDF content to create an infographic, diagram, or visual concept.")
    visual_request = st.text_area(
        "What should the image show?",
        placeholder="Create a clear infographic explaining the main stages and benefits",
    )
    if st.button("Generate image from PDF", type="primary", disabled=not visual_request.strip()):
        with st.spinner("Finding relevant PDF content and generating image..."):
            try:
                visual_sources = retrieve(
                    visual_request, chunks, vectorizer, matrix, top_k=5
                )
                image_prompt = create_visual_brief(
                    client, visual_request, visual_sources, chat_model
                )
                image_bytes, mime_type = generate_image(
                    image_prompt, image_provider, hf_token
                )
                st.session_state.generated_image = image_bytes
                st.session_state.generated_image_type = mime_type
                st.session_state.generated_image_caption = visual_request
                st.session_state.generated_visual_brief = image_prompt
            except Exception as error:
                st.error("Image generation failed.")
                st.caption(f"Image provider error: {error}")

        if st.session_state.get("generated_visual_brief"):
            with st.expander("View generated visual brief"):
                st.write(st.session_state.generated_visual_brief)

    if st.session_state.get("generated_image"):
        st.image(
            st.session_state.generated_image,
            caption=st.session_state.get("generated_image_caption", "Generated image"),
        )
        st.download_button(
            "Download image",
            data=st.session_state.generated_image,
            file_name="papertrail-pdf-image.png",
            mime=st.session_state.get("generated_image_type", "image/png"),
        )

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything about the uploaded PDFs")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the document library..."):
                retrieved = retrieve(
                    question, chunks, vectorizer, matrix
                )
                try:
                    answer = answer_question(client, question, retrieved, chat_model)
                except Exception as error:
                    st.error(
                        "Groq could not answer this request. Check your API key, "
                        "model ID, and account rate limits."
                    )
                    st.caption(f"Groq error: {error}")
                    answer = "I could not get an answer from Groq."
            st.markdown(answer)
            with st.expander("Retrieved sources"):
                for index, (chunk, score) in enumerate(retrieved, start=1):
                    st.markdown(f"**[{index}] {chunk.source}, page {chunk.page}** · similarity {score:.3f}")
                    st.caption(chunk.text)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
