# Papertrail PDF Q&A Agent

A Streamlit app that answers questions from one or more uploaded subject-matter PDFs. It extracts text, searches locally with TF-IDF, and asks a Groq chat model to answer with page-based source citations.

## Run locally on Windows

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:GROQ_API_KEY = "your-groq-api-key"
streamlit run app.py
```

Or enter the key in the sidebar after starting the app. The included `ultimate-guide-to-digital-marketing.pdf` can be used as a first test.

## Push to GitHub

```powershell
git init
git add app.py requirements.txt README.md .gitignore .env.example .streamlit/config.toml
git commit -m "Build PDF question answering agent"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Do not commit `.env`, `.streamlit/secrets.toml`, or your Groq key.

## Deploy to Streamlit Community Cloud

1. Push this project to GitHub.
2. Create an app at [share.streamlit.io](https://share.streamlit.io/).
3. Select the repository, branch, and `app.py` as the main file.
4. In **Advanced settings**, add this secret:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

5. Deploy. Upload PDFs in the app; files are processed in the current session and are not committed to the repository.

## Image generation

The **Image studio** generates visuals grounded in your uploaded PDFs using open-model providers. Describe what you want, such as `Create an infographic explaining the main stages`, and the app retrieves relevant PDF passages before building the image prompt.

- **Pollinations:** no application key required; availability and rate limits are controlled by the provider.
- **Hugging Face FLUX.1-schnell:** add an `HF_TOKEN` with inference access for more predictable access.

Generated images are held in the current Streamlit session and can be downloaded. The image prompt is sent to the selected provider.

## Notes

- Text-based PDFs are supported. Scanned PDFs need OCR before upload.
- Each new PDF selection is re-indexed for the current session.
- PDF search runs locally. Only the retrieved excerpts and your question are sent to Groq.
