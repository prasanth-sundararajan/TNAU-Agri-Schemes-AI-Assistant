# Multimodel Agricultural Scheme Assistant

This project is a multi-model AI application built to explore agricultural government schemes from the TNAU (Tamil Nadu Agricultural University) website. It crawls scheme pages, extracts text from PDFs, looks for related links, and uses AI to understand scheme content and image-based information.

The app is designed to help a user ask questions like:
- Which agriculture schemes are available?
- What are the eligibility conditions?
- What support or subsidy is offered?
- Which documents mention a particular scheme or benefit?

## What this project does

The application combines several steps:

- Scrapes agricultural scheme pages from the TNAU website
- Extracts text from HTML pages and PDF documents
- Finds related URLs and scheme metadata
- Uses OCR-style image analysis for PDFs with diagrams or screenshots
- Sends extracted content to a language model for answer generation
- Stores the processed content in a local vector database for retrieval
- Provides a simple user interface using Streamlit

## Main file

- `agri_app.py` — main application logic

## Key libraries used

The project depends on:

- `streamlit`
- `pymupdf`
- `requests`
- `beautifulsoup4`
- `langchain-core`
- `langchain-openai`
- `langchain-community`
- `langchain-chroma`
- `langchain-text-splitters`
- `pillow`
- `pytesseract`
- `python-dotenv`

See `requirements.txt` for the project dependency list.

## Setup

1. Open PowerShell in the project folder:
   ```powershell
   cd C:\Users\HP\GenAI_Program\multimodel
   ```

2. Create and activate a virtual environment if needed:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install the Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR on your machine.
   - For Windows, install Tesseract from the official installer and make sure it is added to PATH.
   - The Python package alone is not enough.

5. Create a `.env` file in this folder with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   LANGCHAIN_API_KEY=your_langchain_key_if_used
   LANGCHAIN_PROJECT=tnau-agri-schemes-assistant
   ```

## Run the app

From the project folder:

```powershell
streamlit run agri_app.py
```

Then open the local Streamlit URL shown in the terminal.

## Important notes

- This project sends information to external AI services, so your API key must be valid.
- The app uses PDF processing and OCR, so Tesseract must be installed correctly.
- Local vector database folders such as `chroma_tnau_agri_db` may be created in this project when the app runs.
- The project is a learning/demo application, not a production-ready government service.

## Task completed

This project demonstrates:
- web scraping from a local government/agriculture portal,
- multimodal document processing,
- retrieval-based AI answer generation,
- and a simple Streamlit interface for exploring agricultural scheme content.

## Beginner challenge question

How would you modify this app so that it works on a different agriculture website or supports Tamil-language scheme search as well as English?

## Notes for future improvement

Possible improvements:
- add better filtering for scheme titles,
- improve Hindi/Tamil text extraction,
- support more sources and document types,
- add a search box for specific scheme names,
- add logging and export options for results.
