# Youtube InsightRAG

A small Python RAG pipeline that extracts playlist audio with yt-dlp, transcribes it locally with faster-whisper, groups timestamped speech into time-based chunks, embeds them with FastEmbed, stores them in Qdrant, and answers study questions with Groq. Answers include clickable links to the relevant video timestamps.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- yt-dlp and faster-whisper for local audio extraction and transcription
- A [Groq](https://console.groq.com/keys) API key
- Qdrant Cloud credentials, or Docker for a local Qdrant instance

Only transcribe videos you own or are otherwise authorized to process, and comply with YouTube's terms and applicable copyright rules.

## Install

```powershell
uv sync
Copy-Item .env.example .env
```

Edit `.env` and set `GROQ_API_KEY`. For Qdrant Cloud, set `QDRANT_URL` and `QDRANT_API_KEY`. The default URL in the example is for local Qdrant; start it with Docker:

```powershell
docker run --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

The first ingestion downloads the Whisper model selected by `WHISPER_MODEL` (default `base`). faster-whisper runs on CPU with int8 compute by default. The first ingestion or question also downloads the FastEmbed `sentence-transformers/all-MiniLM-L6-v2` model. Public model downloads do not require a Hugging Face token, but setting `HF_TOKEN` in `.env` avoids unauthenticated Hub rate limits. If a model download repeatedly fails with a connection reset on Windows or a restrictive network, set `HF_HUB_DISABLE_XET=1` in `.env` and restart the app to use the standard download path. FastEmbed retries failed downloads; a partial download may need to be fetched again.

## Ingest a playlist

Pass either a YouTube playlist URL or playlist ID:

```powershell
uv run python main.py ingest "https://www.youtube.com/playlist?list=PLxxxxxxxxxxxxxxxx"
```

yt-dlp reads the playlist and extracts each video's audio in sequence. faster-whisper generates timestamped speech segments locally; videos that cannot be downloaded or transcribed are reported and skipped. Chunks target 100 seconds (bounded by the configured 90–120-second range under normal segment lengths); a shorter final chunk is retained. Audio files are temporary and removed after transcription.

## Use the web UI

Start the Streamlit app:

```powershell
uv run python -m streamlit run app.py
```

Open the local URL printed in the terminal. Use **Add a playlist** to extract audio and index local Whisper transcriptions, then **Ask a question** to query all chunks in the configured Qdrant collection. The sidebar shows whether required environment variables are present without displaying their values.

## Ask a revision question

```powershell
uv run python main.py ask "How does the playlist explain gradient descent?"
```

The default retrieval count is four (configurable with `--top-k`, from 3 to 5). Each returned answer includes source links with video title and timestamp.

## Configuration

| Variable             | Required      | Description                                                                 |
| -------------------- | ------------- | --------------------------------------------------------------------------- |
| `GROQ_API_KEY`       | For questions | Groq API key                                                                |
| `QDRANT_URL`         | Yes           | Qdrant URL, e.g. `http://localhost:6333`                                    |
| `QDRANT_API_KEY`     | Cloud only    | Qdrant Cloud API key; leave blank for local Qdrant                          |
| `GROQ_MODEL`         | No            | Groq model; defaults to `openai/gpt-oss-120b`                               |
| `QDRANT_COLLECTION`  | No            | Collection name; defaults to `youtube_revision_chunks`                      |
| `WHISPER_MODEL`      | No            | faster-whisper model size; defaults to `base`                               |
| `HF_TOKEN`           | No            | Hugging Face token for higher Hub rate limits during model downloads        |
| `HF_HUB_DISABLE_XET` | No            | Set to `1` to use the standard model download path if Xet connections reset |

The code never drops an existing Qdrant collection. Use a new collection name if changing embedding models or vector dimensions.

## Run tests

```powershell
uv run pytest
```
