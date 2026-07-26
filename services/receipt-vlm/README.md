# Receipt VLM service

This directory is an independently deployable Python 3.10 service. It exposes
an HTTP boundary and invokes the installed `paddleocr doc_parser` CLI. It must
not import `receipt_intelligence` or any application, LLM, RAG, database, or
query-observability module.

The main application communicates with it only through `POST /api/vlm/analyze`
and `GET /health`. Both containers share the configured `var/` volume, so the
request normally contains an allowed image path rather than image bytes.
