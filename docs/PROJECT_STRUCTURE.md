# Project structure

```text
document-intelligence-pipeline/
├── src/receipt_intelligence/
│   ├── extraction/
│   │   ├── transcription/  Paddle geometry, crop planning, Qwen transcription
│   │   ├── structured/     Gemma scalar and item extraction
│   │   ├── validation/     read-only deterministic rules
│   │   ├── correction/     source-evidence specialist correction
│   │   ├── presentation/   categorization and stable publication
│   │   └── stages/         canonical workflow stages
│   ├── pipeline/           public extraction entry points
│   ├── application/        use cases and provider-neutral ports
│   ├── adapters/           Ollama, Paddle, jobs, storage, observability
│   ├── services/           reusable application workflows
│   ├── storage/            SQLite repositories and migrations
│   ├── rag/ and rag_sql/   reviewed-item retrieval and safe analytics
│   └── web/                Flask transport and presenters
├── tests/
├── scripts/
├── docker/
├── static/
├── docs/
└── var/                    generated runtime state
```

The only supported generated-data root is `var/`.

## Compatibility contracts

Receipt schema identifiers and historical artifact filenames may still contain release-oriented
labels such as `v14`. These are persisted data contracts used by saved jobs, review, and regression
fixtures. They do not represent an executable legacy extraction path.
