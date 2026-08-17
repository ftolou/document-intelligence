# Prompt files

| Prompt | Caller | Purpose |
|---|---|---|
| `rag_sql_question_analyzer.txt` | `rag_sql/question_analyzer.py` | Interpret the user goal and identify product entities requiring semantic resolution. |
| `rag_sql_planner.txt` | `rag_sql/planner.py` | Produce a typed read-only SQL plan against the curated analytics catalog. |
| `rag_sql_answer_formatter.txt` | `rag_sql/answer_formatter.py` | Normalize ambiguous reviewed evidence into typed values and supporting item IDs; output is deterministically validated. |
| `item_categorization.txt` | categorization module | Categorize reviewed receipt items. |
| Versioned Qwen/Gemma prompts | extraction modules | Image transcription, structured extraction, and bounded correction. |

Query prompts produce structured contracts only. They do not execute SQL or
perform financial calculations. The answer formatter does not produce final
free-form prose; validated values are rendered by application code.
