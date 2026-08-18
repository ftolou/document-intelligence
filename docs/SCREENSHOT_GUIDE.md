# Screenshot guide

Use a small number of anonymized screenshots that explain the product workflow rather than every
developer surface. Store the final images in `docs/screenshots/`.

## Recommended portfolio set

1. **Process / extraction result**  
   Show an uploaded receipt with the extracted merchant, date, items, total and validation state.
   Keep Advanced settings and Technical details collapsed.

2. **Human review**  
   Show the review queue beside the selected receipt, validation context and editable structured
   fields. This screenshot should make the human trust boundary obvious.

3. **Ask Your Receipts**  
   Show a natural-language question with an evidence-grounded answer based on approved data.

4. **Observability** *(optional)*  
   Show model operations, token usage, latency and estimated cost when that strengthens the
   technical story.

## Capture rules

- Use synthetic or anonymized receipt data.
- Do not expose API keys, local usernames, absolute host paths or customer/personal information.
- Prefer one complete application window per screenshot.
- Avoid screenshots of Docker commands, raw CI logs or large JSON payloads in the main README;
  those are implementation evidence rather than the product story.
