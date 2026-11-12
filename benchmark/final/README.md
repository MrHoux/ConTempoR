# Final dataset format

- 1,375 conversations / 10,198 turns; conversation-level 6:2:2 split.
- Dates are YYYY-MM-DD; year envelopes may encode precision rather than duration.
- `signal` compares the main fact with the constraint; `temporal_link` compares the main fact with the preceding main fact.
- `connection_type` and `temporal_reference` are nullable metadata.
- Context-removed evaluation uses `question`, not `rephrased_question`.
