# Final dataset format

- 1,375 conversations / 10,198 turns; conversation-level 6:2:2 split.
- Audit incomplete: 489 requested training turns and 772 test turns remain unadjudicated. Fixed delivery does not certify correctness.
- Dates are YYYY-MM-DD; year envelopes may encode precision rather than duration. signal compares main fact with constraint; temporal_link compares main fact with preceding main fact.
- connection_type and temporal_reference are retained nullable metadata, not freshly certified. Known reversed intervals train523-3/523-5 anddev971-1 are unchanged pending further review.
- Context-removed evaluation uses question, not rephrased_question.
Only this final dataset is retained. Prior snapshots, model responses, results and audit files were deleted at user request. Question repair resumed at user request on2026-11-03; new field-level changes are inrepair_log.jsonl.