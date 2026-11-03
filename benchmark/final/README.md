# Final dataset format

Fixed on 2026-11-03 at the user-requested audit pause. Use train.json, dev.json and test.json separately. All retained question, answer, entity, evidence and temporal fields exactly match the working source. Generation debug fields and stale review flags are omitted; absent optional fields are null. No semantic correction is applied during finalization.

- 1,375 conversations / 10,198 turns; conversation-level 6:2:2 split.
- Audit incomplete: 489 requested training turns and 772 test turns remain unadjudicated. Fixed delivery does not certify correctness.
- Dates are YYYY-MM-DD; year envelopes may encode precision rather than duration. signal compares main fact with constraint; temporal_link compares main fact with preceding main fact.
- connection_type and temporal_reference are retained nullable metadata, not freshly certified. Known reversed intervals train523-3/523-5 anddev971-1 are unchanged under the requested pause.
- Context-removed evaluation uses question, not rephrased_question.
Only this final dataset is retained. Prior snapshots, model responses, results and audit files were deleted at user request. Question repair remains paused.
