# Evaluation code

Historical experiment outputs and older datasets were removed. Scripts default to benchmark/final/train.json and benchmark/final/test.json; use dev.json for development.

- test.py: original question with preceding questions as context.
- nocontext_test.py: standalone rephrased_question.
- run_bailian.py --context_removed: original question without preceding test questions.
- run_bailian.py --thinking --cot_guidance --thinking_budget 2048 --max_completion_tokens 8192: combined reasoning condition.

Set DASHSCOPE_API_KEY in the environment before invoking run_bailian.py. Future runs write new output_dir results; deleted experiments cannot be resumed.

IoU uses inclusive days; accuracy threshold0.5. Entity scoring uses first-gold-label matching; relation scoring excludes structural first turns and missing labels. Formatting errors remain scored errors.

Offline checks: python -B -m unittest discover -s test_scripts -p test_expanded_release.py. These validate structure and metrics, not historical truth. Three known reversed intervals remain in train/dev; repairs are paused.
