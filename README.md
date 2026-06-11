# Think-again-llm
An experiment conducted to investigate how deliberate self-interruption in llms effects accuracy on resoning across 3 benchmarks and 5 conditions. 
## Note on ARC answer extraction

ARC-Challenge answers are free-text, so the model's chosen option (A–D)
must be parsed from its response. Initial regex-based extraction proved
unreliable: it failed on markdown formatting (e.g. `**D**`) and on
responses where the model self-corrected mid-reasoning (e.g. "actually,
the answer is D"), causing it to capture an earlier, non-final letter.
This biased results against conditions that induce self-correction.

Final ARC answers were therefore extracted using a model-based pass
(see `rescore_arc_model.py`), which reads each response and returns the
final committed answer. Results were spot-checked by hand against raw
responses. StrategyQA (yes/no) and GSM8K (numeric) were unaffected and
used direct parsing.