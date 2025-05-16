# ConvFinQA LLM Agentic Solution

This repository implements an agent-based pipeline to answer Finance-related questions given a specific context. The dataset used is the ConvFinQA dataset.

## Iterative process
The assignement was carried out using an iterative process - i.e creating a baseline and then iterating in order to increase the performance results.

---

## Baseline

### Features

- **Data Loading & Context Building**  
  - Flattens `pre_text` and `post_text` lists into indexed text blocks.  
  - Flattens each table column into a separate indexed block.  

- **2 Prompts under 1 Agent**  
  - The model seemed to struggle with the full context. Therefore it was separated into 2 parts: one prompt extracting the right tables/texts, one prompt answering the question.
  - For the baseline, both prompts are done in a single pass but the second one includes chain-of-thought reasoning via examples.

- **Model** 
  - All models used are local and open-source with Ollama.
  - Basic SeqtoSeq model tried but no possibility to get the right output.
  - Mistral model can sometimes give the right output but not consistent.
  - Gemma3:27b was found to work well. It is also the right balance between memory needed and performance.

- **Structured Output**  
  - `QAResponse` Pydantic model with fields:  
    ```yaml
    answer: float            # Final numeric result  
    explanation: string      # Full chain-of-thought  
    program: string          # Comma-separated tool calls, e.g. “subtract(…); divide(…)”
    ```
  - PydanticOutputParser to enforce JSON schema.

- **Evaluation Metrics**  
  - Is the agent trying to answer the question? -> Hallucination rate (based on `explanation`) using DeepEval Framework. 
  - Is the agent using the right operations? -> Operation-sequence accuracy on `program`.  
  - Is the agent using the right arguments? -> Argument value accuracy.  
  - Did the agent find the right answer? -> Exact-match accuracy on `exe_ans` (rounded up).   
  - Does the explanation sound natural? -> Readability (Flesch Reading Ease).

### Results
  - Answer Acc   : 0.375
  - Prog Ops Acc : 0.875
  - Prog Args Acc: 0.5
  - Hallucination test: 100% pass
  - Avg readability.flesch: 76.83

---

## V2

#### Improvements

The baseline did not include muti-hop reasoning with function calls. Instead, it was generating the answer in a single pass. This improved version includes full muti-hop reasoning, function calling and chain-of-thought.

- **2 Agents**  
  - 1 agent extracting the values needed in a single pass and 1 agent doing the reasoning with muti-hop and CoT reasoning.
  - Four arithmetic tools: `subtract`, `add`, `multiply`, `divide`.  
  - Tools called by agent are printed to show which ones were used.
  - Pydantic‐backed input validation for each tool.  
  - LangChain Agent with ReAct loop (`ZERO_SHOT_REACT_DESCRIPTION`) for multi-step reasoning.

### Results
  - Answer Acc   : 0.5
  - Prog Ops Acc : 0.875
  - Prog Args Acc: 0.5
  - Hallucination test: 100% pass
  - Avg readability.flesch: 68.41

---

## V3

#### Improvements

The V2 version was often swapping arguments leading to a wrong answer.

- **Better prompting**  
  - More examples in prompt.
  - Inserting a note in the prompt to remember that arg1 is for the new value and arg2 the old value. A good pydantic description is not enough.
  - Additional notes about the JSON output.
- **Different fields**
  - Using `exe_ans` instead of `answer` helped as the former ground truth was inconsistently rounded up. 
  - The reference and prediction were both rounded up to one decimal to avoid issues with varying decimal numbers.

### Results
  - Answer Acc   : 0.75
  - Prog Ops Acc : 0.875
  - Prog Args Acc: 0.5
  - Hallucination test: 100% pass
  - Avg readability.flesch: 62.02

---

## Installation

Before running this Agentic module, make sure the following is carried out:

Clone this repo:
```bash
git clone https://github.com/jeremy-bertincourt/ConvFinQA-Agent
cd ConvFinQA-Agent
```

Install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the LLM model and configure DeepEVal.
```bash
ollama pull gemma3:27b
deepeval set-local-model --model-name=gemma3:27b --base-url="http://localhost:11434" --api-key="ollama"
```

Add the `train.json` file under a folder called `data` at root level - next to `main.py`.

## Running the pipeline

```bash
python main.py
```

## Further Improvements

- Evaluation should allow commutative arguments for Add or multiply
- Table/text indices should be evaluated in order to get retrieval performance
- Perform eval on a higher number of examples -e.g at least 100 examples
- Take into account sub-QAs
- Add more tools for calculations, i.e specific calculations such as compound interest, etc
- The model still makes argument swapping mistakes - use openAI to include 2 arguments instead of one string in each function for strict JSON formatting
- Test with other models - a fined tuned version with LoRa on this dataset could increase the performance
- Enhance metrics: better program argument processing (for now they are just seen as strings), F1 score for answer-match and operation-match. Bleu/Rouge scores could also be added as metrics for program, although it would need more processing beforehand (removal of multiplication by 100, cleaning after CoT makes function call mistakes)
- Solve cases where CoT keeps making the same function call multiple times
- Improve monitoring with more logging
- Add explanation faithfulness as a metric
- Improve scalability - use async functions across QAs to parallelise questions and agent calls (more memory might be needed)
- The json parsing can still fail for now, add more robust parser 

## Additional Notes

- Ollama doesn't support function calling with separate arguments like OpenAI does, so the argument needs to be a string which then needs to be processed.
