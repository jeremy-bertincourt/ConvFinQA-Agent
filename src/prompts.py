from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError, Field


prompt_template_extraction = PromptTemplate.from_template("""
You are a financial assistant.
What are the values needed in order to answer the question? 
Do not answer the question. Return the values and what they refer to, no the text/table reference.
The output must contain the right numerical values.

Output example:
- Net Sales 2000: $7983
- Net Sales 2001: $5363

Context:
{text_block}

Table:
{table_block}

Question:
{question}
""")


prompt_template_reasoning = PromptTemplate.from_template("""
You are a financial assistant and need to answer the question using the provided tools.
You have critical thinking: make sure each observation makes sense. Does the position for each argument seem correct?

As a final answer, your response must be valid JSON with exactly all the following fields:
- answer (float): the final numeric result
- explanation (string): step-by-step reasoning chain
- program (string): Tool calls sequence, e.g. 'subtract(...), divide(...)'

You do not need to multiply by 100 to express the answer as the percentage.
You need to return all these three fields in the JSON output.

Remember:
  • arg1 is the **new** measurement  
  • arg2 is the **old** measurement

Once the final value is obtained, trigger Final Answer.

Example:
Thought: We need the difference between 206588 (new) and 181001 (old).
Action: subtract
Action Input: 206588, 181001
Observation: 25587
Thought: We then divide by the old value to get growth rate.
Action: divide
Action Input: 25587, 181001
Observation: 0.14136
Final Answer:
```json
{{
"answer": 0.14136,
"explanation": "Thought: I need to find the right answer, etc",
"program": "subtract(206588, 181001), divide(#0, 181001)"
}}
```

Context:
{context}

Question:
{question}
""")


# --- Pydantic schema for final answers ---
class QAResponse(BaseModel):
    answer: float = Field(..., description="Final numeric answer")
    explanation: str = Field(..., description="Chain-of-thought explanation")
    program: str = Field(..., description="Tool calls sequence, e.g. 'subtract(...), divide(...)'")
    model_config = {"extra": "ignore"}

# --- Manage prompt templates ---
class PromptManager:
    def __init__(self):
        self.prompt_template_extraction = prompt_template_extraction
        self.prompt_template_reasoning = prompt_template_reasoning
        self.parser = PydanticOutputParser(pydantic_object=QAResponse)

    def build_extract_prompt(self, context: Dict[str,str], question: str) -> str:
        return self.prompt_template_extraction.format(
            text_block=context["text_block"],
            table_block=context["table_block"],
            question=question,
        )

    def build_answer_prompt(self, context_str: str, question: str) -> str:
        return self.prompt_template_reasoning.format(
            context=context_str,
            question=question,
        )

    def parse_answer(self, raw: str) -> Optional[QAResponse]:
        try:
            return self.parser.parse(raw)
        except ValidationError:
            # fallback: extract JSON substring
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if match:
                return self.parser.parse(match.group(0))
        return None

