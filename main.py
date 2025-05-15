from pathlib import Path
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, ValidationError, Field

from src.data_loader import ConvFinQALoader, ContextBuilder
from src.agent import AgentFactory
from src.tools import SubtractToolBuilder, AddToolBuilder, MultiplyToolBuilder, DivideToolBuilder
from src.prompts import prompt_template_reasoning, prompt_template_extraction, PromptManager, QAResponse
from src.metrics import MetricEvaluator


# --- Execute QA with an agent ---
class QAExecutor:
    def __init__(self, agent_extract, agent_answer):
        self.agent_extract = agent_extract
        self.agent_answer = agent_answer
        self.prompts = PromptManager()

    def extract_context(self, context: Dict[str,str], question: str) -> str:
        prompt = self.prompts.build_extract_prompt(context, question)
        result = self.agent_extract.invoke({"input":prompt})
        return result["output"]

    def answer_question(self, res_str: str, question: str) -> Optional[QAResponse]:
        prompt = self.prompts.build_answer_prompt(res_str, question)
        result = self.agent_answer.invoke({"input":prompt})
        print("\nTool calls:")
        for action, obs in result["intermediate_steps"]:
            print(f" • {action.tool}({action.tool_input}) → {obs}")
        return self.prompts.parse_answer(result["output"])


# --- Pipeline to run everything ---
class QAPipeline:
    def __init__(self, model_name="gemma3:27b", base_url="http://localhost:11434"):
        # load data
        self.loader = ConvFinQALoader(Path("data/train.json"))
        self.data = self.loader.load()

        # build tools & agent
        tools = [
            SubtractToolBuilder.build_tool(),
            AddToolBuilder.build_tool(),
            MultiplyToolBuilder.build_tool(),
            DivideToolBuilder.build_tool(),
        ]
        self.agent_extract, self.agent_answer = AgentFactory(model_name).create(tools)
        self.executor = QAExecutor(self.agent_extract, self.agent_answer)
        self.evaluator = MetricEvaluator()

    def run(self, limit: Optional[int]=None):
        for idx, entry in enumerate(self.data):
            if limit and idx>=limit:
                break
            if "qa" not in entry:
                continue
            question = entry["qa"]["question"]
            context = ContextBuilder.build(entry)

            # Extract
            ctx_str = self.executor.extract_context(context, question)
            # Answer
            pred = self.executor.answer_question(ctx_str, question)
            if pred:
                ctx = "Question:"+question+"\n"+"context retrieved:"+ctx_str
                self.evaluator.update(entry, pred, ctx)
        # Report
        self.evaluator.report()


# --- Main Execution ---
if __name__ == "__main__":
    pipeline = QAPipeline(model_name="gemma3")
    pipeline.run(limit=10)  # process first 10 examples
