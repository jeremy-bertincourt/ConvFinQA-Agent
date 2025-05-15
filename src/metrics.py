from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric
from textstat import flesch_reading_ease
from typing import List, Dict, Any, Optional
from src.prompts import QAResponse
import re
import numpy as np

# --- Evaluate metrics over a dataset ---
class MetricEvaluator:
    def __init__(self):
        self.answer_correct = []
        self.program_op_correct = []
        self.program_args_correct = []
        self.explanation_list = []
        self.context_list = []

    @staticmethod
    def convert_program(prog: str) -> List[Dict[str,Any]]:
        ops = re.findall(r'(\w+)\(([^)]+)\)', prog or "")
        return [{"operation":op, "args":[a.strip() for a in args.split(",")]} for op, args in ops]

    def update(self, label: Dict[str,Any], pred: QAResponse, context: str):
        # Answer accuracy
        label_ans = np.round(float(label["qa"]["exe_ans"]),1)
        pred.answer = np.round(pred.answer,1)
        self.answer_correct.append(int(label_ans == pred.answer))
        print("answer:", self.answer_correct)

        # Program ops
        lab_prog = self.convert_program(label["qa"]["program"])
        pr_prog  = self.convert_program(pred.program)

        ops_match = all(lp["operation"]==pp["operation"] for lp,pp in zip(lab_prog,pr_prog))
        self.program_op_correct.append(int(ops_match))
        print("ops:", self.program_op_correct)
        
        # Program args
        args_match = all(
            all(
                (la.startswith("#") or la == pa)
                for la, pa in zip(lp["args"], pp["args"])
            )
            for lp, pp in zip(lab_prog, pr_prog)
        )
        self.program_args_correct.append(int(args_match))
        print("args:", self.program_args_correct)

        self.explanation_list.append(pred.explanation)
        self.context_list.append(context)


    def report(self):
        print("=== Metrics ===")
        print("Answer Acc   :", np.mean(self.answer_correct))
        print("Prog Ops Acc :", np.mean(self.program_op_correct))
        print("Prog Args Acc:", np.mean(self.program_args_correct))

        test_cases = [
            LLMTestCase(
                input="",
                actual_output=expl,
                context=[ctx],
            )
            for ctx, expl in zip(self.context_list, self.explanation_list)
        ]

        halluc_metric = HallucinationMetric(threshold=0.5)

        results = evaluate(
            test_cases,
            metrics=[halluc_metric],
        )

        # Compute Flesch Reading Ease
        read_scores = [flesch_reading_ease(expl) for expl in self.explanation_list]
        avg_read = sum(read_scores) / len(read_scores)
        print(f"Avg readability.flesch: {avg_read:.2f}")