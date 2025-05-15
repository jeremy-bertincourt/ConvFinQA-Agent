import json
from pathlib import Path
from typing import List, Dict, Any

# --- Load data ---
class ConvFinQALoader:
    """Load and flatten ConvFinQA JSON into Documents."""
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> List[Dict[str, Any]]:
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)


# --- Build context blocks ---
class ContextBuilder:
    @staticmethod
    def build(entry: Dict[str, Any]) -> Dict[str, str]:
        # Flatten text
        all_texts = entry.get("pre_text", []) + entry.get("post_text", [])
        indexed_texts = [f"[text_{i}]: {t}" for i, t in enumerate(all_texts)]
        text_block = "\n".join(indexed_texts)

        # Flatten tables
        table_strs = [f"{col[0]}: {', '.join(map(str,col[1:]))}" for col in entry.get("table", [])]
        indexed_tables = [f"[table_{i}]: {tbl}" for i, tbl in enumerate(table_strs)]
        table_block = "\n".join(indexed_tables)

        return {"text_block": text_block, "table_block": table_block}