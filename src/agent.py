from typing import List, Dict, Any

from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

def load_t5():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model     = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    text2text = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.0,
    )

    llm = HuggingFacePipeline(pipeline=text2text)

    return llm

def load_gemma3():
    llm = ChatOllama(model="gemma3:27b", base_url="http://localhost:11434")
    return llm

def load_mistral():
    llm = ChatOllama(model="mistral:instruct", base_url="http://localhost:11434")
    return llm

class AgentFactory:
    """Creates a LangChain agent with ChatOllama, memory, and tools."""
    def __init__(self, model: str = "t5"):
        self.model = model

    def create(self, tools: List[Tool]) -> Any:
        if self.model == "t5":
            llm = load_t5()
        elif self.model == "mistral":
            llm = load_mistral()
        else:
            llm = load_gemma3()

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        agent_extract = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            temperature=0,
        )

        agent_reasoning = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            temperature=0,
            return_intermediate_steps=True,
        )

        return agent_extract, agent_reasoning

