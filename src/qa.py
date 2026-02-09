from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class LocalLLMQA:
    def __init__(
        self,
        model_name="qwen2.5:1.5b",
        base_url="http://localhost:11434",
        temperature=0.0,
    ):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )

    def answer(self, question, contexts):
        if question is None or question == "":
            raise ValueError("question must be a non-empty string.")
        if contexts is None or len(contexts) == 0:
            raise ValueError("contexts must be a non-empty list of strings.")
        context_block = "\n\n".join(
            [f"[{i+1}] {c}" for i, c in enumerate(contexts)]
        )
        
        system_msg = (
            "You are a helpful assistant that ONLY answers questions based on the provided context. "
            "You MUST NOT use any external knowledge or information outside of the context. "
            "If the answer is not explicitly in the context, you MUST respond with 'I don't know'. "
            "Always respond in English only. "
            "Do not include any sources or citations. "
            "Be concise and to the point."
        )
        
        user_msg = f"Context:\n{context_block}\n\nQuestion: {question}"
        
        try:
            result = self.llm.invoke(
                [SystemMessage(content=system_msg), HumanMessage(content=user_msg)]
            )
        except Exception as exc:
            raise RuntimeError("LLM invocation failed. Check model availability and base_url.") from exc
        
        text = result.content.strip() if hasattr(result, "content") else str(result).strip()
        return text
