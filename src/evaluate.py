import numpy as np
import nest_asyncio
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, faithfulness
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


nest_asyncio.apply()

def cosine_similarity(a, b):
    # Guard against zero vectors to avoid division by zero.
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("cosine_similarity received a zero vector; cannot compute similarity.")
    return float(np.dot(a, b) / (norm_a * norm_b))

def evaluate_cosine_similarity(predictions, ground_truths, model_name="intfloat/e5-base"):
    if predictions is None or ground_truths is None:
        raise ValueError("predictions and ground_truths must be provided.")
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must be the same length.")
    # Load the embedding model once for all pairs.
    model = SentenceTransformer(model_name)
    scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        emb_pred = model.encode(pred, convert_to_numpy=True)
        emb_gt = model.encode(gt, convert_to_numpy=True)
        scores.append(cosine_similarity(emb_pred, emb_gt))
        
    return scores


def evaluate_ragas(questions, answers, contexts, llm, ground_truths=None, embeddings=None):
    if llm is None:
        raise ValueError("llm must be provided for RAGAS evaluation.")
    if questions is None or answers is None or contexts is None:
        raise ValueError("questions, answers, and contexts must be provided.")
    # Keep arrays aligned for evaluation.
    min_len = min(len(questions), len(answers), len(contexts))
    if ground_truths:
        min_len = min(min_len, len(ground_truths))
    if min_len == 0:
        raise ValueError("questions, answers, and contexts must be non-empty.")
    
    questions = questions[:min_len]
    answers = answers[:min_len]
    contexts = contexts[:min_len]
    if ground_truths:
        ground_truths = ground_truths[:min_len]

    if contexts and isinstance(contexts[0], str):
        contexts = [[ctx] for ctx in contexts]

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths
        
    dataset = Dataset.from_dict(data)

    eval_llm = LangchainLLMWrapper(llm)
    
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base")
    eval_embeddings = LangchainEmbeddingsWrapper(embeddings)

    run_config = RunConfig(
        timeout=360,
        max_retries=3,
        max_workers=1
    )

    results = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config,
        raise_exceptions=False
    )
    
    return results
