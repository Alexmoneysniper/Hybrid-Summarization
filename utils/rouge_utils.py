from rouge_score import rouge_scorer

def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = [scorer.score(r, p) for p, r in zip(preds, refs)]
    return results