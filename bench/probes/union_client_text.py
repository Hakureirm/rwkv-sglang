#!/usr/bin/env python3
"""The F0080 union probe, on real text instead of random token ids.

F0080 measured a ~16% always-dead channel core across 64 random-id prompts and explicitly
did NOT claim it as a pruning candidate, because random ids drive the model off-distribution
and the activations may not resemble anything it sees in service. This sends 64 genuinely
different pieces of natural text -- prose, code, dialogue, Chinese, math, structured data --
so the union is measured on the distribution that actually matters.

Two outcomes, both useful: a union that survives on real text is a static-pruning candidate
worth a real evaluation; a union that collapses closes the question and the sparse kernel
stays a bsz1 feature for good.
"""
import concurrent.futures as cf, sys
import requests

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/generate"

TEXTS = [
    "The Eiffel Tower stands in the seventh arrondissement of Paris and was completed in 1889.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]",
    "User: Explain why the sky appears blue during the day.\n\nAssistant: Sunlight scatters",
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。这首诗出自唐代诗人李白",
    "SELECT customer_id, SUM(total) FROM orders WHERE created_at > '2024-01-01' GROUP BY",
    "In thermodynamics, the second law states that the entropy of an isolated system",
    "机器学习中的过拟合是指模型在训练集上表现很好，但在测试集上表现较差的现象。",
    "The patient presented with a three-day history of fever, productive cough and pleuritic",
    "import numpy as np\nimport torch\n\nclass Attention(torch.nn.Module):\n    def __init__(self",
    "Q: If a train travels 120 km in 1.5 hours and then 80 km in 0.5 hours, what is its average",
    "Dear Sir or Madam,\n\nI am writing to enquire about the position advertised in Monday's",
    "第一条 为了保护消费者的合法权益，维护社会经济秩序，促进社会主义市场经济健康发展",
    "Once upon a time in a village at the foot of a mountain there lived a blacksmith who",
    "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells",
    "git rebase -i HEAD~5\n# pick the first commit, squash the rest, then force-push with",
    "Ich habe gestern ein sehr interessantes Buch über die Geschichte der Mathematik gelesen",
]
PROMPTS = [t + f"  [{i}]" for i, t in enumerate(TEXTS * 4)][:64]


def one(i):
    r = requests.post(url, json={"text": PROMPTS[i], "sampling_params":
                      {"temperature": 0.0, "max_new_tokens": 8}}, timeout=300)
    r.raise_for_status()
    return 1


with cf.ThreadPoolExecutor(max_workers=64) as ex:
    print("completed", sum(f.result() for f in [ex.submit(one, i) for i in range(len(PROMPTS))]))
