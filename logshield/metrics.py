"""Exact character-span metrics, keeping every full gold and every log row."""
from collections import Counter

from .io import span_set


def scores(tp, predicted, gold):
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    return {"tp": tp, "predicted": predicted, "gold": gold,
            "fp": predicted - tp, "fn": gold - tp,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (predicted + gold) if predicted + gold else 0.0}


class SpanMetrics:
    def __init__(self, categories):
        self.categories = list(categories)
        self.counts = {category: Counter() for category in categories}
        self.logs = Counter()

    def add(self, text, gold, predictions):
        gold = span_set(text, gold, self.categories)
        predicted = span_set(text, predictions, self.categories)
        common = gold & predicted
        for name, spans in (("gold", gold), ("predicted", predicted), ("tp", common)):
            for _, _, category in spans:
                self.counts[category][name] += 1
        self.logs.update({"rows": 1, "exact_span_set": int(gold == predicted),
                          "positive_gold": int(bool(gold)), "positive_predicted": int(bool(predicted)),
                          "binary_tp": int(bool(gold) and bool(predicted)),
                          "binary_tn": int(not gold and not predicted)})

    def merge(self, other):
        if self.categories != other.categories:
            raise ValueError("Cannot merge different taxonomies")
        self.logs.update(other.logs)
        for category in self.categories:
            self.counts[category].update(other.counts[category])

    def to_dict(self):
        per_category = {category: scores(c["tp"], c["predicted"], c["gold"])
                        for category, c in self.counts.items()}
        totals = sum(self.counts.values(), Counter())
        rows = self.logs["rows"]
        supported = [x for x in per_category.values() if x["gold"]]
        return {"rows": rows,
                "span_micro": scores(totals["tp"], totals["predicted"], totals["gold"]),
                "span_macro_f1_18": sum(x["f1"] for x in per_category.values()) / len(self.categories),
                "span_macro_f1_gold_present": sum(x["f1"] for x in supported) / len(supported) if supported else 0.0,
                "exact_span_set_accuracy": self.logs["exact_span_set"] / rows if rows else 0.0,
                "log_binary": {**scores(self.logs["binary_tp"], self.logs["positive_predicted"], self.logs["positive_gold"]),
                               "tn": self.logs["binary_tn"],
                               "accuracy": (self.logs["binary_tp"] + self.logs["binary_tn"]) / rows if rows else 0.0},
                "per_category": per_category}
