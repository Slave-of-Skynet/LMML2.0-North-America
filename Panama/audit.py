"""Reproduce the CanalQA contamination audit using only the supplied bundle.

Usage: python audit.py /path/to/canalqa_audit.json --output outputs/report.json
No seal access, model predictions, or correctness labels are used to match facts.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata


QUESTION_PREFIXES = (
    "according to the canal registry what is the",
    "per the canal records which value is given as the",
    "which value is recorded as the", "which value is given as the",
    "what value is listed as the", "tell me the", "what is the", "state the", "report the",
    "segun el registro del canal cual es", "cual es", "indique",
)
PROPERTY_ALIASES = {
    "la instalacion": "facility", "el ano de finalizacion": "completion year",
    "el ano de inauguracion": "opening year", "el tramo del canal": "canal section",
    "la longitud": "length", "la manga maxima": "maximum beam",
    "el desnivel": "lift height", "la eslora maxima": "maximum length",
    "overall length": "length", "count of chambers": "number of chambers",
    "establishment year": "established year",
}
ANSWER_ALIASES = {
    "atlantic terminus": "atlantic terminal",
    "balboa district": "balboa township",
    "gates rolling": "rolling gates",
    "division de dredging": "dredging division",
    "division of dredging": "dredging division",
    "dredging division": "dredging division",
}
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
                "uno": "1", "una": "1", "dos": "2", "tres": "3", "cuatro": "4",
                "cinco": "5", "seis": "6", "siete": "7", "ocho": "8", "nueve": "9"}


def normalize(value):
    value = "".join(c for c in unicodedata.normalize("NFKD", value.casefold())
                    if not unicodedata.combining(c))
    value = re.sub(r"(?<=\d),(?=\d)", ".", value)
    return " ".join(re.sub(r"[^a-z0-9.]+", " ", value).split()).strip(" .")


def translate(value, glossary):
    value = normalize(value)
    words = {normalize(k): normalize(v) for k, v in glossary.items()}
    if words:
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in sorted(words, key=len, reverse=True)) + r")\b")
        value = pattern.sub(lambda m: words[m.group()], value)
    return value


def property_name(value, glossary):
    value = normalize(value)
    if value in PROPERTY_ALIASES:
        return PROPERTY_ALIASES[value]
    value = translate(value, glossary)
    return PROPERTY_ALIASES.get(value, value)


def answer_name(value, glossary):
    value = translate(value, glossary)
    value = re.sub(r"\bsq\s+km\b", "square kilometers", value)
    value = re.sub(r"\bkm\b", "kilometers", value)
    value = re.sub(r"\bm\b", "meters", value)
    value = re.sub(r"\b(kilometres|kilometer)\b", "kilometers", value)
    value = re.sub(r"\b(metres|meter)\b", "meters", value)
    value = re.sub(r"\b(?:" + "|".join(NUMBER_WORDS) + r")\b",
                   lambda m: NUMBER_WORDS[m.group()], value)
    return ANSWER_ALIASES.get(value, value)


def strip_prefix(question):
    question = normalize(question)
    for prefix in QUESTION_PREFIXES:
        if question.startswith(prefix + " "):
            return question[len(prefix):].strip()
    raise ValueError("Unrecognized question framing: " + question)


def benchmark_fact(item, glossary, entities=()):
    body = strip_prefix(item["question"])
    # Both relation and entity can contain 'of': number of chambers / Port of
    # Balboa / Bridge of the Americas. Prefer the longest known entity suffix.
    candidates = [entity for entity in entities if body.endswith(" of " + entity)]
    if candidates:
        entity = max(candidates, key=len)
        attribute = body[:-len(" of " + entity)]
    else:
        attribute, entity = body.rsplit(" of ", 1)
    return {"entity": normalize(entity), "attribute": property_name(attribute, glossary),
            "answer": answer_name(item["gold"], glossary)}


def corpus_fact(document, glossary):
    text = document["text"]
    if "Question on record:" in text:
        question, answer = re.search(r"Question on record: '(.*?)' Answer on record: '(.*?)'", text).groups()
        native = entity = re.search(r"Verified against the (.*?) registry entry", text).group(1)
        relation = "of"
        style = "english-verbatim"
    elif "entry asks:" in text:
        question, answer = re.search(r"entry asks: '(.*?)' The logged response reads '(.*?)'", text).groups()
        native = entity = re.search(r"' for (.*?)\.$", text).group(1)
        relation = "of"
        style = "english-paraphrase"
    elif "Respuesta registrada:" in text:
        question, answer = re.search(r"Registro del canal LOG-\d+\. (.*?) Respuesta registrada: '(.*?)'", text).groups()
        native, entity = re.search(r"Verificado en la entrada de (.*?) \((.*?)\)", text).groups()
        relation = "de"
        style = "spanish"
    else:
        return None
    body = strip_prefix(question)
    suffix = " " + relation + " " + normalize(native)
    if not body.endswith(suffix):
        raise ValueError("Unparsed fact suffix in " + document["doc_id"])
    attribute = body[:-len(suffix)].strip()
    return {"doc_id": document["doc_id"], "lang": document["lang"], "style": style,
            "entity": normalize(entity), "attribute": property_name(attribute, glossary),
            "answer": answer_name(answer, glossary), "raw_answer": answer}


def unique_index(records, field):
    indexed = {record[field]: record for record in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate " + field)
    return indexed


def audit(bundle):
    benchmark = unique_index(bundle["benchmark"], "id")
    predictions = unique_index(bundle["model_predictions"], "id")
    unique_index(bundle["corpus"], "doc_id")
    if set(benchmark) != set(predictions):
        raise ValueError("Prediction IDs must equal benchmark IDs")
    if any(type(prediction["correct"]) is not bool for prediction in predictions.values()):
        raise ValueError("correct labels must be booleans")
    glossary = bundle["glossary_es_en"]
    facts = [fact for document in bundle["corpus"]
             if (fact := corpus_fact(document, glossary)) is not None]
    entities = {fact["entity"] for fact in facts}
    decisions = []
    for item_id, item in benchmark.items():
        target = benchmark_fact(item, glossary, entities)
        same_entity = [fact for fact in facts if fact["entity"] == target["entity"]]
        supports = [fact for fact in same_entity
                    if fact["attribute"] == target["attribute"] and fact["answer"] == target["answer"]]
        decisions.append({"id": item_id, "exclude": bool(supports), **target,
                          "support": supports,
                          "same_entity_other_facts": [fact for fact in same_entity if fact not in supports]})
    excluded = sorted(decision["id"] for decision in decisions if decision["exclude"])
    excluded_set = set(excluded)
    retained = [item_id for item_id in benchmark if item_id not in excluded_set]
    correct = sum(predictions[item_id]["correct"] for item_id in retained)
    if not retained:
        raise ValueError("No uncontaminated items remain")
    total_correct = sum(prediction["correct"] for prediction in predictions.values())
    report = {"excluded_ids": excluded, "honest_accuracy": correct / len(retained)}
    summary = {"benchmark_items": len(benchmark), "training_documents": len(bundle["corpus"]),
               "qa_training_records": len(facts), "reported_correct": total_correct,
               "recomputed_reported_accuracy": total_correct / len(benchmark),
               "excluded_items": len(excluded),
               "excluded_correct": total_correct - correct,
               "retained_items": len(retained), "retained_correct": correct,
               "honest_accuracy": report["honest_accuracy"],
               "support_styles": dict(Counter(fact["style"] for decision in decisions
                                              for fact in decision["support"]))}
    if abs(summary["recomputed_reported_accuracy"] - bundle["reported_accuracy"]) > 1e-12:
        raise ValueError("Reported accuracy disagrees with supplied correctness labels")
    return report, summary, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/report.json"))
    parser.add_argument("--evidence", type=Path, default=Path("outputs/decisions.json"))
    args = parser.parse_args()
    report, summary, decisions = audit(json.loads(args.bundle.read_text(encoding="utf-8-sig")))
    for path, data in ((args.output, report), (args.evidence, {"summary": summary, "decisions": decisions})):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
