"""Pura Vida, No PII: offline, document-local PII redaction (Python 3.10+).

Default: bare test-input.json -> test-output.json in the current directory.
Training fallback: train-input.json -> train-output.json, without overwriting
an existing golden file. Pass explicit input/output paths for local practice.
"""

import argparse
from datetime import date
import json
from pathlib import Path
import re
import unicodedata


WORD = r"[^\W\d_]+(?:['’\-][^\W\d_]+)*"
MONTHS = (
    ("january", "enero", "jan"), ("february", "febrero", "feb"),
    ("march", "marzo", "mar"), ("april", "abril", "apr", "abr"),
    ("may", "mayo"), ("june", "junio", "jun"),
    ("july", "julio", "jul"), ("august", "agosto", "aug", "ago"),
    ("september", "septiembre", "setiembre", "sep", "sept"),
    ("october", "octubre", "oct"), ("november", "noviembre", "nov"),
    ("december", "diciembre", "dec", "dic"),
)
MONTH_NUMBER = {name: i for i, names in enumerate(MONTHS, 1) for name in names}
MONTH = "(?:" + "|".join(sorted(MONTH_NUMBER, key=len, reverse=True)) + ")"
DATE = (r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|"
        + MONTH + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,|\s)\s*\d{4}|"
        r"\d{1,2}\s+(?:de\s+)?" + MONTH + r"\.?(?:\s+de)?\s+\d{4})")
BOOKING = re.compile(r"\bPV-[A-Z0-9]{5}\b", re.I)
PLACEHOLDER = re.compile(r"\[(?:NAME|EMAIL|PHONE|ADDRESS|ID|DOB|ACCOUNT)\]")
EMAIL = re.compile(r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)
PHONE = re.compile(r"(?<![\w\d])(?:\+?506[\s().\-]*)?[2-9](?:[\s().\-]*\d){7}(?![\w\d])")
NATIONAL_ID = re.compile(r"(?<![\w\d])[1-9](?:[\s.\-]*\d){8}(?![\w\d])")
ACCOUNT = re.compile(r"(?<!\w)CR(?:[\s.\-]*\d){20}(?![\w\d])", re.I)
PRIORITY = {"NAME": 1, "PHONE": 2, "DOB": 3, "ID": 4, "EMAIL": 5, "ACCOUNT": 6, "ADDRESS": 7}
NAME_STOP = {
    "i", "my", "me", "is", "am", "and", "will", "can", "please", "today", "tomorrow",
    "urgent", "priority", "regards", "thanks", "thank", "sincerely", "hello", "dear", "hi",
    "pura", "vida", "voyages", "team", "support", "booking", "reference", "reservation",
    "refund", "payment", "reschedule", "complaint", "at", "from", "for", "to", "with",
    "soy", "llamo", "nombre", "gracias", "saludos", "hola", "equipo",
    "writing", "requesting", "still", "waiting", "trying", "contacting", "looking",
    "interested", "happy", "very", "disappointed", "upset", "calling", "travelling", "traveling",
}
PARTICLES = {"de", "del", "la", "las", "los", "y", "da", "do", "dos", "van", "von"}


def normalized(value):
    return "".join(c for c in unicodedata.normalize("NFKD", value.casefold()) if c.isalnum())


def indexed_normalized(text):
    chars, starts, ends = [], [], []
    for pos, char in enumerate(text):
        for clean in unicodedata.normalize("NFKD", char.casefold()):
            if clean.isalnum():
                chars.append(clean)
                starts.append(pos)
                ends.append(pos + 1)
    return "".join(chars), starts, ends


def overlaps(start, end, protected):
    return any(start < b and end > a for a, b in protected)


class Redactor:
    def __init__(self, text):
        self.text = text
        self.spans = []
        self.aliases = {}
        self.protected = [m.span() for pattern in (BOOKING, PLACEHOLDER) for m in pattern.finditer(text)]

    def span(self, start, end, category):
        if start < end and not overlaps(start, end, self.protected):
            self.spans.append((start, end, category))

    def alias(self, value, category, minimum=3):
        key = normalized(value)
        if len(key) < minimum:
            return
        current = self.aliases.get(key)
        if current is None or PRIORITY[category] > PRIORITY[current]:
            self.aliases[key] = category

    def entity(self, start, end, category):
        if overlaps(start, end, self.protected):
            return
        self.span(start, end, category)
        self.alias(self.text[start:end], category)

    def name(self, start, end):
        if overlaps(start, end, self.protected):
            return
        # Captures are context-bounded; tokenize only within the supplied value.
        value = self.text[start:end]
        words = list(re.finditer(WORD, value, re.UNICODE))
        accepted = []
        for match in words[:8]:
            key = normalized(match.group())
            if key in NAME_STOP:
                break
            accepted.append(match)
        while accepted and normalized(accepted[-1].group()) in PARTICLES:
            accepted.pop()
        if not accepted:
            return
        first, last = accepted[0].start() + start, accepted[-1].end() + start
        if normalized(self.text[first:last]) == "puravidavoyages":
            return
        self.entity(first, last, "NAME")
        for match in accepted:
            if normalized(match.group()) not in PARTICLES:
                self.alias(match.group(), "NAME", minimum=2)

    def find_names(self):
        rules = (
            r"\b(?:my\s+(?:full\s+)?name\s+is\s+|name\s*[:=]\s*|me\s+llamo\s+|mi\s+nombre\s+es\s+)(?P<value>[^\n.!?;,:]+)",
            r"\b(?:I\s+am|I'm|I’m|soy)\s+(?P<value>" + WORD + r"(?:\s+" + WORD + r"){0,5})(?=[,.!;\n]|$)",
            r"\b(?:my|mi)\s+(?:wife|husband|son|daughter|partner|spouse|friend|mother|father|sister|brother|"
            r"esposa|esposo|hijo|hija|pareja)\s+(?P<value>[^\n,.!?;]+?)(?=\s+(?:will|is|can|and|travels?|viaja|viajar[aá])\b|[,.!?;\n]|$)",
            r"(?im)^(?:regards|kind\s+regards|best\s+regards|best|sincerely|thank\s+you|thanks|saludos|atentamente)[,!:.\s]*\n[ \t]*(?P<value>[^\n]+)",
            r"\b(?:Mr|Mrs|Ms|Miss|Dr|Sr|Sra|Srta)\.?\s+(?P<value>" + WORD + r"(?:\s+" + WORD + r"){0,3})",
        )
        for rule_number, rule in enumerate(rules):
            for match in re.finditer(rule, self.text, re.I):
                if rule_number in (1, 4) and not match.group("value")[0].isupper():
                    continue
                self.name(*match.span("value"))

    def find_structured(self):
        for match in EMAIL.finditer(self.text):
            # The regex can include sentence-final periods in exotic domains.
            start, end = match.span()
            while end > start and self.text[end - 1] == ".":
                end -= 1
            self.entity(start, end, "EMAIL")
            local = self.text[start:end].split("@", 1)[0]
            if normalized(local) not in self.aliases:
                self.alias(local, "EMAIL")
        for category, pattern in (("ACCOUNT", ACCOUNT), ("ID", NATIONAL_ID), ("PHONE", PHONE)):
            for match in pattern.finditer(self.text):
                if overlaps(*match.span(), self.protected):
                    continue
                if category == "PHONE" and re.fullmatch(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", match.group()):
                    # An itinerary date is useful data, not a phone number.
                    # Birth dates are handled by their explicit context above.
                    continue
                self.entity(*match.span(), category)
                if category == "PHONE":
                    digits = re.sub(r"\D", "", match.group())
                    if digits.startswith("506") and len(digits) == 11:
                        self.alias(digits[3:], "PHONE")

    def find_birth_dates(self):
        pattern = (r"\b(?:date\s+of\s+birth|birth\s*date|birthday|dob|born\s+(?:on\s+)?|fecha\s+de\s+nacimiento|"
                   r"nac[ií](?:\s+el)?)\s*(?:(?:is|es|was)\s*|[:=]\s*)?(?P<value>" + DATE + r")")
        for match in re.finditer(pattern, self.text, re.I):
            value = match.group("value")
            self.entity(*match.span("value"), "DOB")
            parts = re.findall(r"\d+|[^\W\d_]+", value.casefold())
            numbers = re.findall(r"\d+", value)
            month = next((MONTH_NUMBER[p] for p in parts if p in MONTH_NUMBER), None)
            if month:
                day, year = int(numbers[0]), int(numbers[-1])
            elif len(numbers[0]) == 4:
                year, month, day = map(int, numbers)
            else:
                day, month, year = map(int, numbers)
                if month > 12 and day <= 12:
                    day, month = month, day
            try:
                date(year, month, day)
            except ValueError:
                continue
            self.alias(f"{year:04d}-{month:02d}-{day:02d}", "DOB")
            for d, m in ((day, month), (month, day)):
                for ds in (str(d), f"{d:02d}"):
                    for ms in (str(m), f"{m:02d}"):
                        self.alias(f"{ds}/{ms}/{year}", "DOB")
            for name in MONTHS[month - 1]:
                for ds in (str(day), f"{day:02d}"):
                    for ending in ("", "st", "nd", "rd", "th"):
                        self.alias(f"{name} {ds}{ending}, {year}", "DOB")
                    self.alias(f"{ds} {name} {year}", "DOB")
                    self.alias(f"{ds} de {name} de {year}", "DOB")

    def address(self, start, end):
        if overlaps(start, end, self.protected):
            return
        while end > start and self.text[end - 1] in " .;\t\r":
            end -= 1
        self.entity(start, end, "ADDRESS")
        chunks = [p.strip() for p in self.text[start:end].split(",")]
        for chunk in chunks:
            if re.search(r"\b(?:calle|avenida|av\.?|street|road|ruta|casa|house|apartamento|edificio|"
                         r"metros|frente|contiguo|iglesia|parque|panader[ií]a|supermercado)\b", chunk, re.I):
                self.alias(chunk, "ADDRESS")
                landmark = re.search(r"\b(?:iglesia|parque|panader[ií]a|supermercado|casa|house|"
                                     r"apartamento|edificio)\b.+", chunk, re.I)
                if landmark:
                    self.alias(landmark.group(), "ADDRESS")
                    # A landmark can be repeated using only its distinctive name.
                    short = re.sub(r"^(?:iglesia|parque|panader[ií]a|supermercado)\s+(?:de\s+)?", "", landmark.group(), flags=re.I)
                    if len(normalized(short)) >= 6 and len(short.split()) >= 2:
                        self.alias(short, "ADDRESS")

    def find_addresses(self):
        # Stop before booking/urgency clauses so their full wording survives.
        street = r"(?:calle|avenida|av\.|street|road|ruta|barrio|urbanizaci[oó]n|casa|apartamento)\b"
        patterns = (
            r"\b(?:pickup\s+at|paperwork\s+to|written\s+reply\s+to|(?:postal\s+)?address\s*(?:is|:)|"
            r"direcci[oó]n\s*(?:es|:)|vivo\s+en)\s+(?P<value>[^\n]+?)"
            r"(?=\s+to\s+our\s+booking\b|\s+(?:and\s+)?keep\s+reference\b|[.;\n]|$)",
            r"\b(?P<value>" + street + r"\s+[^\n;.]+?)"
            r"(?=\s+to\s+our\s+booking\b|\s+(?:and\s+)?keep\s+reference\b|[.;\n]|$)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, self.text, re.I):
                self.address(*match.span("value"))

    def expand_aliases(self):
        if not self.aliases:
            return
        clean, starts, ends = indexed_normalized(self.text)
        pattern = re.compile("|".join(re.escape(key) for key in sorted(self.aliases, key=len, reverse=True)))
        for match in pattern.finditer(clean):
            start, end = starts[match.start()], ends[match.end() - 1]
            category = self.aliases[match.group()]
            if category == "NAME" and ((start and self.text[start - 1].isalnum()) or
                                       (end < len(self.text) and self.text[end].isalnum())):
                continue
            # Include trailing combining accents in a decomposed spelling.
            while end < len(self.text) and unicodedata.combining(self.text[end]):
                end += 1
            self.span(start, end, category)

    def run(self):
        self.find_names()
        self.find_birth_dates()
        self.find_addresses()
        self.find_structured()
        self.expand_aliases()
        merged = []
        for start, end, category in sorted(self.spans, key=lambda s: (s[0], -s[1])):
            if merged and start < merged[-1][1]:
                a, b, cat = merged[-1]
                if (end - start, PRIORITY[category]) > (b - a, PRIORITY[cat]):
                    cat = category
                merged[-1] = (a, max(b, end), cat)
            else:
                merged.append((start, end, category))
        pieces = []
        previous = 0
        for start, end, category in merged:
            pieces.extend((self.text[previous:start], "[" + category + "]"))
            previous = end
        pieces.append(self.text[previous:])
        return "".join(pieces) or "[REDACTED]"


def redact(text):
    return Redactor(text).run()


def solve(data):
    return {"redacted_documents": {document["id"]: redact(document["text"])
                                    for document in data["documents"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("-o", "--output", dest="output_option")
    args = parser.parse_args()
    input_path = args.input or ("test-input.json" if Path("test-input.json").is_file() else "train-input.json")
    training = Path(input_path).name == "train-input.json"
    output_path = args.output_option or args.output or ("train-output.json" if training else "test-output.json")
    if training and not args.output_option and not args.output and Path(output_path).exists():
        raise FileExistsError("Refusing to overwrite training golden file; use -o practice-output.json")
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Input and output paths must differ")
    with open(input_path, encoding="utf-8-sig") as stream:
        result = solve(json.load(stream))
    with open(output_path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
