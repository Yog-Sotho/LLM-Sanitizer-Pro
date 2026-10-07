"""Generate the bundled PII evaluation set (sanitizer_pro/data/pii_eval.jsonl).

Deterministic synthetic records with exact character spans:
  * positives: emails, URLs, phones (NANP + international), payment cards
    (valid Luhn, several issuers), IBANs (valid mod-97), SSNs, IPv4/IPv6, and
    the NER kinds person / location / address, partly in FR/DE/ES
  * hard negatives: things regexes confuse with PII (non-Luhn order IDs,
    timestamps, version strings, git SHAs, UUIDs, ISBNs, prices, code slices,
    loopback addresses)

All values are synthetic; reserved/documentation ranges are used where they
exist (example.* domains, 555-01xx numbers, TEST-NET and 2001:db8:: IPs).

    python scripts/make_pii_eval.py
"""
import json
import random
import string
from pathlib import Path
from typing import Callable, List, Tuple

OUT = Path(__file__).resolve().parent.parent / 'sanitizer_pro' / 'data' / 'pii_eval.jsonl'
rng = random.Random(20261007)

FIRST = ["Maria", "John", "Aiko", "Mohammed", "Olga", "Liam", "Chen", "Fatima", "Lucas", "Priya",
         "Sofia", "Kwame", "Elena", "Hiroshi", "Amara", "Diego", "Ingrid", "Ravi", "Nadia", "Tomasz"]
LAST = ["Jensen", "Smith", "Tanaka", "Al-Hassan", "Petrova", "O'Brien", "Wang", "Diallo",
        "Moreau", "Sharma", "Rossi", "Mensah", "Kowalski", "Nakamura", "Okafor", "Garcia",
        "Lindqvist", "Iyer", "Haddad", "Novak"]
CITIES = ["Lisbon", "Nairobi", "Osaka", "Toronto", "Krakow", "Marseille", "Bogota", "Adelaide",
          "Leipzig", "Chennai", "Valparaiso", "Tromso"]
STREETS = ["Elm Street", "Rue de la Paix", "Hauptstrasse", "Calle Mayor", "Maple Avenue",
           "King Road", "Via Roma", "Oak Lane"]
DOMAINS = ["example.com", "example.org", "mail.example.net", "corp.example.co.uk"]


def luhn_complete(prefix: str, length: int) -> str:
    body = prefix + ''.join(rng.choice(string.digits) for _ in range(length - len(prefix) - 1))
    for check in string.digits:
        digits = body + check
        total = 0
        for i, ch in enumerate(reversed(digits)):
            d = int(ch)
            if i % 2:
                d = d * 2 - 9 if d > 4 else d * 2
            total += d
        if total % 10 == 0:
            return digits
    raise AssertionError


def not_luhn(length: int) -> str:
    while True:
        d = ''.join(rng.choice(string.digits) for _ in range(length))
        if d[0] in '23456' and luhn_complete(d[:-1], length) != d:
            return d


def group(d: str, sizes: Tuple[int, ...], sep: str) -> str:
    out, i = [], 0
    for s in sizes:
        out.append(d[i:i + s])
        i += s
    return sep.join(out)


def card() -> str:
    kind = rng.randrange(4)
    sep = rng.choice([' ', '-', ''])
    if kind == 0:
        return group(luhn_complete('4', 16), (4, 4, 4, 4), sep)
    if kind == 1:
        return group(luhn_complete(rng.choice(['51', '53', '55']), 16), (4, 4, 4, 4), sep)
    if kind == 2:
        return group(luhn_complete(rng.choice(['34', '37']), 15), (4, 6, 5), sep)
    return group(luhn_complete('6011', 16), (4, 4, 4, 4), sep)


def iban() -> str:
    country, bban = rng.choice([
        ('DE', ''.join(rng.choice(string.digits) for _ in range(18))),
        ('GB', 'WEST' + ''.join(rng.choice(string.digits) for _ in range(14))),
        ('FR', ''.join(rng.choice(string.digits) for _ in range(23))),
        ('NL', 'ABNA' + ''.join(rng.choice(string.digits) for _ in range(10))),
    ])
    rearranged = bban + country + '00'
    check = 98 - int(''.join(str(int(c, 36)) for c in rearranged)) % 97
    compact = f"{country}{check:02d}{bban}"
    if rng.random() < 0.5:
        return ' '.join(compact[i:i + 4] for i in range(0, len(compact), 4))
    return compact


def ssn() -> str:
    area = rng.choice([a for a in range(1, 900) if a != 666])
    return f"{area:03d}-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}"


def phone() -> str:
    n = f"{rng.randint(10, 99):02d}"
    return rng.choice([
        f"({rng.choice(['212', '415', '646'])}) 555-01{n}",
        f"{rng.choice(['212', '415', '646'])}-555-01{n}",
        f"{rng.choice(['212', '415', '646'])}.555.01{n}",
        f"+1 415 555 01{n}",
        f"+44 20 7946 09{n}",
        f"+49 30 9018 20{n}",
        f"+33 1 42 68 53 {n}",
    ])


def email(first: str, last: str) -> str:
    local = rng.choice([f"{first}.{last}", f"{first[0]}{last}", f"{first}+news"]).lower()
    return local.replace("'", '') + '@' + rng.choice(DOMAINS)


def ipv4() -> str:
    return rng.choice([f"192.0.2.{rng.randint(1, 254)}", f"198.51.100.{rng.randint(1, 254)}",
                       f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"])


def ipv6() -> str:
    return rng.choice([f"2001:db8:{rng.randint(1, 0xffff):x}::{rng.randint(1, 0xffff):x}",
                       f"2001:db8:85a3:0:0:8a2e:370:{rng.randint(1, 0xffff):x}"])


def url() -> str:
    return rng.choice([f"https://www.{rng.choice(DOMAINS)}/account?id={rng.randint(100, 999)}",
                       f"http://{rng.choice(DOMAINS)}/users/{rng.randint(1, 99)}",
                       f"www.{rng.choice(DOMAINS)}/profile"])


def address() -> str:
    return f"{rng.randint(1, 250)} {rng.choice(STREETS)}"


Piece = Tuple[str, str]  # (text, kind or '')


def build(pieces: List[Piece]) -> dict:
    text, spans = '', []
    for chunk, kind in pieces:
        if kind:
            spans.append([len(text), len(text) + len(chunk), kind])
        text += chunk
    return {"text": text, "spans": spans}


def positive() -> dict:
    f, last = rng.choice(FIRST), rng.choice(LAST)
    person = f"{f} {last}"
    templates: List[Callable[[], List[Piece]]] = [
        lambda: [("Please email ", ''), (person, 'person'), (" at ", ''),
                 (email(f, last), 'email'), (" before Friday.", '')],
        lambda: [("Call ", ''), (person, 'person'), (" on ", ''), (phone(), 'phone'),
                 (" or write to ", ''), (address(), 'address'), (".", '')],
        lambda: [("Card ", ''), (card(), 'card'), (" was charged; refund to IBAN ", ''),
                 (iban(), 'iban'), (".", '')],
        lambda: [("Server ", ''), (ipv4(), 'ip'), (" logged a request from ", ''),
                 (ipv6(), 'ip'), (" at 09:14.", '')],
        lambda: [("Details are on the portal (", ''), (url(), 'url'), (").", '')],
        lambda: [("SSN on file for ", ''), (person, 'person'), (": ", ''), (ssn(), 'ssn'), (".", '')],
        lambda: [(person, 'person'), (" moved from ", ''), (rng.choice(CITIES), 'location'),
                 (" to ", ''), (rng.choice(CITIES), 'location'), (" last year.", '')],
        lambda: [("Bonjour, ici ", ''), (person, 'person'), (". Rappelez-moi au ", ''),
                 (phone(), 'phone'), (" ou écrivez à ", ''), (email(f, last), 'email'), (".", '')],
        lambda: [("Bitte überweisen Sie den Betrag auf ", ''), (iban(), 'iban'),
                 (" (Kontoinhaber: ", ''), (person, 'person'), (").", '')],
        lambda: [("Hola ", ''), (person, 'person'), (", tu pedido llegará a ", ''),
                 (address(), 'address'), (", ", ''), (rng.choice(CITIES), 'location'), (".", '')],
    ]
    return build(rng.choice(templates)())


def negative() -> dict:
    options = [
        lambda: f"Order {not_luhn(16)} shipped on 2026-03-14.",
        lambda: f"Tracking number {''.join(rng.choice(string.digits) for _ in range(22))} is live.",
        lambda: f"Timestamp {rng.randint(1_600_000_000, 1_800_000_000)} was recorded.",
        lambda: f"Upgrade to version {rng.randint(1, 9)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}"
                f".{rng.randint(0, 99)} today.",
        lambda: f"Commit {''.join(rng.choice('0123456789abcdef') for _ in range(40))} fixed it.",
        lambda: "Request id 550e8400-e29b-41d4-a716-446655440000 processed.",
        lambda: "ISBN 978-3-16-148410-0 is the second edition.",
        lambda: f"The bundle costs ${rng.randint(100, 9999):,}.{rng.randint(10, 99)} for 3 seats.",
        lambda: "In Python, a[1::2] takes every second item and b[::3] every third.",
        lambda: "The service listens on 127.0.0.1 and 0.0.0.0 by default.",
        lambda: "Invalid SSN placeholders like 000-00-0000 must be rejected.",
        lambda: "Population grew from 1,204,551 to 1,388,902 between 2010 and 2020.",
    ]
    return {"text": rng.choice(options)(), "spans": []}


def main() -> None:
    records = [positive() for _ in range(400)] + [negative() for _ in range(200)]
    rng.shuffle(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        for r in records:
            for start, end, _ in r["spans"]:
                assert r["text"][start:end].strip() == r["text"][start:end]
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"wrote {len(records)} records to {OUT}")


if __name__ == '__main__':
    main()
