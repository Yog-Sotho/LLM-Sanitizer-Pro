<div align="center">
  <img src="https://raw.githubusercontent.com/Yog-Sotho/LLM-Sanitizer-Pro/main/Images/banner.png" alt="LLM Sanitizer PRO: scrub, redact, dedup and curate data for LLM training" width="100%"/>
  <p>
    <a href="https://pypi.org/project/llm-sanitizer-pro/"><img src="https://img.shields.io/pypi/v/llm-sanitizer-pro?color=8b5cf6&logo=pypi&logoColor=white" alt="PyPI: llm-sanitizer-pro"></a>
    <a href="https://pypi.org/project/llm-sanitizer-pro/"><img src="https://img.shields.io/pypi/pyversions/llm-sanitizer-pro?color=3b82f6" alt="Python versions"></a>
    <a href="https://github.com/Yog-Sotho/LLM-Sanitizer-Pro/actions/workflows/ci.yml"><img src="https://github.com/Yog-Sotho/LLM-Sanitizer-Pro/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://github.com/Yog-Sotho/LLM-Sanitizer-Pro/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://yog-sotho.github.io/LLM-Sanitizer-Pro/"><img src="https://img.shields.io/badge/docs-online-8b5cf6" alt="Documentation"></a>
  </p>
</div>

# 🧼 LLM Sanitizer PRO

**Your AI is only as clean as the data you feed it.**<br>
Point it at a messy dataset → it scrubs out personal info, leaked passwords, duplicates and junk → you get training data you're not embarrassed to use.

---

## 🤔 What is this?

So you want to train or fine-tune an AI on your own data: support chats, documents, scraped web pages, old forum posts. Great idea! Tiny problem: real-world data is *filthy*.

It's full of people's email addresses and phone numbers. Someone pasted an API key into a support ticket back in 2019. Half the rows are copy-pastes of the other half. And a good chunk of it is just… garbage.

Feed that to a model and it learns **all** of it, including your customers' phone numbers. 😬

**LLM Sanitizer PRO is the washing machine for training data.** One command, and out comes a dataset that is:

- 🕵️ **Private.** Emails, phone numbers, credit cards, bank accounts (IBAN), US social security numbers and IP addresses are swapped for safe placeholders.
- 🔑 **Secret-free.** Leaked API keys, tokens, passwords and private keys (AWS, GitHub, OpenAI, Stripe, Slack and many more) get scrubbed.
- 👯 **Duplicate-free.** Exact copies, near-copies, and even reworded copies get thrown out.
- 🗑️ **Junk-free.** Too-short, gibberish, ALL-CAPS-SCREAMING and low-quality text is filtered out.
- 🎓 **Cheat-proof.** Questions from famous AI tests (MMLU, GSM8K, HumanEval and friends) are removed, so your model can't just memorize the answers.
- 🍽️ **Ready to serve.** Exported in the formats training tools expect (ChatML, Alpaca), split into train/validation/test if you like.

No PhD required. If you can copy-paste one command, you can do this.

---

## 🖼️ How it works

<div align="center">
  <img src="https://raw.githubusercontent.com/Yog-Sotho/LLM-Sanitizer-Pro/main/Images/pipeline.png" alt="Raw records go through parse, PII and secrets redaction, dedup, quality, decontamination and formatting to become a clean dataset" width="750"/>
  <p><em>Dirty data goes in the left, clean data comes out the right. Every record takes the same trip through the machine.</em></p>
</div>

---

## 🔍 See it in action

Here's one record before and after. This is real output, not a mock-up.

**Before:**

> Hi, I'm Jane. Email me at jane.doe@example.com or call +1 415-555-0132. My card is 4111 1111 1111 1111. The deploy key is ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8 so please keep it safe.

**After:**

> Hi, I'm Jane. Email me at [PII_EMAIL] or call [PII_PHONE]. My card is [PII_CARD]. The deploy key is [SECRET_GITHUB_TOKEN] so please keep it safe.

And the identical copy of that record further down the file? Gone. Duplicates removed. ✨

Not a fan of placeholders? Pick your style:

| Style | Add this | Jane's email becomes |
|---|---|---|
| Placeholder (default) | *(nothing)* | `[PII_EMAIL]` |
| Partially hidden | `--pii-mask` | `j***e@example.com` |
| Fake but consistent | `--pii-pseudonymize` | `email_0001@redacted.local` |

> 💡 **Did you notice "Jane" survived?** Names, places and company names need a smarter, AI-based detector. Switch it on with `--pii-ner` (one extra install, explained [here](https://yog-sotho.github.io/LLM-Sanitizer-Pro/features/#installing-ner-backed-pii-detection)).

---

## 🚀 Get started in 2 minutes

### 1. Install it

```bash
pip install "llm-sanitizer-pro[all]"
```

You need Python 3.10 or newer. Don't want to install anything? Use the container instead (it already has everything inside). Put your data in the current folder and run (the `--user` part lets it save the result into your folder):

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/data" ghcr.io/yog-sotho/llm-sanitizer-pro \
    --input my_data.jsonl --output clean.jsonl --profile fine-tune --report audit.html
```

### 2. Clean your data

```bash
sanitize --input my_data.jsonl --output clean.jsonl --profile fine-tune --report audit.html
```

That's it. `clean.jsonl` is your sparkling new dataset.

### 3. Check what happened

Open `audit.html` in your browser. It shows what got removed and why, what kind of personal info was hidden, and before/after samples. It's a single file, so you can email it to your boss or your privacy team. 📎

<div align="center">
  <img src="https://raw.githubusercontent.com/Yog-Sotho/LLM-Sanitizer-Pro/main/Images/terminal.png" alt="A sanitize run with the fine-tune profile, secrets redaction, fuzzy dedup and 4 jobs" width="750"/>
</div>

---

## 🎛️ Not sure which settings? Pick a profile

Tell it what the data is for, and it picks sensible settings for you:

| Profile | Use it for | What it does |
|---|---|---|
| `--profile fine-tune` | Chat or question/answer data for fine-tuning a model | Hides personal info and secrets, removes duplicates, strips HTML, drops very short rows |
| `--profile pretrain` | Big piles of text and documents | Removes secrets and duplicates, strips HTML, keeps only long, decent-quality text |
| `--profile rag` | Text chunks for a chatbot's knowledge base | Hides personal info and secrets, removes duplicates, strips HTML, drops tiny chunks |

Anything you add yourself always wins over the profile. Run `sanitize --profile list` to see exactly what each one switches on.

---

## 🧰 "I want to…" cheat sheet

| I want to… | Add this |
|---|---|
| hide personal info (emails, phones, cards…) | `--remove-pii` |
| …and also names, places and companies | `--pii-ner` ([one extra install](https://yog-sotho.github.io/LLM-Sanitizer-Pro/features/#installing-ner-backed-pii-detection)) |
| remove leaked passwords and API keys | `--redact-secrets` |
| remove exact duplicates | `--deduplicate` |
| remove near-duplicates | `--fuzzy-dedup` |
| remove reworded duplicates | `--semantic-dedup` (needs `pip install "llm-sanitizer-pro[semantic]"`) |
| keep only good-quality text | `--quality-min-score 0.5` |
| keep only one language | `--lang-filter en` |
| stop my model cheating on AI tests | `--decontaminate all` |
| get ChatML or Alpaca format for training | `--format-chatml` or `--format-instruct` |
| check chat conversations aren't broken | `--validate-chat` |
| split into train / validation / test | `--split train=0.9,val=0.05,test=0.05` |
| go faster | `--jobs 4` |
| see what would happen without writing anything | `--dry-run` |
| pick up where a huge job left off after a crash | `--resume` |

Mix and match freely. More copy-paste examples live in the [recipes](https://yog-sotho.github.io/LLM-Sanitizer-Pro/recipes/).

---

## 📂 What can I feed it?

JSONL, JSON, CSV/TSV, plain text, Parquet and Excel, zipped (`.gz`) or not. Give it a single file, a whole folder, or a dataset straight from the Hugging Face Hub (`--input hf://owner/dataset`).

---

## 🛡️ Your data stays yours

Everything runs on your own computer. Your data is never uploaded anywhere. The only downloads are optional extras that you switch on yourself: AI models for name detection or quality scoring, and the public test sets used for `--decontaminate`.

---

## 🤓 For the nerds

The full technical story lives in the [documentation](https://yog-sotho.github.io/LLM-Sanitizer-Pro/):

- 📖 [Features in detail](https://yog-sotho.github.io/LLM-Sanitizer-Pro/features/): every feature, its flag, and its measured accuracy
- 🍳 [Recipes](https://yog-sotho.github.io/LLM-Sanitizer-Pro/recipes/): 26 copy-paste commands for common jobs
- ⌨️ [CLI reference](https://yog-sotho.github.io/LLM-Sanitizer-Pro/cli/): every option and its default
- 🐍 [Python API](https://yog-sotho.github.io/LLM-Sanitizer-Pro/python-api/): use it as a library, plus how to develop on it
- ⚡ [Performance](https://yog-sotho.github.io/LLM-Sanitizer-Pro/performance/): about 20,000 records per second with 4 jobs
- 🧾 [Provenance](https://yog-sotho.github.io/LLM-Sanitizer-Pro/provenance/): run manifests and Hugging Face dataset cards
- 🔏 [Verifying releases](https://yog-sotho.github.io/LLM-Sanitizer-Pro/releases/): signatures and build provenance
- 📜 [Changelog](https://github.com/Yog-Sotho/LLM-Sanitizer-Pro/blob/main/CHANGELOG.md)

---

## License

MIT

## Author

Yogsotho

<div align="center">
  <img src="https://raw.githubusercontent.com/Yog-Sotho/LLM-Sanitizer-Pro/main/Images/logo.png" alt="LLM Sanitizer PRO logo" width="220"/>
</div>
