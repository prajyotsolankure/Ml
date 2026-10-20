#!/usr/bin/env python3
"""
test2.py – 100% Offline Translator (Pi 5 Optimised)
Target display: 3.5" RPi touchscreen — 480×320 px

Supports: English, Hindi, French, Chinese, Spanish, German,
          Telugu, Tamil, Kannada

Features:
  ✦ Voice Input  – Vosk (offline STT)
  ✦ Speech Output – Piper TTS / espeak-ng fallback
  ✦ Keyboard Input
  ✦ ZERO network calls during translation
  ✦ MarianMT (OPUS-MT) for Telugu/Tamil/Kannada
  ✦ Context-aware chunking for better sentence/paragraph translation

Pre-install packages (run ONCE while online):
    cd /home/pns/Desktop/Trasnlator
    ./lingo_env/bin/python test2.py --install-packages
"""

import os
import json
import math
import queue
import struct
import subprocess
import sys
import threading
import tkinter as tk
import unicodedata
from pathlib import Path

# Only enforce strict offline mode if we are NOT running the installer
if "--install-packages" not in sys.argv:
    os.environ["ARGOS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # ── FIX 1: Block Stanza import ─────────────────────────────────────
    # Stanza (used by argostranslate for sentence splitting) silently
    # downloads NLP tokenizer models on first use.  Blocking the import
    # forces argostranslate to use its built-in regex sentence splitter.

    # ── FIX 2: Block ALL outgoing network connections ──────────────────
    # Some libraries (stanza, sentencepiece, huggingface hub, argos
    # internals) sneak in downloads on first use even with env vars set.
    # We monkey-patch socket.create_connection so that NO library can
    # open an internet connection.  Localhost is still allowed.
    import socket as _socket
    _real_create_connection = _socket.create_connection

    def _blocked_create_connection(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
            return _real_create_connection(address, *args, **kwargs)
        raise OSError(
            f"[OFFLINE] Blocked connection to {host} — "
            f"run --install-packages once with internet to cache all models."
        )

    _socket.create_connection = _blocked_create_connection

# ══════════════════════════════════════════════════════════════════════════════
#  CLI: pre-install packages (run once while online)
# ══════════════════════════════════════════════════════════════════════════════

def cli_install_packages():
    try:
        import argostranslate.package
        import argostranslate.translate
    except ImportError:
        print("[ERROR] argostranslate not found. Run: pip install argostranslate")
        sys.exit(1)

    pairs = [
        ("en", "hi"), ("en", "fr"), ("en", "zh"), ("en", "es"), ("en", "de"),
        ("hi", "en"), ("fr", "en"), ("zh", "en"), ("es", "en"), ("de", "en"),
    ]
    print("Updating package index (requires internet)...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    print("\n--- 1. Downloading Translation Dictionaries ---")
    for src, tgt in pairs:
        pkg = next((p for p in available if p.from_code == src and p.to_code == tgt), None)
        if pkg is None:
            print(f"  [SKIP] No package found for {src}→{tgt}")
            continue
        print(f"  Installing {src}→{tgt} ...")
        try:
            argostranslate.package.install_from_path(pkg.download())
            print(f"  [OK]   {src}→{tgt}")
        except Exception as e:
            print(f"  [FAIL] {src}→{tgt}: {e}")

    print("\n--- 2. Downloading Stanza NLP Models (Crucial for offline) ---")

    # Step 2a: Explicitly download Stanza tokenizer resources per language
    try:
        import stanza
        stanza_lang_map = {
            "en": "en",  "hi": "hi",      "fr": "fr",
            "zh": "zh-hans", "es": "es",  "de": "de",
        }
        for code, stanza_code in stanza_lang_map.items():
            print(f"  Downloading Stanza tokenizer for {code} ({stanza_code}) ...")
            try:
                stanza.download(stanza_code, processors="tokenize", logging_level="WARN")
                print(f"  [OK]   Stanza {code}")
            except Exception as e:
                print(f"  [FAIL] Stanza {code}: {e}")
    except ImportError:
        print("  [INFO] stanza not installed — app will use regex sentence splitter (that's fine)")

    # Step 2b: Warm up by doing dummy translations (caches any remaining resources)
    print("  Warming up translation models with dummy translations...")
    installed = argostranslate.translate.get_installed_languages()
    for src, tgt in pairs:
        src_lang = next((l for l in installed if l.code == src), None)
        tgt_lang = next((l for l in installed if l.code == tgt), None)
        
        if src_lang and tgt_lang:
            translator = src_lang.get_translation(tgt_lang)
            if translator:
                print(f"  Warming up {src}→{tgt} ...")
                try:
                    translator.translate("Hello world.")
                    print(f"  [OK]   {src}→{tgt} ready")
                except Exception as e:
                    print(f"  [FAIL] Warmup {src}→{tgt}: {e}")

    # ── 3. Download OPUS-MT (MarianMT) models for Telugu, Tamil, Kannada ──
    print("\n--- 3. Downloading OPUS-MT Models (Telugu, Tamil, Kannada) ---")
    try:
        from transformers import MarianMTModel, MarianTokenizer
        # Dedicated models provide vastly superior translation accuracy for South Indian languages,
        # with fallback to the multilingual models if needed.
        marian_models = [
            "Helsinki-NLP/opus-mt-en-te",   # English → Telugu (dedicated, high accuracy)
            "Helsinki-NLP/opus-mt-en-ta",   # English → Tamil (dedicated, high accuracy)
            "Helsinki-NLP/opus-mt-en-kn",   # English → Kannada (dedicated, high accuracy)
            "Helsinki-NLP/opus-mt-dra-en",   # Dravidian → English
            "Helsinki-NLP/opus-mt-en-dra",   # English → Dravidian (multilingual fallback)
        ]
        for model_name in marian_models:
            print(f"  Downloading {model_name} ...")
            try:
                MarianTokenizer.from_pretrained(model_name)
                MarianMTModel.from_pretrained(model_name)
                print(f"  [OK]   {model_name}")
            except Exception as e:
                print(f"  [FAIL] {model_name}: {e}")

        # Warm up with dummy translations to ensure all models are cached and ready
        print("  Warming up MarianMT models ...")
        warmup_tests = [
            ("Helsinki-NLP/opus-mt-en-te", "Hello world."),
            ("Helsinki-NLP/opus-mt-en-ta", "Hello world."),
            ("Helsinki-NLP/opus-mt-en-kn", "Hello world."),
            ("Helsinki-NLP/opus-mt-dra-en", "నమస్కారం"),
            ("Helsinki-NLP/opus-mt-en-dra", ">>tel<< Hello world."),
        ]
        for mdl_name, test_input in warmup_tests:
            try:
                tok = MarianTokenizer.from_pretrained(mdl_name)
                mdl = MarianMTModel.from_pretrained(mdl_name)
                inputs = tok(test_input, return_tensors="pt", padding=True)
                mdl.generate(**inputs, max_new_tokens=30, num_beams=2)
                print(f"  [OK] MarianMT warm-up complete for {mdl_name}")
            except Exception as e:
                print(f"  [SKIP] MarianMT warm-up {mdl_name}: {e}")
    except ImportError:
        print("  [SKIP] transformers not installed. Run: pip install transformers sentencepiece")

    print("\nDone! All packages and NLP models installed. App will now work fully offline.")
    sys.exit(0)


if "--install-packages" in sys.argv:
    cli_install_packages()


# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import re as _re

try:
    import argostranslate.translate
    import argostranslate.package
except ImportError:
    print("[ERROR] argostranslate not found. Run: pip install argostranslate")
    sys.exit(1)

# ── FIX: Replace Stanza sentence splitter with offline regex splitter ──
# Stanza downloads resources.json from raw.githubusercontent.com on every
# first use.  We replace argostranslate's sentence boundary detection with
# a smart context-aware splitter that works 100% offline, no downloads ever.
if "--install-packages" not in sys.argv:
    import argostranslate.sbd as _sbd

    # ── Common abbreviations that must NOT trigger sentence splits ────
    _ABBREV_PATTERN = (
        r'(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Gen|Gov|Sgt|Corp|Inc|Ltd|Rev|'
        r'Vol|Fig|Dept|Est|Approx|Rep|Sec|Capt|Col|Cmdr|Lt|Maj|'
        r'vs|etc|approx|viz|'
        r'i\.e|e\.g|a\.m|p\.m|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    )

    def _robust_sentence_split(text):
        """Split text into individual sentences, respecting abbreviations,
        decimal numbers, ellipses, and single-letter initials like U.S.A.
        
        This is the low-level splitter — it returns one sentence per element.
        """
        if not text or not text.strip():
            return [text.strip()] if text and text.strip() else []

        text = text.strip()

        # --- Step 1: Shield patterns that look like sentence endings ---
        shields = {}
        _cnt = [0]

        def _shield(m):
            key = f'\x00S{_cnt[0]}\x00'
            shields[key] = m.group(0)
            _cnt[0] += 1
            return key

        protected = text
        # Shield abbreviations followed by space (Mr. Smith → keep together)
        protected = _re.sub(
            r'\b' + _ABBREV_PATTERN + r'\.\s',
            _shield, protected, flags=_re.IGNORECASE
        )
        # Shield decimal numbers (3.14, $2.50, 0.001)
        protected = _re.sub(r'\d+\.\d+', _shield, protected)
        # Shield ellipses (... or …)
        protected = _re.sub(r'\.{2,}', _shield, protected)
        protected = _re.sub(r'…', _shield, protected)
        # Shield single-letter initials (U.S.A., J.K., D.C.)
        protected = _re.sub(r'(?:[A-Z]\.){2,}', _shield, protected)
        # Shield URLs (http://..., www....)
        protected = _re.sub(r'https?://\S+', _shield, protected)
        protected = _re.sub(r'www\.\S+', _shield, protected)
        # Shield time patterns (10.30, 5.00 when preceded by "at" or similar)
        protected = _re.sub(r'\b\d{1,2}\.\d{2}\b', _shield, protected)

        # --- Step 2: Split on real sentence boundaries ---
        # Split after .!? when followed by:
        #   - whitespace + uppercase letter (Latin)
        #   - whitespace + Devanagari (Hindi)
        #   - whitespace + CJK character (Chinese)
        #   - whitespace + Dravidian script (Telugu/Tamil/Kannada)
        #   - whitespace + digit (numbered lists)
        #   - whitespace + quote mark
        parts = _re.split(
            r'(?<=[.!?।])\s+'
            r'(?=['
            r'A-Z'                   # Latin uppercase
            r'\u0900-\u097F'         # Devanagari (Hindi)
            r'\u4e00-\u9fff'         # CJK (Chinese)
            r'\u0C00-\u0C7F'         # Telugu
            r'\u0B80-\u0BFF'         # Tamil
            r'\u0C80-\u0CFF'         # Kannada
            r'\u0D00-\u0D7F'         # Malayalam
            r'0-9'                   # Numbered items
            r'"\u201C\u201D'         # Quotation marks
            r"'"                     # Apostrophe/single quote
            r'])',
            protected
        )

        # --- Step 3: Restore shielded patterns ---
        result = []
        for part in parts:
            for key, val in shields.items():
                part = part.replace(key, val)
            part = part.strip()
            if part:
                result.append(part)

        return result if result else [text]

    def _group_sentences(sentences, max_chunk_chars=450):
        """Group sentences into translation-sized chunks to preserve context.

        Keeping 2-3 sentences together lets the translation model see
        inter-sentence context (pronouns, topic continuity) which produces
        much better paragraph translation than sentence-by-sentence.
        """
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        chunks = []
        current_group = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            # Start new chunk if adding this sentence would exceed limit
            if current_len + sent_len > max_chunk_chars and current_group:
                chunks.append(' '.join(current_group))
                current_group = [sent]
                current_len = sent_len
            else:
                current_group.append(sent)
                current_len += sent_len + 1  # +1 for the joining space

        if current_group:
            chunks.append(' '.join(current_group))

        return chunks if chunks else sentences

    def _offline_split(text, lang_code=None):
        """Smart offline text chunking for translation.

        Strategy:
          • Short text  (< 500 chars):  return as single chunk — no splitting
            ↳ The translation model sees full context → best quality
          • Medium text (500–2000 chars): split into sentences, group 2-3 together
            ↳ Preserves inter-sentence context while staying within model limits
          • Long text   (> 2000 chars):  paragraph-aware splitting with grouping
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # SHORT TEXT: translate as one unit for maximum context retention
        if len(text) < 500:
            return [text]

        # LONGER TEXT: split into paragraphs first, then chunk within each
        paragraphs = _re.split(r'\n\s*\n', text)

        all_chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) < 500:
                # Short paragraph: keep as one chunk
                all_chunks.append(para)
            else:
                # Long paragraph: split into sentences and group
                sentences = _robust_sentence_split(para)
                grouped = _group_sentences(sentences, max_chunk_chars=450)
                all_chunks.extend(grouped)

        return all_chunks if all_chunks else [text]

    # Override all sentence detection functions with our improved version
    if hasattr(_sbd, 'detect_sentences'):
        _sbd.detect_sentences = _offline_split
    if hasattr(_sbd, 'split_sentences'):
        _sbd.split_sentences = _offline_split

# MarianMT (OPUS-MT) for Telugu, Tamil, Kannada
try:
    from transformers import MarianMTModel, MarianTokenizer
    MARIAN_AVAILABLE = True
except ImportError:
    MARIAN_AVAILABLE = False
    print("[INFO] transformers not found – Telugu/Tamil/Kannada translation disabled.")
    print("       Run: pip install transformers sentencepiece")

try:
    import vosk
    import sounddevice as sd
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("[INFO] vosk/sounddevice not found – voice input disabled.")


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DISPLAY_W   = 480
DISPLAY_H   = 320
FS_TITLE    = 10
FS_LABEL    = 9
FS_LANG     = 10
FS_TEXT     = 10
FS_BTN      = 9
FS_STATUS   = 8
FS_CHAR     = 8
FS_DIR      = 8
INPUT_ROWS  = 3
OUTPUT_ROWS = 3
BTN_PADY    = 3
BTN_PADX    = 10

_BEST_FONT_FAMILY = None

def get_ui_font(size: int, weight: str = "normal"):
    """Return an OpenType font family that properly renders complex Indic scripts."""
    global _BEST_FONT_FAMILY
    if _BEST_FONT_FAMILY is None:
        try:
            import tkinter.font as tkfont
            avail = set(tkfont.families())
            candidates = [
                "Nirmala UI",          # Windows standard with complete Indic OpenType layout
                "Noto Sans",           # Linux / Raspberry Pi OS standard
                "Noto Sans Telugu",    # Specific Indic Noto fonts
                "Noto Sans Tamil",
                "Noto Sans Kannada",
                "Noto Sans Devanagari",
                "Lohit Telugu",        # Lohit Indic fonts (common on Debian/RPi)
                "Lohit Tamil",
                "Lohit Kannada",
                "DejaVu Sans",         # Linux standard proportional
                "Segoe UI",            # Windows standard proportional
                "Arial",
            ]
            for c in candidates:
                if c in avail:
                    _BEST_FONT_FAMILY = c
                    break
            if not _BEST_FONT_FAMILY:
                _BEST_FONT_FAMILY = "TkDefaultFont"
        except Exception:
            _BEST_FONT_FAMILY = "TkDefaultFont"

    if weight and weight != "normal":
        return (_BEST_FONT_FAMILY, size, weight)
    return (_BEST_FONT_FAMILY, size)


# ══════════════════════════════════════════════════════════════════════════════
#  LANGUAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

LANGUAGES = {
    "English": "en",
    "Hindi":   "hi",
    "French":  "fr",
    "Chinese": "zh",
    "Spanish": "es",
    "German":  "de",
    "Telugu":  "te",
    "Tamil":   "ta",
    "Kannada": "kn",
}

LANG_FLAGS = {
    "English": "🇬🇧",
    "Hindi":   "🇮🇳",
    "French":  "🇫🇷",
    "Chinese": "🇨🇳",
    "Spanish": "🇪🇸",
    "German":  "🇩🇪",
    "Telugu":  "🇮🇳",
    "Tamil":   "🇮🇳",
    "Kannada": "🇮🇳",
}

PIPER_DIR = Path.home() / "piper"
PIPER_MODELS = {
    "en": "en_US-lessac-medium.onnx",
    "hi": "hi_IN-swara-medium.onnx",
    "fr": "fr_FR-siwis-medium.onnx",
    "zh": "zh_CN-huayan-medium.onnx",
    "es": "es_ES-carlfm-medium.onnx",
    "de": "de_DE-thorsten-medium.onnx",
    "te": "te_IN-maya-medium.onnx",
}

# Vosk models — LARGER models first (much better accuracy), small as fallback.
# The loader tries each in order and uses the first one found on disk.
# Run --install-packages to download larger models, or manually place them in ~/vosk-models/
VOSK_MODEL_PRIORITY = {
    "en": ["vosk-model-en-us-0.22-lgraph",  "vosk-model-en-us-0.22",  "vosk-model-small-en-us-0.15"],
    "hi": ["vosk-model-hi-0.22",             "vosk-model-small-hi-0.22"],
    "fr": ["vosk-model-fr-0.22-lgraph",      "vosk-model-small-fr-0.22"],
    "zh": ["vosk-model-cn-0.22",             "vosk-model-small-cn-0.22"],
    "es": ["vosk-model-es-0.42",             "vosk-model-small-es-0.42"],
    "de": ["vosk-model-de-0.21",             "vosk-model-small-de-0.15"],
    "te": ["vosk-model-small-te-0.42"],
    # NOTE: No Vosk models exist for Tamil (ta) or Kannada (kn).
    #       Voice input is keyboard-only for those languages.
}

# Backward-compat: flat map used by error messages
VOSK_MODEL_NAMES = {k: v[0] for k, v in VOSK_MODEL_PRIORITY.items()}

VOSK_MODELS_DIR = Path.home() / "vosk-models"

# Languages that use MarianMT (OPUS-MT) instead of Argos Translate
MARIAN_LANGUAGES = {"te", "ta", "kn"}

# Dedicated English → Language models (vastly higher accuracy and grammatical correctness)
MARIAN_DEDICATED_MODELS = {
    "te": "Helsinki-NLP/opus-mt-en-te",
    "ta": "Helsinki-NLP/opus-mt-en-ta",
    "kn": "Helsinki-NLP/opus-mt-en-kn",
}

# Fallback OPUS-MT target language tokens for the multilingual en→dra model
MARIAN_LANG_TOKENS = {
    "te": ">>tel<<",
    "ta": ">>tam<<",
    "kn": ">>kan<<",
}

# Model names on HuggingFace (downloaded once, then cached locally)
MARIAN_MODEL_EN_DRA = "Helsinki-NLP/opus-mt-en-dra"   # English → Dravidian fallback
MARIAN_MODEL_DRA_EN = "Helsinki-NLP/opus-mt-dra-en"   # Dravidian → English


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT PRE / POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _clean_indic_orthography(text: str) -> str:
    """Normalize and re-attach detached matras, combining vowels, and viramas in Indic scripts.

    SentencePiece subword tokenization frequently breaks Indic syllables by placing spaces
    before dependent vowel signs (matras), anusvara, or after viramas (halants/ottu).
    This function restores correct composite glyphs and canonical Unicode (NFC).
    """
    if not text:
        return text
    # 1. Canonical Unicode composition normalization
    text = unicodedata.normalize("NFC", text)
    # 2. Re-attach dependent vowels (matras), anusvara, visarga, and viramas to preceding consonants
    text = _re.sub(
        r'[ \t]+([\u0901-\u094D\u0962\u0963\u0B82\u0BBE-\u0BCD\u0BD7\u0C01-\u0C03\u0C3E-\u0C4D\u0C55\u0C56\u0C81-\u0C83\u0CBE-\u0CCD\u0CD5\u0CD6\u0D01-\u0D03\u0D3E-\u0D4D\u0D57])',
        r'\1',
        text
    )
    # 3. Remove space after viramas (halants) so that consonants fuse into valid conjuncts (ottu)
    text = _re.sub(r'([\u094D\u0BCD\u0C4D\u0CCD\u0D4D])[ \t]+', r'\1', text)
    # 4. Strip invisible control characters that break complex text layout
    text = text.replace('\u200b', '').replace('\ufeff', '')
    return text


def _preprocess_text(text: str) -> str:
    """Normalise whitespace and clean input before translation."""
    if not text:
        return text
    text = _clean_indic_orthography(text)
    # Collapse runs of spaces/tabs into single space
    text = _re.sub(r'[ \t]+', ' ', text)
    # Collapse 3+ newlines into double newline (paragraph break)
    text = _re.sub(r'\n{3,}', '\n\n', text)
    # Ensure space after sentence-ending punctuation if missing (supports Latin, Hindi, CJK, Telugu, Tamil, Kannada, Malayalam)
    text = _re.sub(r'([.!?।])([A-Z\u0900-\u097F\u4e00-\u9fff\u0C00-\u0C7F\u0B80-\u0BFF\u0C80-\u0CFF\u0D00-\u0D7F])', r'\1 \2', text)
    return text.strip()


def _postprocess_text(text: str) -> str:
    """Clean up common translation artifacts."""
    if not text:
        return text
    # Remove any leaked model target tokens (e.g. >>tel<<, >>tam<<, >>kan<<)
    text = _re.sub(r'>>[a-zA-Z_-]+<<', '', text)
    # Fix multiple consecutive spaces
    text = _re.sub(r'  +', ' ', text)
    # Remove stray spaces before punctuation
    text = _re.sub(r'\s+([.!?,;:。！？，；：।])', r'\1', text)
    # Add space after punctuation when directly followed by a word char
    text = _re.sub(
        r'([.!?,;:।])([A-Za-z\u0900-\u097F\u4e00-\u9fff\u0C00-\u0C7F\u0B80-\u0BFF\u0C80-\u0CFF\u0D00-\u0D7F])',
        r'\1 \2',
        text
    )
    # Fix repeated punctuation from model artifacts (e.g. "....." → "...")
    text = _re.sub(r'([.!?।])\1{2,}', r'\1\1\1', text)
    # Restore correct Indic orthography, combining marks, and conjuncts
    text = _clean_indic_orthography(text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSLATION  (NO network calls — purely local)
# ══════════════════════════════════════════════════════════════════════════════

_translation_cache: dict = {}
_marian_availability_cache: dict = {}


def _is_marian_model_available(model_name: str) -> bool:
    """Check if MarianMT model files exist locally in Hugging Face cache (zero network calls)."""
    if not MARIAN_AVAILABLE:
        return False
    if model_name in _marian_availability_cache:
        return _marian_availability_cache[model_name]
    if model_name in _marian_model_cache:
        _marian_availability_cache[model_name] = True
        return True
    try:
        MarianTokenizer.from_pretrained(model_name, local_files_only=True)
        _marian_availability_cache[model_name] = True
        return True
    except Exception:
        _marian_availability_cache[model_name] = False
        return False


def _is_marian_tgt_available(tgt_code: str) -> bool:
    """Check if any model capable of translating English to tgt_code is available offline."""
    dedicated = MARIAN_DEDICATED_MODELS.get(tgt_code)
    if dedicated and _is_marian_model_available(dedicated):
        return True
    return _is_marian_model_available(MARIAN_MODEL_EN_DRA)


def _is_argos_pair_installed(src_code: str, tgt_code: str) -> bool:
    try:
        installed = argostranslate.translate.get_installed_languages()
        src_lang = next((l for l in installed if l.code == src_code), None)
        tgt_lang = next((l for l in installed if l.code == tgt_code), None)
        if src_lang is None or tgt_lang is None:
            return False
        return src_lang.get_translation(tgt_lang) is not None
    except Exception:
        return False


# ── Argos Translate helpers (for en/hi/fr/zh/es/de) ──────────────────────────

def _is_pair_installed(src_code: str, tgt_code: str) -> bool:
    """Check if a translation pair is ready (Argos or MarianMT)."""
    if src_code == tgt_code:
        return True

    src_marian = src_code in MARIAN_LANGUAGES
    tgt_marian = tgt_code in MARIAN_LANGUAGES

    # Both are Dravidian (e.g. te→ta): requires dra→en and en→tgt
    if src_marian and tgt_marian:
        if not MARIAN_AVAILABLE:
            return False
        return _is_marian_model_available(MARIAN_MODEL_DRA_EN) and _is_marian_tgt_available(tgt_code)

    # Source is Dravidian (e.g. te→hi): requires dra→en, then Argos en→tgt
    if src_marian and not tgt_marian:
        if not MARIAN_AVAILABLE or not _is_marian_model_available(MARIAN_MODEL_DRA_EN):
            return False
        return True if tgt_code == "en" else _is_argos_pair_installed("en", tgt_code)

    # Target is Dravidian (e.g. hi→te): requires Argos src→en, then en→tgt
    if tgt_marian and not src_marian:
        if not MARIAN_AVAILABLE or not _is_marian_tgt_available(tgt_code):
            return False
        return True if src_code == "en" else _is_argos_pair_installed(src_code, "en")

    # Pure Argos pairs
    return _is_argos_pair_installed(src_code, tgt_code)


def _get_translation_obj(src_code: str, tgt_code: str):
    key = (src_code, tgt_code)
    if key in _translation_cache:
        return _translation_cache[key]

    installed = argostranslate.translate.get_installed_languages()
    src_lang = next((l for l in installed if l.code == src_code), None)
    tgt_lang = next((l for l in installed if l.code == tgt_code), None)

    if src_lang is None or tgt_lang is None:
        return None

    translation = src_lang.get_translation(tgt_lang)
    if translation:
        _translation_cache[key] = translation
    return translation


# ── MarianMT helpers (for te/ta/kn) ──────────────────────────────────────────

_marian_model_cache: dict = {}  # {model_name: (tokenizer, model)}


def _load_marian_model(model_name: str):
    """Load a MarianMT model from local HuggingFace cache (no network)."""
    if model_name in _marian_model_cache:
        return _marian_model_cache[model_name]

    if not MARIAN_AVAILABLE:
        return None

    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name, local_files_only=True)
        model = MarianMTModel.from_pretrained(model_name, local_files_only=True)
        model.eval()
        _marian_model_cache[model_name] = (tokenizer, model)
        _marian_availability_cache[model_name] = True
        return (tokenizer, model)
    except Exception as e:
        print(f"[MarianMT load error for {model_name}]: {e}")
        return None


def _marian_translate(text: str, model_name: str, lang_token: str = "") -> str:
    """Translate using a MarianMT model with context-aware chunking."""
    pair = _load_marian_model(model_name)
    if pair is None:
        return (
            f"[OPUS-MT model not cached: {model_name}]\n"
            f"Run: ./lingo_env/bin/python test2.py --install-packages"
        )
    tokenizer, model = pair

    text = _preprocess_text(text)
    if not text:
        return ""

    def _translate_chunk(chunk: str) -> str:
        """Translate a single chunk through MarianMT."""
        src_text = f"{lang_token} {chunk}".strip() if lang_token else chunk
        print(f"[MarianMT] Input: '{src_text}' | Model: {model_name}")

        try:
            inputs = tokenizer(
                src_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            # no_repeat_ngram_size is intentionally omitted: Dravidian languages use
            # frequent subword repetitions in regular morphology; setting n-gram bans
            # causes severe word corruption and abrupt truncation.
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if lang_token:
                result = result.replace(lang_token.strip(), "")
            result = _re.sub(r'>>[a-zA-Z_-]+<<', '', result).strip()
            result = _clean_indic_orthography(result)
            print(f"[MarianMT] Output: '{result}'")

            if result and result.strip():
                return result.strip()

            # Fallback: try decoding without skipping special tokens if result is empty
            raw = tokenizer.decode(outputs[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)
            if raw and raw.strip():
                for tok in ["</s>", "<pad>", "<s>", "<unk>"]:
                    raw = raw.replace(tok, "")
                raw = _re.sub(r'>>[a-zA-Z_-]+<<', '', raw).strip()
                if raw.strip():
                    return raw.strip()
            return ""
        except Exception as e:
            print(f"[MarianMT translation chunk error]: {e}")
            return ""

    # SHORT TEXT (< 400 chars): translate as one unit for best context
    if len(text) < 400:
        result = _translate_chunk(text)
        if result:
            return _postprocess_text(result)
        return "[No result — model may not support this language pair]"

    # LONGER TEXT: split into smart chunks and translate each
    paragraphs = _re.split(r'(\n\s*\n)', text)
    translated_parts = []

    for part in paragraphs:
        if not part.strip():
            translated_parts.append(part)
            continue

        if len(part) < 400:
            result = _translate_chunk(part)
            translated_parts.append(result if result else part)
        else:
            sentences = _robust_sentence_split(part)
            chunks = _group_sentences(sentences, max_chunk_chars=350)
            chunk_results = []
            for chunk in chunks:
                result = _translate_chunk(chunk)
                chunk_results.append(result if result else chunk)
            translated_parts.append(' '.join(chunk_results))

    final = ''.join(translated_parts)
    if final.strip():
        return _postprocess_text(final)
    return "[No result — model may not support this language pair]"


def _translate_en_to_dravidian(english_text: str, tgt_code: str) -> str:
    """Translate English into a Dravidian language (te, ta, kn).

    Prefers dedicated single-language models (Helsinki-NLP/opus-mt-en-te,
    opus-mt-en-ta, opus-mt-en-kn) for vastly superior translation quality.
    Gracefully falls back to multilingual Helsinki-NLP/opus-mt-en-dra if
    the dedicated model is not cached.
    """
    dedicated_model = MARIAN_DEDICATED_MODELS.get(tgt_code)
    if dedicated_model and _is_marian_model_available(dedicated_model):
        res = _marian_translate(english_text, dedicated_model, lang_token="")
        if res and not res.startswith("["):
            return res

    # Fallback to shared multilingual Dravidian model (en-dra)
    token = MARIAN_LANG_TOKENS.get(tgt_code, "")
    return _marian_translate(english_text, MARIAN_MODEL_EN_DRA, token)


# ── Unified translate function ───────────────────────────────────────────────

def do_translate(text: str, src_code: str, tgt_code: str) -> str:
    """Translate offline — zero network calls."""
    if src_code == tgt_code:
        return text

    src_marian = src_code in MARIAN_LANGUAGES
    tgt_marian = tgt_code in MARIAN_LANGUAGES

    try:
        # Case 1: Both are Dravidian → pivot through English (te→en→ta)
        if src_marian and tgt_marian:
            english = _marian_translate(text, MARIAN_MODEL_DRA_EN)
            if not english or english.startswith("["):
                return english if english else "[Translation failed: could not translate to English]"
            return _translate_en_to_dravidian(english, tgt_code)

        # Case 2: Source is Dravidian → translate to English first, then Argos
        if src_marian and not tgt_marian:
            english = _marian_translate(text, MARIAN_MODEL_DRA_EN)
            if not english or english.startswith("["):
                return english if english else "[Translation failed: could not translate to English]"
            if tgt_code == "en":
                return english
            # English → target via Argos
            return _argos_translate(english, "en", tgt_code)

        # Case 3: Target is Dravidian → Argos to English first, then MarianMT
        if tgt_marian and not src_marian:
            if src_code == "en":
                english = text
            else:
                english = _argos_translate(text, src_code, "en")
                if not english or english.startswith("["):
                    return english if english else "[Translation failed: could not translate to English]"
            return _translate_en_to_dravidian(english, tgt_code)

        # Case 4: Neither is Dravidian → pure Argos
        return _argos_translate(text, src_code, tgt_code)

    except Exception as e:
        return f"[Translation error: {e}]"


def _argos_translate(text: str, src_code: str, tgt_code: str) -> str:
    """Translate using Argos Translate with context-aware chunking.

    Strategy:
      • Short text  (< 500 chars):  pass as single unit to Argos
        ↳ Our overridden sbd returns it as-is → model sees full context
      • Longer text:  split into paragraphs, then smart-chunk each paragraph
        ↳ Each chunk keeps 2-3 sentences together for inter-sentence context
    """
    t = _get_translation_obj(src_code, tgt_code)
    if t is None:
        return (
            f"[Package not installed: {src_code}→{tgt_code}]\n"
            f"Run: ./lingo_env/bin/python test2.py --install-packages"
        )

    text = _preprocess_text(text)

    # SHORT TEXT: let Argos translate as a single unit
    # Our overridden _offline_split (in sbd) returns text <500 chars as-is,
    # so Argos will translate the whole thing without splitting → best quality.
    if len(text) < 500:
        result = t.translate(text)
        return _postprocess_text(result.strip()) if result else "[No result]"

    # LONGER TEXT: manually chunk and translate for paragraph coherence
    paragraphs = _re.split(r'(\n\s*\n)', text)
    translated_parts = []

    for part in paragraphs:
        if not part.strip():
            translated_parts.append(part)  # preserve paragraph breaks
            continue

        if len(part) < 500:
            # Short paragraph: translate as one unit
            result = t.translate(part)
            translated_parts.append(result.strip() if result else part)
        else:
            # Long paragraph: split into sentences, group into chunks
            sentences = _robust_sentence_split(part)
            chunks = _group_sentences(sentences, max_chunk_chars=450)
            chunk_results = []
            for chunk in chunks:
                result = t.translate(chunk)
                chunk_results.append(result.strip() if result else chunk)
            translated_parts.append(' '.join(chunk_results))

    final = ''.join(translated_parts)
    return _postprocess_text(final) if final.strip() else "[No result]"


# ══════════════════════════════════════════════════════════════════════════════
#  VOSK SPEECH-TO-TEXT (with audio normalisation & confidence filtering)
# ══════════════════════════════════════════════════════════════════════════════

_vosk_model_cache: dict = {}
_vosk_active_model_name: dict = {}  # lang_code → model name that was loaded


def get_vosk_model(lang_code: str):
    """Load the best available Vosk model for a language.

    Tries larger, more accurate models first and falls back to small ones.
    """
    if not VOSK_AVAILABLE:
        return None
    if lang_code in _vosk_model_cache:
        return _vosk_model_cache[lang_code]

    candidates = VOSK_MODEL_PRIORITY.get(lang_code, [])
    if not candidates:
        return None

    # Try each model in priority order (largest/best first)
    for model_name in candidates:
        model_path = VOSK_MODELS_DIR / model_name
        if model_path.exists():
            try:
                vosk.SetLogLevel(-1)
                model = vosk.Model(str(model_path))
                _vosk_model_cache[lang_code] = model
                _vosk_active_model_name[lang_code] = model_name
                print(f"[Vosk] Loaded model: {model_name}")
                return model
            except Exception as e:
                print(f"[Vosk load error for {model_name}]: {e}")
                continue

    print(f"[Vosk] No model found for '{lang_code}'. Tried: {candidates}")
    return None


def _normalise_audio(raw_bytes: bytes, target_rms: float = 3000.0) -> bytes:
    """Normalise audio volume so quiet speech is boosted for better recognition.

    Vosk works best when audio RMS is around 2000-4000.  If the mic is quiet
    (RMS < 500) recognition accuracy drops drastically.  This function
    measures the current RMS and applies gain to bring it to `target_rms`.
    """
    n_samples = len(raw_bytes) // 2
    if n_samples == 0:
        return raw_bytes

    samples = struct.unpack(f'<{n_samples}h', raw_bytes)

    # Calculate RMS (root mean square) of the audio
    sum_sq = sum(s * s for s in samples)
    rms = math.sqrt(sum_sq / n_samples) if n_samples > 0 else 0.0

    if rms < 10.0:
        # Silence — don't amplify noise
        return raw_bytes

    # Calculate gain factor (cap at 10x to avoid blowing up noise)
    gain = min(target_rms / rms, 10.0)

    if 0.8 <= gain <= 1.2:
        # Already at good level — skip processing
        return raw_bytes

    # Apply gain and clamp to int16 range
    boosted = []
    for s in samples:
        v = int(s * gain)
        v = max(-32768, min(32767, v))  # clamp to int16
        boosted.append(v)

    return struct.pack(f'<{n_samples}h', *boosted)


def _filter_by_confidence(result_json: dict, min_conf: float = 0.45) -> str:
    """Filter recognised words by confidence score.

    Low-confidence words are often misrecognitions.  If word-level results
    are available, keep only words above `min_conf`.  Falls back to the
    plain text if no word-level data.
    """
    # If word-level results are available, filter by confidence
    words = result_json.get("result", [])
    if words:
        kept = [w["word"] for w in words if w.get("conf", 1.0) >= min_conf]
        if kept:
            return " ".join(kept)

    # Fallback to plain text result
    return result_json.get("text", "").strip()


class VoiceRecorder:
    SAMPLERATE = 16000
    CHANNELS   = 1
    DTYPE      = "int16"
    BLOCKSIZE  = 4000   # 250ms blocks (was 500ms) — more responsive

    # Confidence threshold: words below this are likely misrecognitions
    MIN_CONFIDENCE = 0.45

    def __init__(self, lang_code, on_result, on_partial, on_error):
        self.lang_code  = lang_code
        self.on_result  = on_result
        self.on_partial = on_partial
        self.on_error   = on_error
        self._stop_evt  = threading.Event()
        self._thread    = None

    def start(self):
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()

    def _run(self):
        model = get_vosk_model(self.lang_code)
        if model is None:
            candidates = VOSK_MODEL_PRIORITY.get(self.lang_code, [])
            self.on_error(
                f"Vosk model for '{self.lang_code}' not found.\n"
                f"Download one of: {', '.join(candidates) if candidates else '(none available)'}\n"
                f"Place in: {VOSK_MODELS_DIR}/"
            )
            return

        active = _vosk_active_model_name.get(self.lang_code, "?")
        print(f"[Vosk] Recording with model: {active}")

        q = queue.Queue()
        rec = vosk.KaldiRecognizer(model, self.SAMPLERATE)
        # Enable word-level results with confidence scores
        rec.SetWords(True)
        rec.SetPartialWords(True)
        full_text = []

        def callback(indata, frames, time_info, status):
            q.put(bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=self.SAMPLERATE,
                channels=self.CHANNELS,
                dtype=self.DTYPE,
                blocksize=self.BLOCKSIZE,
                callback=callback,
            ):
                while not self._stop_evt.is_set():
                    try:
                        data = q.get(timeout=0.3)
                    except queue.Empty:
                        continue

                    # Normalise audio volume for better recognition
                    data = _normalise_audio(data)

                    if rec.AcceptWaveform(data):
                        r = json.loads(rec.Result())
                        # Use confidence-filtered words instead of raw text
                        word = _filter_by_confidence(r, self.MIN_CONFIDENCE)
                        if word:
                            full_text.append(word)
                            self.on_partial(" ".join(full_text))
                    else:
                        p = json.loads(rec.PartialResult())
                        partial = p.get("partial", "").strip()
                        if partial:
                            self.on_partial(" ".join(full_text + [partial]))

            # Final result with confidence filtering
            r = json.loads(rec.FinalResult())
            word = _filter_by_confidence(r, self.MIN_CONFIDENCE)
            if word:
                full_text.append(word)
            self.on_result(" ".join(full_text).strip())

        except Exception as e:
            self.on_error(f"Microphone error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TTS
# ══════════════════════════════════════════════════════════════════════════════

def check_tts() -> bool:
    if (PIPER_DIR / "piper").exists():
        return True
    try:
        r = subprocess.run(["espeak-ng", "--version"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def speak(text: str, lang_code: str):
    piper_exec = PIPER_DIR / "piper"
    model_name = PIPER_MODELS.get(lang_code)

    if piper_exec.exists() and model_name:
        model_path = PIPER_DIR / model_name
        if model_path.exists():
            try:
                safe = text.replace('"', "'").replace('\n', ' ')
                cmd = (
                    f'echo "{safe}" | {piper_exec} '
                    f'--model {model_path} --output-raw | '
                    f'paplay --raw --channels=1 --rate=22050 --format=s16le'
                )
                subprocess.Popen(cmd, shell=True)
                return
            except Exception as e:
                print(f"[Piper error]: {e}")

    # Fallback: espeak-ng
    espeak_voices = {
        "en": "en-us", "hi": "hi", "fr": "fr",
        "zh": "zh",    "es": "es", "de": "de",
        "te": "te",    "ta": "ta", "kn": "kn",
    }
    voice = espeak_voices.get(lang_code, "en")
    try:
        safe = text.replace('"', "'").replace('\n', ' ')
        subprocess.Popen(
            ["espeak-ng", "-v", voice, safe],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print("[TTS] espeak-ng not found. Run: sudo apt install espeak-ng")
    except Exception as e:
        print(f"[TTS error]: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════════

C = {
    "bg":              "#0D0D0D",
    "bg2":             "#181818",
    "bg_input":        "#1C1C1C",
    "bg_output":       "#121212",
    "btn_primary":     "#FFFFFF",
    "btn_primary_hov": "#DDDDDD",
    "btn_dark":        "#1E1E1E",
    "btn_dark_hov":    "#2E2E2E",
    "btn_record":      "#7A0000",
    "btn_record_live": "#FF2020",
    "btn_record_hov":  "#990000",
    "fg":              "#FFFFFF",
    "fg_dim":          "#777777",
    "fg_btn":          "#0D0D0D",
    "border":          "#2A2A2A",
    "border_focus":    "#FFFFFF",
    "ok":              "#999999",
    "err":             "#FF6B6B",
    "warn":            "#FFB347",
    "sep":             "#252525",
}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

class OffLingoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OffLingo")
        self.root.configure(bg=C["bg"])
        self.root.geometry(f"{DISPLAY_W}x{DISPLAY_H}+0+0")
        self.root.resizable(False, False)
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

        self.tts_ok     = check_tts()
        self.voice_ok   = VOSK_AVAILABLE
        self._recording = False
        self._recorder  = None

        self.src_lang = tk.StringVar(value="English")
        self.tgt_lang = tk.StringVar(value="Hindi")

        self._build_ui()
        self._refresh_status()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        main = tk.Frame(self.root, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=4, pady=2)

        # ROW 1: title + selectors + swap + exit
        row1 = tk.Frame(main, bg=C["bg"])
        row1.pack(fill="x", pady=(0, 2))

        tk.Label(
            row1, text="◉ OffLingo",
            font=get_ui_font(FS_TITLE, "bold"),
            fg=C["fg"], bg=C["bg"]
        ).pack(side="left", padx=(0, 6))

        self._make_lang_selector(row1, "FROM", self.src_lang, side="left")

        swap = tk.Label(
            row1, text="⇄",
            font=get_ui_font(11, "bold"), fg=C["fg"], bg=C["btn_dark"],
            width=2, cursor="hand2", relief="flat", padx=4, pady=2
        )
        swap.pack(side="left", padx=3)
        swap.bind("<Button-1>", self._swap_languages)
        self._hover(swap, C["btn_dark"], C["btn_dark_hov"])

        self._make_lang_selector(row1, "TO", self.tgt_lang, side="left")

        exit_btn = tk.Label(
            row1, text="✖",
            font=get_ui_font(11, "bold"), fg=C["err"], bg=C["bg"],
            cursor="hand2", padx=4
        )
        exit_btn.pack(side="right", padx=(4, 0))
        exit_btn.bind("<Button-1>", lambda e: self.root.destroy())

        min_btn = tk.Label(
            row1, text="—",
            font=get_ui_font(11, "bold"), fg=C["fg"], bg=C["bg"],
            cursor="hand2", padx=4
        )
        min_btn.pack(side="right", padx=(4, 4))
        min_btn.bind("<Button-1>", lambda e: self.root.iconify())

        self.dir_label = tk.Label(
            row1, text="",
            font=get_ui_font(FS_DIR),
            fg=C["fg_dim"], bg=C["bg"]
        )
        self.dir_label.pack(side="right", padx=(4, 0))
        self._update_dir_label()

        # Separator
        tk.Frame(main, bg=C["sep"], height=1).pack(fill="x", pady=(1, 2))

        # ROW 2: input box
        in_wrap = tk.Frame(
            main, bg=C["border"],
            highlightthickness=1,
            highlightbackground=C["border"],
            highlightcolor=C["border_focus"]
        )
        in_wrap.pack(fill="x")
        self.input_text = tk.Text(
            in_wrap,
            height=INPUT_ROWS,
            font=get_ui_font(FS_TEXT),
            fg=C["fg"], bg=C["bg_input"],
            insertbackground=C["fg"],
            selectbackground=C["border"],
            selectforeground=C["fg"],
            relief="flat", bd=3,
            wrap="word",
            highlightthickness=0
        )
        self.input_text.pack(fill="both", expand=True)
        self.input_text.bind("<Return>", self._on_enter)
        self.input_text.bind("<KeyRelease>", self._update_char_count)
        self.input_text.focus_set()

        # ROW 3: buttons
        btn_row = tk.Frame(main, bg=C["bg"])
        btn_row.pack(fill="x", pady=(2, 2))

        self.translate_btn = self._make_btn(
            btn_row, "GO",
            C["btn_primary"], C["btn_primary_hov"], C["fg_btn"],
            lambda e: self._translate(), side="left"
        )

        self._make_btn(
            btn_row, "CLR",
            C["btn_dark"], C["btn_dark_hov"], C["fg_dim"],
            lambda e: self._clear(), side="left", padx_l=3
        )

        if self.voice_ok:
            self.mic_btn = tk.Label(
                btn_row, text="🎙 SPEAK",
                font=get_ui_font(FS_BTN),
                fg=C["fg"], bg=C["btn_record"],
                padx=BTN_PADX, pady=BTN_PADY, cursor="hand2"
            )
            self.mic_btn.pack(side="left", padx=(3, 0))
            self.mic_btn.bind("<Button-1>", self._toggle_recording)
        else:
            tk.Label(
                btn_row, text="🎙 –",
                font=get_ui_font(FS_BTN, "italic"),
                fg=C["fg_dim"], bg=C["bg"]
            ).pack(side="left", padx=(3, 0))

        if self.tts_ok:
            self._make_btn(
                btn_row, "🔊",
                C["btn_dark"], C["btn_dark_hov"], C["fg_dim"],
                lambda e: self._speak_output(), side="left", padx_l=3
            )

        self.char_count = tk.Label(
            btn_row, text="0",
            font=get_ui_font(FS_CHAR),
            fg=C["fg_dim"], bg=C["bg"]
        )
        self.char_count.pack(side="right", padx=(0, 2))

        # ROW 4: output box
        out_wrap = tk.Frame(
            main, bg=C["border"],
            highlightthickness=1,
            highlightbackground=C["border"]
        )
        out_wrap.pack(fill="both", expand=True, pady=(0, 1))
        self.output_text = tk.Text(
            out_wrap,
            height=OUTPUT_ROWS,
            font=get_ui_font(FS_TEXT),
            fg=C["fg"], bg=C["bg_output"],
            relief="flat", bd=3,
            wrap="word",
            state="disabled",
            cursor="arrow",
            highlightthickness=0
        )
        self.output_text.pack(fill="both", expand=True)

        # ROW 5: status bar
        st = tk.Frame(main, bg=C["bg"])
        st.pack(fill="x")
        self.status_lbl = tk.Label(
            st, text="Ready",
            font=get_ui_font(FS_STATUS),
            fg=C["fg_dim"], bg=C["bg"], anchor="w"
        )
        self.status_lbl.pack(side="left")
        self.model_lbl = tk.Label(
            st, text="",
            font=get_ui_font(FS_STATUS),
            fg=C["fg_dim"], bg=C["bg"], anchor="e"
        )
        self.model_lbl.pack(side="right")

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _make_lang_selector(self, parent, label_text, var, side="left"):
        container = tk.Frame(parent, bg=C["bg"])
        container.pack(side=side, fill="x", expand=True)

        tk.Label(
            container, text=label_text,
            font=get_ui_font(FS_LABEL, "bold"),
            fg=C["fg_dim"], bg=C["bg"]
        ).pack(anchor="w")

        sf = tk.Frame(
            container, bg=C["bg_input"],
            highlightthickness=1,
            highlightbackground=C["border"],
            highlightcolor=C["border_focus"]
        )
        sf.pack(fill="x")

        btn = tk.Menubutton(
            sf, textvariable=var,
            font=get_ui_font(FS_LANG),
            fg=C["fg"], bg=C["bg_input"],
            activebackground=C["btn_dark_hov"],
            activeforeground=C["fg"],
            relief="flat", bd=3,
            indicatoron=False,
            cursor="hand2",
            anchor="w", width=9
        )
        menu = tk.Menu(
            btn, tearoff=False,
            font=get_ui_font(FS_LANG),
            fg=C["fg"], bg=C["bg2"],
            activebackground=C["btn_primary"],
            activeforeground=C["fg_btn"],
            relief="flat", bd=2
        )
        for name in LANGUAGES:
            menu.add_command(
                label=f"{LANG_FLAGS.get(name, '')} {name}",
                command=lambda n=name: self._select_lang(var, n)
            )
        btn["menu"] = menu
        btn.pack(fill="x")

    def _make_btn(self, parent, text, bg, bg_hov, fg, cmd, side="left", padx_l=0):
        lbl = tk.Label(
            parent, text=text,
            font=get_ui_font(FS_BTN, "bold"),
            fg=fg, bg=bg,
            padx=BTN_PADX, pady=BTN_PADY,
            cursor="hand2"
        )
        lbl.pack(side=side, padx=(padx_l, 0))
        lbl.bind("<Button-1>", cmd)
        self._hover(lbl, bg, bg_hov)
        return lbl

    @staticmethod
    def _hover(widget, bg_normal, bg_hover):
        widget.bind("<Enter>", lambda e: widget.configure(bg=bg_hover))
        widget.bind("<Leave>", lambda e: widget.configure(bg=bg_normal))

    # ── Language helpers ──────────────────────────────────────────────────────

    def _select_lang(self, var, name):
        var.set(name)
        self._update_dir_label()
        self._refresh_status()

    def _swap_languages(self, _=None):
        s, t = self.src_lang.get(), self.tgt_lang.get()
        self.src_lang.set(t)
        self.tgt_lang.set(s)
        in_text  = self.input_text.get("1.0", "end-1c").strip()
        out_text = self.output_text.get("1.0", "end-1c").strip()
        if out_text and not out_text.startswith("["):
            self._set_input_text(out_text)
            self._set_output(in_text)
        self._update_dir_label()
        self._refresh_status()

    def _update_dir_label(self):
        s, t = self.src_lang.get(), self.tgt_lang.get()
        fs, ft = LANG_FLAGS.get(s, ""), LANG_FLAGS.get(t, "")
        self.dir_label.configure(text=f"{fs}→{ft}")

    # ── Translation ───────────────────────────────────────────────────────────

    def _on_enter(self, _):
        self.root.after(1, self._translate)
        return "break"

    def _translate(self):
        text = self.input_text.get("1.0", "end-1c").strip()
        if not text:
            self._set_status("Type something to translate.", ok=True)
            return

        src_code = LANGUAGES[self.src_lang.get()]
        tgt_code = LANGUAGES[self.tgt_lang.get()]

        if src_code == tgt_code:
            self._set_output(text)
            self._set_status("Same language – showing input as-is.", ok=True)
            return

        # Block if package not installed — no network calls
        if not _is_pair_installed(src_code, tgt_code):
            self._set_status(
                f"⚠ Package {src_code}→{tgt_code} missing. Run --install-packages",
                ok=False
            )
            return

        self._set_status(
            f"Translating {self.src_lang.get()[:3]}→{self.tgt_lang.get()[:3]} …",
            ok=True
        )
        self.translate_btn.configure(bg=C["fg_dim"])
        self.root.update_idletasks()

        def run():
            result = do_translate(text, src_code, tgt_code)
            self.root.after(0, lambda: self._on_done(result))

        threading.Thread(target=run, daemon=True).start()

    def _on_done(self, result):
        self._set_output(result)
        self.translate_btn.configure(bg=C["btn_primary"])
        src, tgt = self.src_lang.get(), self.tgt_lang.get()
        fs, ft = LANG_FLAGS.get(src, ""), LANG_FLAGS.get(tgt, "")
        if result.startswith("["):
            self._set_status(f"✗ {result[:60]}", ok=False)
        else:
            self._set_status(f"✓ {fs}{src[:3]}→{ft}{tgt[:3]}", ok=True)
        self._refresh_status()

    # ── Voice Input ───────────────────────────────────────────────────────────

    def _toggle_recording(self, _=None):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self._recording:
            return
        src_code = LANGUAGES[self.src_lang.get()]
        model = get_vosk_model(src_code)
        if model is None:
            model_name = VOSK_MODEL_NAMES.get(src_code, "?")
            self._set_status(f"⚠ Vosk model missing: {model_name}", ok=False)
            return

        self._recording = True
        self.mic_btn.configure(bg=C["btn_record_live"], text="🔴 STOP")
        self._set_status("🎙 Recording… tap STOP when done.", ok=True)
        self.input_text.delete("1.0", "end")

        self._recorder = VoiceRecorder(
            lang_code=src_code,
            on_result=self._voice_result,
            on_partial=self._voice_partial,
            on_error=self._voice_error,
        )
        self._recorder.start()

    def _stop_recording(self):
        if not self._recording:
            return
        self._recording = False
        if self._recorder:
            self._recorder.stop()
            self._recorder = None
        if self.voice_ok:
            self.mic_btn.configure(bg=C["btn_record"], text="🎙 SPEAK")
        self._set_status("Processing speech…", ok=True)

    def _voice_partial(self, text: str):
        self.root.after(0, lambda: self._set_input_text(text + " …"))

    def _voice_result(self, text: str):
        self.root.after(0, lambda: self._on_voice_result(text))

    def _on_voice_result(self, text: str):
        self._set_input_text(text)
        self._update_char_count()
        if text:
            short = text[:40] + ("…" if len(text) > 40 else "")
            self._set_status(f"✓ Got: '{short}'", ok=True)
            self._translate()
        else:
            self._set_status("No speech detected – try again.", ok=True)

    def _voice_error(self, msg: str):
        self.root.after(0, lambda: self._set_status(f"🎙 {msg[:70]}", ok=False))
        if self.voice_ok:
            self.root.after(0, lambda: self.mic_btn.configure(
                bg=C["btn_record"], text="🎙 SPEAK"
            ))
        self._recording = False

    def _set_input_text(self, text: str):
        text = _clean_indic_orthography(text)
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", text)

    # ── Speech Output ─────────────────────────────────────────────────────────

    def _speak_output(self):
        text = self.output_text.get("1.0", "end-1c").strip()
        if not text or text.startswith("["):
            self._set_status("Nothing to speak.", ok=True)
            return
        tgt_code = LANGUAGES[self.tgt_lang.get()]
        self._set_status("🔊 Speaking…", ok=True)
        threading.Thread(
            target=lambda: (
                speak(text, tgt_code),
                self.root.after(0, lambda: self._set_status("✓ Done.", ok=True))
            ),
            daemon=True
        ).start()

    # ── Misc ──────────────────────────────────────────────────────────────────

    def _clear(self):
        self.input_text.delete("1.0", "end")
        self._set_output("")
        self._set_status("Cleared.", ok=True)
        self._update_char_count()
        self.input_text.focus_set()

    def _set_output(self, text: str):
        text = _clean_indic_orthography(text)
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text)
        self.output_text.configure(state="disabled")

    def _set_status(self, msg: str, ok: bool = True):
        self.status_lbl.configure(text=msg[:72], fg=C["ok"] if ok else C["err"])

    def _update_char_count(self, _=None):
        n = len(self.input_text.get("1.0", "end-1c"))
        self.char_count.configure(text=f"{n}c")

    def _refresh_status(self):
        src_code = LANGUAGES[self.src_lang.get()]
        tgt_code = LANGUAGES[self.tgt_lang.get()]
        if src_code == tgt_code:
            self._set_status("Same language selected.", ok=True)
            self.model_lbl.configure(text="")
            return
        cached    = (src_code, tgt_code) in _translation_cache
        installed = _is_pair_installed(src_code, tgt_code)
        if cached:
            label, color = "✓ cached", C["ok"]
        elif installed:
            label, color = "✓ installed", C["ok"]
        else:
            label, color = "⚠ not installed", C["warn"]
        self._set_status(
            "Ready · 100% Offline" if installed else "⚠ Run --install-packages",
            ok=installed
        )
        self.model_lbl.configure(text=label, fg=color)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = OffLingoApp(root)
    root.mainloop()
