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
  ✦ NLLB-200 (distilled 600M) for Telugu/Tamil/Kannada  (direct, no pivot)
  ✦ Argos Translate for en/hi/fr/zh/es/de (unchanged)
  ✦ Context-aware chunking for better sentence/paragraph translation

Pre-install packages (run ONCE while online):
    cd /home/pns/Desktop/Trasnlator
    ./lingo_env/bin/pip install ctranslate2 sentencepiece huggingface_hub
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
from pathlib import Path

# Only enforce strict offline mode if we are NOT running the installer
if "--install-packages" not in sys.argv:
    os.environ["ARGOS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Block ALL outgoing network connections (localhost still allowed).
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
#  NLLB CONFIG (Telugu / Tamil / Kannada)
# ══════════════════════════════════════════════════════════════════════════════

# Languages handled by NLLB instead of Argos
DRAVIDIAN_LANGUAGES = {"te", "ta", "kn"}

# Pre-converted CTranslate2 int8 build of NLLB-200 distilled 600M (~600 MB,
# ~0.7 GB RAM) — fits on a 2 GB Raspberry Pi. Tokenizer files are included.
NLLB_REPO = "Code-Dev/nllb-200-distilled-600M-ct2-int8"
NLLB_DIR  = Path.home() / "nllb-ct2"
NLLB_MODEL = str(NLLB_DIR)   # key used for the in-memory cache
NLLB_SPM   = NLLB_DIR / "sentencepiece.bpe.model"   # light tokenizer (no transformers)
NLLB_SPM_FALLBACK_REPO = "olob0/nllb-200-distilled-600M-ct2-int8_float16"

# App language code -> NLLB (FLORES-200) code
NLLB_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "kn": "kan_Knda",
}


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

    # ── 3. Download NLLB-200 for Telugu, Tamil, Kannada ──
    print("\n--- 3. Downloading NLLB-200 int8 (Telugu, Tamil, Kannada) ---")
    print(f"     (about 600 MB → {NLLB_DIR})")
    try:
        from huggingface_hub import snapshot_download, hf_hub_download

        # Files are written straight to disk — no big in-RAM model rebuild.
        snapshot_download(repo_id=NLLB_REPO, local_dir=str(NLLB_DIR))
        print(f"  [OK]   {NLLB_REPO}")

        # The sentencepiece tokenizer file is required; fetch it if missing.
        if not NLLB_SPM.exists():
            print("  Fetching sentencepiece.bpe.model ...")
            hf_hub_download(repo_id=NLLB_SPM_FALLBACK_REPO,
                            filename="sentencepiece.bpe.model",
                            local_dir=str(NLLB_DIR))
        print(f"  [OK]   tokenizer: {NLLB_SPM.exists()}")

        # Warm-up runs in a FRESH process so Stanza/Argos memory used above
        # doesn't push the 2 GB Pi into the out-of-memory killer.
        print("  Warming up NLLB for te / ta / kn (separate process) ...")
        subprocess.run([sys.executable, os.path.abspath(__file__), "--nllb-test"])
    except ImportError as e:
        print(f"  [SKIP] missing package ({e}). "
              "Run: pip install ctranslate2 sentencepiece huggingface_hub")
    except Exception as e:
        print(f"  [FAIL] {NLLB_REPO}: {e}")

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

# ── Replace Stanza sentence splitter with offline regex splitter ──
if "--install-packages" not in sys.argv:
    import argostranslate.sbd as _sbd

    _ABBREV_PATTERN = (
        r'(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Gen|Gov|Sgt|Corp|Inc|Ltd|Rev|'
        r'Vol|Fig|Dept|Est|Approx|Rep|Sec|Capt|Col|Cmdr|Lt|Maj|'
        r'vs|etc|approx|viz|'
        r'i\.e|e\.g|a\.m|p\.m|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    )

    def _robust_sentence_split(text):
        """Split text into individual sentences, respecting abbreviations,
        decimal numbers, ellipses, and single-letter initials like U.S.A."""
        if not text or not text.strip():
            return [text.strip()] if text and text.strip() else []

        text = text.strip()

        shields = {}
        _cnt = [0]

        def _shield(m):
            key = f'\x00S{_cnt[0]}\x00'
            shields[key] = m.group(0)
            _cnt[0] += 1
            return key

        protected = text
        protected = _re.sub(
            r'\b' + _ABBREV_PATTERN + r'\.\s',
            _shield, protected, flags=_re.IGNORECASE
        )
        protected = _re.sub(r'\d+\.\d+', _shield, protected)
        protected = _re.sub(r'\.{2,}', _shield, protected)
        protected = _re.sub(r'…', _shield, protected)
        protected = _re.sub(r'(?:[A-Z]\.){2,}', _shield, protected)
        protected = _re.sub(r'https?://\S+', _shield, protected)
        protected = _re.sub(r'www\.\S+', _shield, protected)
        protected = _re.sub(r'\b\d{1,2}\.\d{2}\b', _shield, protected)

        parts = _re.split(
            r'(?<=[.!?])\s+'
            r'(?=['
            r'A-Z'
            r'\u0900-\u097F'
            r'\u4e00-\u9fff'
            r'\u0C00-\u0C7F'
            r'\u0B80-\u0BFF'
            r'\u0C80-\u0CFF'
            r'0-9'
            r'"\u201C\u201D'
            r"'"
            r'])',
            protected
        )

        result = []
        for part in parts:
            for key, val in shields.items():
                part = part.replace(key, val)
            part = part.strip()
            if part:
                result.append(part)

        return result if result else [text]

    def _group_sentences(sentences, max_chunk_chars=450):
        """Group sentences into translation-sized chunks to preserve context."""
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        chunks = []
        current_group = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > max_chunk_chars and current_group:
                chunks.append(' '.join(current_group))
                current_group = [sent]
                current_len = sent_len
            else:
                current_group.append(sent)
                current_len += sent_len + 1

        if current_group:
            chunks.append(' '.join(current_group))

        return chunks if chunks else sentences

    def _offline_split(text, lang_code=None):
        """Smart offline text chunking for Argos translation."""
        if not text or not text.strip():
            return []

        text = text.strip()

        if len(text) < 500:
            return [text]

        paragraphs = _re.split(r'\n\s*\n', text)

        all_chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) < 500:
                all_chunks.append(para)
            else:
                sentences = _robust_sentence_split(para)
                grouped = _group_sentences(sentences, max_chunk_chars=450)
                all_chunks.extend(grouped)

        return all_chunks if all_chunks else [text]

    if hasattr(_sbd, 'detect_sentences'):
        _sbd.detect_sentences = _offline_split
    if hasattr(_sbd, 'split_sentences'):
        _sbd.split_sentences = _offline_split

# NLLB-200 (Hugging Face transformers) for Telugu, Tamil, Kannada
try:
    import ctranslate2
    import sentencepiece as spm
    NLLB_AVAILABLE = True
except ImportError:
    NLLB_AVAILABLE = False
    print("[INFO] ctranslate2/sentencepiece not found – Telugu/Tamil/Kannada translation disabled.")
    print("       Run: pip install ctranslate2 sentencepiece")

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
FS_TITLE    = 9
FS_LABEL    = 8
FS_LANG     = 10
FS_TEXT     = 9
FS_BTN      = 8
FS_STATUS   = 7
FS_CHAR     = 7
FS_DIR      = 7
INPUT_ROWS  = 3
OUTPUT_ROWS = 3
BTN_PADY    = 3
BTN_PADX    = 10


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

VOSK_MODEL_PRIORITY = {
    "en": ["vosk-model-en-us-0.22-lgraph",  "vosk-model-en-us-0.22",  "vosk-model-small-en-us-0.15"],
    "hi": ["vosk-model-hi-0.22",             "vosk-model-small-hi-0.22"],
    "fr": ["vosk-model-fr-0.22-lgraph",      "vosk-model-small-fr-0.22"],
    "zh": ["vosk-model-cn-0.22",             "vosk-model-small-cn-0.22"],
    "es": ["vosk-model-es-0.42",             "vosk-model-small-es-0.42"],
    "de": ["vosk-model-de-0.21",             "vosk-model-small-de-0.15"],
    "te": ["vosk-model-small-te-0.42"],
    # NOTE: No Vosk models exist for Tamil (ta) or Kannada (kn).
}

VOSK_MODEL_NAMES = {k: v[0] for k, v in VOSK_MODEL_PRIORITY.items()}

VOSK_MODELS_DIR = Path.home() / "vosk-models"


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT PRE / POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_text(text: str) -> str:
    """Normalise whitespace and clean input before translation."""
    if not text:
        return text
    text = _re.sub(r'[ \t]+', ' ', text)
    text = _re.sub(r'\n{3,}', '\n\n', text)
    text = _re.sub(r'([.!?])([A-Z\u0900-\u097F\u4e00-\u9fff\u0C00-\u0C7F\u0B80-\u0BFF\u0C80-\u0CFF])', r'\1 \2', text)
    return text.strip()


def _postprocess_text(text: str) -> str:
    """Clean up common translation artifacts."""
    if not text:
        return text
    text = _re.sub(r'  +', ' ', text)
    text = _re.sub(r'\s+([.!?,;:。！？，；：])', r'\1', text)
    text = _re.sub(
        r'([.!?,;:])([A-Za-z\u0900-\u097F\u4e00-\u9fff\u0C00-\u0C7F\u0B80-\u0BFF\u0C80-\u0CFF])',
        r'\1 \2',
        text
    )
    text = _re.sub(r'([.!?])\1{2,}', r'\1\1\1', text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSLATION  (NO network calls — purely local)
# ══════════════════════════════════════════════════════════════════════════════

_translation_cache: dict = {}

# ── Argos Translate helpers (for en/hi/fr/zh/es/de) ──────────────────────────

def _nllb_is_cached() -> bool:
    """Check that the downloaded NLLB int8 model exists on disk."""
    return (NLLB_DIR / "model.bin").exists() and NLLB_SPM.exists()


def _is_pair_installed(src_code: str, tgt_code: str) -> bool:
    """Check if a translation pair is ready (Argos or NLLB)."""
    if src_code in DRAVIDIAN_LANGUAGES or tgt_code in DRAVIDIAN_LANGUAGES:
        return NLLB_AVAILABLE and (NLLB_MODEL in _nllb_cache or _nllb_is_cached())

    installed = argostranslate.translate.get_installed_languages()
    src_lang = next((l for l in installed if l.code == src_code), None)
    tgt_lang = next((l for l in installed if l.code == tgt_code), None)
    if src_lang is None or tgt_lang is None:
        return False
    return src_lang.get_translation(tgt_lang) is not None


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


# ── NLLB helpers (for te/ta/kn) ──────────────────────────────────────────────

_nllb_cache: dict = {}  # {model_name: (tokenizer, model)}
_nllb_lock = threading.Lock()


def _load_nllb():
    """Load the int8 NLLB model from disk (no network). Loaded lazily, only
    when a Telugu/Tamil/Kannada translation is requested."""
    if NLLB_MODEL in _nllb_cache:
        return _nllb_cache[NLLB_MODEL]
    if not NLLB_AVAILABLE:
        return None
    with _nllb_lock:
        if NLLB_MODEL in _nllb_cache:
            return _nllb_cache[NLLB_MODEL]
        try:
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.Load(str(NLLB_SPM))
            translator = ctranslate2.Translator(
                str(NLLB_DIR),
                device="cpu",
                compute_type="int8",
                inter_threads=1,
                intra_threads=4,
            )
            _nllb_cache[NLLB_MODEL] = (tokenizer, translator)
            return (tokenizer, translator)
        except Exception as e:
            print(f"[NLLB load error for {NLLB_DIR}]: {e}")
            return None


def _unload_nllb():
    """Free NLLB memory (called when no South Indian language is selected)."""
    with _nllb_lock:
        _nllb_cache.pop(NLLB_MODEL, None)
    import gc
    gc.collect()


def _nllb_translate_chunk(tokenizer, translator, chunk: str, src_code: str, tgt_code: str) -> str:
    """Translate one sentence/chunk with NLLB (CTranslate2)."""
    chunk = chunk.strip()
    if not chunk:
        return ""

    # NLLB input format: [src_lang_code] + sentencepiece pieces + [</s>]
    pieces = tokenizer.encode(chunk, out_type=str)[:250]
    tokens = [NLLB_CODES[src_code]] + pieces + ["</s>"]
    max_new = min(400, max(32, int(len(tokens) * 2.5) + 16))

    results = translator.translate_batch(
        [tokens],
        target_prefix=[[NLLB_CODES[tgt_code]]],
        beam_size=4,
        max_decoding_length=max_new,
        no_repeat_ngram_size=4,
        repetition_penalty=1.1,
    )
    hyp = results[0].hypotheses[0]
    if hyp and hyp[0] == NLLB_CODES[tgt_code]:
        hyp = hyp[1:]
    hyp = [t for t in hyp if t not in ("</s>", "<s>", "<pad>")]
    result = tokenizer.decode(hyp).strip()
    print(f"[NLLB {src_code}→{tgt_code}] '{chunk[:60]}' -> '{result[:60]}'")
    return result


def _nllb_translate(text: str, src_code: str, tgt_code: str) -> str:
    """Translate directly between any two supported languages with NLLB.

    NLLB is trained on sentence-level data, so text is split into sentences
    (short sentences are grouped up to ~200 chars) and paragraphs are kept.
    """
    pair = _load_nllb()
    if pair is None:
        return (
            f"[NLLB model not found in {NLLB_DIR}]\n"
            f"Run: ./lingo_env/bin/python test2.py --install-packages"
        )
    tokenizer, model = pair

    text = _preprocess_text(text)

    def _translate_block(block: str) -> str:
        if len(block) <= 200:
            return _nllb_translate_chunk(tokenizer, model, block, src_code, tgt_code)
        sentences = _robust_sentence_split(block)
        chunks = _group_sentences(sentences, max_chunk_chars=200)
        out = []
        for ch in chunks:
            r = _nllb_translate_chunk(tokenizer, model, ch, src_code, tgt_code)
            out.append(r if r else ch)
        return ' '.join(out)

    try:
        parts = _re.split(r'(\n\s*\n)', text)
        translated = []
        for part in parts:
            if not part.strip():
                translated.append(part)
                continue
            # keep single line breaks as separate blocks
            lines = [ln for ln in part.split('\n') if ln.strip()]
            translated.append('\n'.join(_translate_block(ln.strip()) for ln in lines))
        final = ''.join(translated)
    except Exception as e:
        return f"[NLLB error: {e}]"

    if final.strip():
        return _postprocess_text(final)
    return "[No result — model returned empty output]"


# ── Unified translate function ───────────────────────────────────────────────

def do_translate(text: str, src_code: str, tgt_code: str) -> str:
    """Translate offline — zero network calls."""
    if src_code == tgt_code:
        return text

    try:
        # Any pair involving Telugu / Tamil / Kannada → NLLB, direct (no pivot)
        if src_code in DRAVIDIAN_LANGUAGES or tgt_code in DRAVIDIAN_LANGUAGES:
            return _nllb_translate(text, src_code, tgt_code)

        # Everything else → Argos (unchanged)
        return _argos_translate(text, src_code, tgt_code)

    except Exception as e:
        import traceback
        traceback.print_exc()   # full details go to the terminal
        return f"[Translation error: {type(e).__name__}: {e}]"


def _argos_translate(text: str, src_code: str, tgt_code: str) -> str:
    """Translate using Argos Translate with context-aware chunking."""
    t = _get_translation_obj(src_code, tgt_code)
    if t is None:
        return (
            f"[Package not installed: {src_code}→{tgt_code}]\n"
            f"Run: ./lingo_env/bin/python test2.py --install-packages"
        )

    text = _preprocess_text(text)

    if len(text) < 500:
        result = t.translate(text)
        return _postprocess_text(result.strip()) if result else "[No result]"

    paragraphs = _re.split(r'(\n\s*\n)', text)
    translated_parts = []

    for part in paragraphs:
        if not part.strip():
            translated_parts.append(part)
            continue

        if len(part) < 500:
            result = t.translate(part)
            translated_parts.append(result.strip() if result else part)
        else:
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
_vosk_active_model_name: dict = {}


def get_vosk_model(lang_code: str):
    """Load the best available Vosk model for a language."""
    if not VOSK_AVAILABLE:
        return None
    if lang_code in _vosk_model_cache:
        return _vosk_model_cache[lang_code]

    candidates = VOSK_MODEL_PRIORITY.get(lang_code, [])
    if not candidates:
        return None

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
    """Normalise audio volume so quiet speech is boosted for better recognition."""
    n_samples = len(raw_bytes) // 2
    if n_samples == 0:
        return raw_bytes

    samples = struct.unpack(f'<{n_samples}h', raw_bytes)

    sum_sq = sum(s * s for s in samples)
    rms = math.sqrt(sum_sq / n_samples) if n_samples > 0 else 0.0

    if rms < 10.0:
        return raw_bytes

    gain = min(target_rms / rms, 10.0)

    if 0.8 <= gain <= 1.2:
        return raw_bytes

    boosted = []
    for s in samples:
        v = int(s * gain)
        v = max(-32768, min(32767, v))
        boosted.append(v)

    return struct.pack(f'<{n_samples}h', *boosted)


def _filter_by_confidence(result_json: dict, min_conf: float = 0.45) -> str:
    """Filter recognised words by confidence score."""
    words = result_json.get("result", [])
    if words:
        kept = [w["word"] for w in words if w.get("conf", 1.0) >= min_conf]
        if kept:
            return " ".join(kept)

    return result_json.get("text", "").strip()


class VoiceRecorder:
    SAMPLERATE = 16000
    CHANNELS   = 1
    DTYPE      = "int16"
    BLOCKSIZE  = 4000

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

                    data = _normalise_audio(data)

                    if rec.AcceptWaveform(data):
                        r = json.loads(rec.Result())
                        word = _filter_by_confidence(r, self.MIN_CONFIDENCE)
                        if word:
                            full_text.append(word)
                            self.on_partial(" ".join(full_text))
                    else:
                        p = json.loads(rec.PartialResult())
                        partial = p.get("partial", "").strip()
                        if partial:
                            self.on_partial(" ".join(full_text + [partial]))

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

        row1 = tk.Frame(main, bg=C["bg"])
        row1.pack(fill="x", pady=(0, 2))

        tk.Label(
            row1, text="◉ OffLingo",
            font=("TkFixedFont", FS_TITLE, "bold"),
            fg=C["fg"], bg=C["bg"]
        ).pack(side="left", padx=(0, 6))

        self._make_lang_selector(row1, "FROM", self.src_lang, side="left")

        swap = tk.Label(
            row1, text="⇄",
            font=("TkFixedFont", 11), fg=C["fg"], bg=C["btn_dark"],
            width=2, cursor="hand2", relief="flat", padx=4, pady=2
        )
        swap.pack(side="left", padx=3)
        swap.bind("<Button-1>", self._swap_languages)
        self._hover(swap, C["btn_dark"], C["btn_dark_hov"])

        self._make_lang_selector(row1, "TO", self.tgt_lang, side="left")

        exit_btn = tk.Label(
            row1, text="✖",
            font=("TkFixedFont", 11, "bold"), fg=C["err"], bg=C["bg"],
            cursor="hand2", padx=4
        )
        exit_btn.pack(side="right", padx=(4, 0))
        exit_btn.bind("<Button-1>", lambda e: self.root.destroy())

        min_btn = tk.Label(
            row1, text="—",
            font=("TkFixedFont", 11, "bold"), fg=C["fg"], bg=C["bg"],
            cursor="hand2", padx=4
        )
        min_btn.pack(side="right", padx=(4, 4))
        min_btn.bind("<Button-1>", lambda e: self.root.iconify())

        self.dir_label = tk.Label(
            row1, text="",
            font=("TkFixedFont", FS_DIR),
            fg=C["fg_dim"], bg=C["bg"]
        )
        self.dir_label.pack(side="right", padx=(4, 0))
        self._update_dir_label()

        tk.Frame(main, bg=C["sep"], height=1).pack(fill="x", pady=(1, 2))

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
            font=("TkFixedFont", FS_TEXT),
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
                font=("TkFixedFont", FS_BTN),
                fg=C["fg"], bg=C["btn_record"],
                padx=BTN_PADX, pady=BTN_PADY, cursor="hand2"
            )
            self.mic_btn.pack(side="left", padx=(3, 0))
            self.mic_btn.bind("<Button-1>", self._toggle_recording)
        else:
            tk.Label(
                btn_row, text="🎙 –",
                font=("TkFixedFont", FS_BTN, "italic"),
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
            font=("TkFixedFont", FS_CHAR),
            fg=C["fg_dim"], bg=C["bg"]
        )
        self.char_count.pack(side="right", padx=(0, 2))

        out_wrap = tk.Frame(
            main, bg=C["border"],
            highlightthickness=1,
            highlightbackground=C["border"]
        )
        out_wrap.pack(fill="both", expand=True, pady=(0, 1))
        self.output_text = tk.Text(
            out_wrap,
            height=OUTPUT_ROWS,
            font=("TkFixedFont", FS_TEXT),
            fg=C["fg"], bg=C["bg_output"],
            relief="flat", bd=3,
            wrap="word",
            state="disabled",
            cursor="arrow",
            highlightthickness=0
        )
        self.output_text.pack(fill="both", expand=True)

        st = tk.Frame(main, bg=C["bg"])
        st.pack(fill="x")
        self.status_lbl = tk.Label(
            st, text="Ready",
            font=("TkFixedFont", FS_STATUS),
            fg=C["fg_dim"], bg=C["bg"], anchor="w"
        )
        self.status_lbl.pack(side="left")
        self.model_lbl = tk.Label(
            st, text="",
            font=("TkFixedFont", FS_STATUS),
            fg=C["fg_dim"], bg=C["bg"], anchor="e"
        )
        self.model_lbl.pack(side="right")

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _make_lang_selector(self, parent, label_text, var, side="left"):
        container = tk.Frame(parent, bg=C["bg"])
        container.pack(side=side, fill="x", expand=True)

        tk.Label(
            container, text=label_text,
            font=("TkFixedFont", FS_LABEL, "bold"),
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
            font=("TkFixedFont", FS_LANG),
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
            font=("TkFixedFont", FS_LANG),
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
            font=("TkFixedFont", FS_BTN, "bold"),
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
        # Free the NLLB model (RAM) when no South Indian language is selected
        codes = {LANGUAGES[self.src_lang.get()], LANGUAGES[self.tgt_lang.get()]}
        if not (codes & DRAVIDIAN_LANGUAGES) and NLLB_MODEL in _nllb_cache:
            threading.Thread(target=_unload_nllb, daemon=True).start()

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
        uses_nllb = src_code in DRAVIDIAN_LANGUAGES or tgt_code in DRAVIDIAN_LANGUAGES
        cached    = (NLLB_MODEL in _nllb_cache) if uses_nllb else ((src_code, tgt_code) in _translation_cache)
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

if __name__ == "__main__" and "--nllb-test" in sys.argv:
    # Quick offline self-test of NLLB: ./lingo_env/bin/python test2.py --nllb-test
    for _code in ("te", "ta", "kn"):
        print(f"en→{_code}:", do_translate("Hello, how are you? I am going to the market.", "en", _code))
    print("te→en:", do_translate("మీరు ఎలా ఉన్నారు?", "te", "en"))
    sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = OffLingoApp(root)
    root.mainloop()
