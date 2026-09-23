#!/usr/bin/env python3
"""
Unit tests for Gemma 4 reasoning delimiter auto-detection and Axolotl input_output segment formatting.
Verifies compliance with arXiv:2605.21127v1 (ThinkPack empty reasoning trace masking).
"""

import sys
from pathlib import Path

# Add src-train and src-eval to path
SRC_TRAIN_DIR = Path(__file__).resolve().parent.parent / "src-train"
SRC_EVAL_DIR = Path(__file__).resolve().parent.parent / "src-eval"
sys.path.insert(0, str(SRC_TRAIN_DIR))
sys.path.insert(0, str(SRC_EVAL_DIR))

from prepare_data import detect_reasoning_delimiters, format_input_output_segments
from evaluation import classify_reasoning_trace, compute_structural_reasoning_metrics, count_tokens


class MockTokenizerWithGemma4Template:
    """Simulates a tokenizer loaded with Gemma 4's canonical chat template."""
    def __init__(self):
        self.chat_template = (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<start_of_turn>model\n"
            "{% if enable_thinking %}<|channel>thought\n{{ message['thought'] }}<channel|>\n{% endif %}"
            "{{ message['content'] }}<turn|>\n"
            "{% endif %}"
            "{% endfor %}"
        )

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        prompt = ""
        for msg in conversation:
            if msg["role"] in ("system", "user"):
                prompt += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
        if add_generation_prompt:
            prompt += "<start_of_turn>model\n"
            if enable_thinking:
                prompt += "<|channel>thought\n"
        return prompt


class MockTokenizerWithThinkTags:
    """Simulates a tokenizer using XML-style <think> tags."""
    def __init__(self):
        self.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% endif %}"
            "{% if message['role'] == 'assistant' %}<|im_start|>assistant\n<think>{{ message['thought'] }}</think>\n{{ message['content'] }}<|im_end|>\n{% endif %}"
            "{% endfor %}"
        )


def test_detect_reasoning_delimiters_from_gemma4_template():
    """Verify that detect_reasoning_delimiters correctly extracts Gemma 4's thought channel tokens from template."""
    tokenizer = MockTokenizerWithGemma4Template()
    open_tag, close_tag = detect_reasoning_delimiters(tokenizer)
    assert open_tag == "<|channel>thought", f"Expected '<|channel>thought', got {open_tag}"
    assert close_tag == "<channel|>", f"Expected '<channel|>', got {close_tag}"


def test_detect_reasoning_delimiters_fallback_when_none():
    """Verify that detect_reasoning_delimiters falls back to Gemma 4 native tokens when tokenizer is None."""
    open_tag, close_tag = detect_reasoning_delimiters(None)
    assert open_tag == "<|channel>thought", f"Expected fallback '<|channel>thought', got {open_tag}"
    assert close_tag == "<channel|>", f"Expected fallback '<channel|>', got {close_tag}"


def test_detect_reasoning_delimiters_think_tags():
    """Verify that detect_reasoning_delimiters detects standard <think> tags when present."""
    tokenizer = MockTokenizerWithThinkTags()
    open_tag, close_tag = detect_reasoning_delimiters(tokenizer)
    assert open_tag == "<think>", f"Expected '<think>', got {open_tag}"
    assert close_tag == "</think>", f"Expected '</think>', got {close_tag}"


def test_format_input_output_segments_structure():
    """Verify that input_output segments mask the prompt and empty thought block, and train on the target response."""
    system_prompt = "Du bist ein Übersetzer für Leichte Sprache."
    user_prompt = "Übersetze: Der Antragsteller muss ein Formular ausfüllen."
    assistant_text = "Du musst ein Formular ausfüllen."

    segments = format_input_output_segments(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        assistant_text=assistant_text,
        tokenizer=None,
        open_tag="<|channel>thought",
        close_tag="<channel|>",
        turn_token="<turn|>",
    )

    assert len(segments) == 2, f"Expected exactly 2 segments, got {len(segments)}"

    seg0 = segments[0]
    seg1 = segments[1]

    # Segment 0: MUST be masked (label: False)
    assert seg0["label"] is False, "Segment 0 must have label: False (masked from loss)"
    assert system_prompt in seg0["text"], "Segment 0 must contain system prompt"
    assert user_prompt in seg0["text"], "Segment 0 must contain user prompt"
    assert "<|channel>thought" in seg0["text"], "Segment 0 must contain opening thought tag"
    assert "<channel|>" in seg0["text"], "Segment 0 must contain closing thought tag"
    assert assistant_text not in seg0["text"], "Segment 0 must NOT contain the target response text"

    # Segment 1: MUST be trained (label: True)
    assert seg1["label"] is True, "Segment 1 must have label: True (supervised loss)"
    assert assistant_text in seg1["text"], "Segment 1 must contain the target response text"
    assert "<turn|>" in seg1["text"], "Segment 1 must contain the turn ending marker"
    assert "<|channel>thought" not in seg1["text"], "Segment 1 must NOT contain thought tags"
    assert system_prompt not in seg1["text"], "Segment 1 must NOT contain system prompt"


def test_format_input_output_segments_with_tokenizer():
    """Verify segment generation when tokenizer.apply_chat_template is used."""
    tokenizer = MockTokenizerWithGemma4Template()
    system_prompt = "System Prompt"
    user_prompt = "User Prompt"
    assistant_text = "Target Translation"

    segments = format_input_output_segments(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        assistant_text=assistant_text,
        tokenizer=tokenizer,
        open_tag="<|channel>thought",
        close_tag="<channel|>",
        turn_token="<turn|>",
    )

    assert len(segments) == 2
    assert segments[0]["label"] is False
    assert segments[1]["label"] is True
    assert "<|channel>thought" in segments[0]["text"]
    assert segments[1]["text"] == "Target Translation<turn|>\n"


class MockTokenizerWithFullGemma4Template:
    """Simulates Gemma 4 tokenizer where apply_chat_template(add_generation_prompt=True) produces <|channel>thought\n<channel|>."""
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        return "<bos><|turn>system\nSystem<turn|>\n<|turn>user\nUser<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def test_format_input_output_segments_with_full_delimiters_template():
    """Verify segment generation does not duplicate empty thought tags when template already emits them."""
    tokenizer = MockTokenizerWithFullGemma4Template()
    segments = format_input_output_segments(
        system_prompt="System",
        user_prompt="User",
        assistant_text="Target",
        tokenizer=tokenizer,
        open_tag="<|channel>thought",
        close_tag="<channel|>",
        turn_token="<turn|>",
    )
    assert len(segments) == 2
    assert segments[0]["text"].count("<|channel>thought") == 1
    assert segments[0]["text"].count("<channel|>") == 1
    assert segments[0]["label"] is False
    assert segments[1]["label"] is True
    assert segments[1]["text"] == "Target<turn|>\n"


def test_classify_reasoning_trace_gemma4():
    """Verify classification of Gemma 4 thought channel traces according to arXiv:2605.21127v1."""
    # Valid: open and close present, non-empty thought content
    valid_sample = "<|channel>thought\nSchritt 1: Satz analysieren.\nSchritt 2: Vereinfachen.\n<channel|>\nDas ist eine einfache Übersetzung."
    assert classify_reasoning_trace(valid_sample) == "valid"

    # Empty: open and close present, empty or whitespace-only thought
    empty_sample1 = "<|channel>thought\n<channel|>\nDas ist eine einfache Übersetzung."
    assert classify_reasoning_trace(empty_sample1) == "empty"

    empty_sample2 = "<|channel>thought\n   \n\t  <channel|>\nDas ist eine einfache Übersetzung."
    assert classify_reasoning_trace(empty_sample2) == "empty"

    # Truncated: open tag present, but closing tag missing (e.g. hit max_new_tokens during reasoning)
    truncated_sample = "<|channel>thought\nSchritt 1: Satz analysieren und..."
    assert classify_reasoning_trace(truncated_sample) == "truncated"

    # Missing: no thought tags present (direct answer / collapsed reasoning)
    missing_sample = "Das ist eine direkte Übersetzung ohne Denkprozess."
    assert classify_reasoning_trace(missing_sample) == "missing"

    # Edge cases
    assert classify_reasoning_trace("") == "missing"
    assert classify_reasoning_trace(None) == "missing"


def test_classify_reasoning_trace_alternative_delimiters():
    """Verify classification with XML <think> and <|thought|> tags."""
    # XML <think> tags
    assert classify_reasoning_trace("<think>Schrittweise überlegen</think>Antwort") == "valid"
    assert classify_reasoning_trace("<think></think>Antwort") == "empty"
    assert classify_reasoning_trace("<think>Unfertig...") == "truncated"

    # Alternative <|thought|> tags
    assert classify_reasoning_trace("<|thought|>Überlegung</thought>Antwort") == "valid"


def test_compute_structural_reasoning_metrics():
    """Verify computation of ThinkPack VR, ER, MR, TR rates on a batch of outputs."""
    # 4 samples: 1 valid, 1 empty, 1 missing, 1 truncated
    batch = [
        {"text": "<|channel>thought\nSchritt 1...\n<channel|>\nAntwort"},    # valid
        {"text": "<|channel>thought\n<channel|>\nAntwort"},                  # empty
        {"text": "Direkte Antwort ohne Denken"},                              # missing
        {"text": "<|channel>thought\nAbgebrochener Gedankengang..."},         # truncated
    ]

    metrics = compute_structural_reasoning_metrics(batch)
    assert metrics["total"] == 4
    assert metrics["valid_count"] == 1
    assert metrics["empty_count"] == 1
    assert metrics["missing_count"] == 1
    assert metrics["truncated_count"] == 1

    assert metrics["valid_reasoning_rate"] == 25.0
    assert metrics["empty_reasoning_rate"] == 25.0
    assert metrics["missing_reasoning_rate"] == 25.0
    assert metrics["truncated_reasoning_rate"] == 25.0

    # Edge case: empty batch
    empty_metrics = compute_structural_reasoning_metrics([])
    assert empty_metrics["total"] == 0
    assert empty_metrics["valid_reasoning_rate"] == 0.0


def test_count_tokens():
    """Verify that count_tokens properly handles edge cases and mock tokenizers."""
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    tok = DummyTokenizer()
    assert count_tokens("Hallo Welt", tok) == 2
    assert count_tokens("", tok) == 0
    assert count_tokens("   ", tok) == 0
    assert count_tokens(None, tok) == 0
    assert count_tokens("Hallo", None) == 0


def test_token_counting_logic():
    """Verify reasoning vs answer token counting logic."""
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    tok = DummyTokenizer()
    reasoning = "Schritt 1: Denken und analysieren"  # 5 tokens
    answer = "Das ist die Antwort."                 # 4 tokens

    reasoning_tokens = count_tokens(reasoning, tok)
    answer_tokens = count_tokens(answer, tok)
    total_tokens = reasoning_tokens + answer_tokens

    assert reasoning_tokens == 5
    assert answer_tokens == 4
    assert total_tokens == 9


if __name__ == "__main__":
    test_detect_reasoning_delimiters_from_gemma4_template()
    test_detect_reasoning_delimiters_fallback_when_none()
    test_detect_reasoning_delimiters_think_tags()
    test_format_input_output_segments_structure()
    test_format_input_output_segments_with_tokenizer()
    test_format_input_output_segments_with_full_delimiters_template()
    test_classify_reasoning_trace_gemma4()
    test_classify_reasoning_trace_alternative_delimiters()
    test_compute_structural_reasoning_metrics()
    test_count_tokens()
    test_token_counting_logic()
    print("[SUCCESS] All Gemma 4 reasoning delimiter, input_output segment & structural reasoning metric tests passed!")
