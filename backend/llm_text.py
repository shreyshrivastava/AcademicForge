"""Post-processing for generated text."""
import re


def _clean_response(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.split(r"<end_of_turn>|<eos>|</s>", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = text.strip()

    # Strip raw headers like "Summary\n" or "Guidance\n" at the start
    text = re.sub(r"^(summary|guidance|research\s+plan):?\s*\n+", "", text, flags=re.IGNORECASE)

    # Process line-by-line to filter out echoes, planning, and constraint discussion
    cleaned_lines = []
    for line in text.splitlines():
        trimmed = line.strip()
        # Strip common robotic planning/thinking prefixes from the beginning of any line
        trimmed = re.sub(
            r"^(?:need\s+to\s+summarize|i['\u2019]ll\s+extract|we\s+need\s+to|so\s+the\s+paper|the\s+summary\s+should|i\s+need\s+to|let['\u2019]s\s+extract|we\s+are\s+asked|the\s+user\s+(?:wants|is\s+asking|requested)).*?:\s*",
            "",
            trimmed,
            flags=re.IGNORECASE
        )

        # Skip echoed prompt metadata blocks
        if re.match(r"^(?:the\s+)?paper\s+title:|^authors:|^abstract:|^evidence\s+\[\d+\]", trimmed, re.IGNORECASE):
            continue

        # Skip lines containing typical meta-commentary, reasoning, or prompt echo phrases
        if re.search(
            r"(?:the\s+summary\s+should|we\s+need\s+to|paper\s+says\s+that\s+i|so\s+the\s+paper|"
            r"no\s+bullet\s+points|just\s+what\s+the\s+paper\s+says|write\s+a\s+concise\s+summary|"
            r"avoid\s+hype|avoid\s+abstract-like|do\s+not\s+invent|do\s+not\s+use|do\s+not\s+give|"
            r"builder\s+recommendations|2-4\s+sentences|be\s+specific)",
            trimmed,
            re.IGNORECASE
        ):
            continue

        # Skip reasoning comments or constraint discussion
        if re.match(r"^(?:we\s+are\s+asked|the\s+user\s+(?:wants|is\s+asking|requested)|based\s+on\s+the\s+title\s+and\s+abstract|so\s+the\s+paper\s+proposes|the\s+summary\s+should\s+capture\s+that)", trimmed, re.IGNORECASE):
            continue
        # Skip short constraint notes / meta disclaimers
        if re.match(r"^\(that['\u2019]s|i\s+need\s+\d|i['\u2019]ll\s+mention|avoid\s+abstract-like|i['\u2019]ll\s+be", trimmed, re.IGNORECASE):
            continue

        # Strip transitional indicators like "I'll craft a summary: " or "Here is the summary: "
        trimmed = re.sub(
            r"^(?:i['\u2019]ll\s+craft\s+a\s+(?:concise\s+)?summary|i\s+will\s+craft\s+a\s+(?:concise\s+)?summary|here\s+is\s+a\s+(?:concise\s+)?summary|here(?:['\u2019]s|\s+is)\s+the\s+summary|summary):?\s*",
            "",
            trimmed,
            flags=re.IGNORECASE,
        )
        trimmed = re.sub(
            r"^(okay|sure|here(?:['\u2019]s|\s+is)),?\s+(?:a|an|the)?\s*",
            "",
            trimmed,
            flags=re.IGNORECASE,
        )
        if trimmed:
            cleaned_lines.append(trimmed)

    text = "\n".join(cleaned_lines).strip()

    heading_match = re.search(r"(\*\*[^*\n]+:\*\*|##\s+)", text)
    if heading_match and heading_match.start() < 280:
        preface = text[:heading_match.start()].lower()
        if any(marker in preface for marker in ("summary", "research plan", "formatted", "requested", "provided")):
            text = text[heading_match.start():].lstrip()
    fence_match = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return text.strip()
