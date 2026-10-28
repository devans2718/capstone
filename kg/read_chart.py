#!/usr/bin/env python3
"""Send a chart image to Claude and save the prompt and the reply, verbatim.

HOW TO RUN (from the repo root, in WSL)
    .venv-app/bin/python kg/read_chart.py CHART.png
    Writes CHART-claude-reading.md beside the chart. Needs ANTHROPIC_API_KEY,
    as kg/graphrag.py. Same model as graphrag.py.

WHY (Challenge 4 AI Design, multimodal)
    The community x /t/-/d/ chart is read by the model from the image alone,
    no rows: what a non-linguist would be told the chart says. The reply is
    then checked by hand against kg/communities/summary.json, so the image is
    the only input and any mismatch is the model's reading, not the data's.
"""

from __future__ import annotations

import base64
import sys
from datetime import datetime
from pathlib import Path

import anthropic

from graphrag import MODEL, load_dotenv

PROMPT = """This chart comes from a study of the "flap" in American English: \
the quick tap in the middle of words like "water" and "ladder", where a /t/ \
or a /d/ in the dictionary form comes out sounding the same. Each row is a \
group of flap sites that a community-detection algorithm found by looking \
only at the sounds around the flap (the vowel before it and the sound after \
it). The algorithm never saw whether each site was a /t/ or a /d/.

Explain the chart in plain language for someone who is not a linguist: what \
it shows, how to read it, and what it does and doesn't tell us about whether \
the surrounding sounds line up with /t/ versus /d/. Quote the numbers you \
rely on from the chart. At most about 250 words."""


def main() -> None:
    chart = Path(sys.argv[1])
    load_dotenv()
    response = anthropic.Anthropic().messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.standard_b64encode(chart.read_bytes()).decode()}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    reply = "".join(b.text for b in response.content if b.type == "text")
    out = chart.with_name(chart.stem + "-claude-reading.md")
    out.write_text(
        f"# Claude's reading of {chart.name}\n\n"
        f"Model: {MODEL}. Sent {datetime.now():%Y-%m-%d %H:%M} by "
        f"kg/read_chart.py; the image was the only input besides the prompt.\n\n"
        f"## Prompt (verbatim)\n\n{PROMPT}\n\n## Reply (verbatim)\n\n{reply}\n",
        encoding="utf-8", newline="\n")
    print(reply)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
