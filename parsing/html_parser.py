from __future__ import annotations

import re
from pathlib import Path
from typing import Union


def parse_html(source: Union[str, bytes, Path]) -> str:
    """
    Extract clean readable text from HTML.
    Strips scripts, styles, navigation, ads, and excessive whitespace.
    """
    from bs4 import BeautifulSoup

    if isinstance(source, Path):
        raw = source.read_text(encoding="utf-8", errors="replace")
    elif isinstance(source, bytes):
        raw = source.decode("utf-8", errors="replace")
    else:
        raw = source

    soup = BeautifulSoup(raw, "lxml")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "form", "button", "iframe", "svg", "img"]):
        tag.decompose()

    # Prefer main content containers
    main = soup.find("main") or soup.find("article") or soup.find("div", {"id": "content"})
    root = main if main else soup.body or soup

    text = root.get_text(separator="\n", strip=True)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
