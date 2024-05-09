import logging
from typing import Union

import trafilatura  # type: ignore

MAX_RESULT_LENGTH_CHAR = 1000 * 4 * 100  # roughly 100k tokens
cache: dict[str, str] = {}
logger = logging.getLogger(__name__)


def page_result(text: str, cursor: int, max_length: int) -> str:
    """Page through `text` and return a substring of `max_length` characters starting from `cursor`."""
    return text[cursor : cursor + max_length]


def get_url(url: str) -> str:
    """Fetch URL and return the contents as a string."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ValueError("Could not download article.")
    return (
        trafilatura.extract(downloaded, include_links=True, include_tables=True)
        or "<empty>"
    )


def scrape_text(url: str) -> Union[str, None]:
    try:
        if url in cache:
            return cache[url]
        page_contents = get_url(url)

        if len(page_contents) > MAX_RESULT_LENGTH_CHAR:
            page_contents = (
                page_result(page_contents, 0, MAX_RESULT_LENGTH_CHAR)
                + " ... <truncated>"
            )

        cache[url] = page_contents
        if len(cache) > 100:
            oldest_url = next(iter(cache))
            del cache[oldest_url]
        return page_contents
    except Exception as e:
        logger.error(f"Failed to read article from [{url}].")
        return None
