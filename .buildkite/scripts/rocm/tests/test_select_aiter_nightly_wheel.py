# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import datetime
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import select_aiter_nightly_wheel as selector  # noqa: E402

PROMOTED = "https://example.com/whl-multi-arch/amd-aiter/"
STAGING = "https://example.com/whl-multi-arch-staging/amd-aiter/"
IMAGE = ("10.0.0", "2.13", "cp312")

# Published 2026-10-27 by UFB's legacy pipeline.
LEGACY = (
    "amd_aiter-0.1.27+rocm10.0.0.torch2.13.ae984f5.d20261027"
    "-cp312-cp312-linux_x86_64.whl"
)
# Published 2026-10-28 by UFB's canonical pipeline, which tags wheels manylinux.
CANONICAL = (
    "amd_aiter-0.1.28+rocm10.0.0.torch2.13.1dd826c.d20261028"
    "-cp312-cp312-manylinux_2_28_x86_64.whl"
)


def _page(*filenames: str) -> str:
    links = "".join(
        f'<a href="{name.replace("+", "%2B")}">{name}</a>' for name in filenames
    )
    return f"<html><body>{links}</body></html>"


@pytest.fixture
def pages(monkeypatch):
    pages: dict[str, str] = {}

    def fetch(url):
        if url not in pages:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return pages[url]

    monkeypatch.setattr(selector, "fetch", fetch)
    monkeypatch.setattr(selector, "_image", lambda: IMAGE)
    return pages


def test_selects_manylinux_wheel_published_only_to_staging(pages):
    """The 2026-10-28 nightly was on staging only and tagged manylinux; it must
    win over the older wheel on the promoted index."""
    pages[PROMOTED] = _page(LEGACY)
    pages[STAGING] = _page(CANONICAL)

    wheel = selector.select([PROMOTED, STAGING])

    assert wheel.url == STAGING + CANONICAL.replace("+", "%2B")
    assert (wheel.commit, wheel.build_date) == (
        "1dd826c",
        datetime.date(2026, 10, 28),
    )


def test_wheel_on_both_indexes_comes_from_the_first(pages):
    pages[PROMOTED] = _page(CANONICAL)
    pages[STAGING] = _page(CANONICAL)

    assert selector.select([PROMOTED, STAGING]).url.startswith(PROMOTED)


def test_unreachable_index_fails_instead_of_reading_only_the_other(pages):
    """A missing index exits 1 (Buildkite retries) rather than silently testing
    an older wheel from the index that did load."""
    pages[PROMOTED] = _page(LEGACY)

    with pytest.raises(urllib.error.HTTPError):
        selector.select([PROMOTED, STAGING])


def test_retry_reinstalls_pinned_manylinux_wheel(pages):
    url = STAGING + CANONICAL.replace("+", "%2B")

    assert selector.select([], wheel_url=url).url == url
