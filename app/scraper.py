from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

def scrape_website(url: str, max_bytes: int = 2_000_000) -> str:
    """Fetch a public HTML page while rejecting private-network destinations."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Website URLs must use http or https")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise ValueError("Private or local website addresses are not allowed")
    except socket.gaierror as error:
        raise ValueError("Website hostname could not be resolved") from error

    response = requests.get(
        url,
        headers={"User-Agent": "LocalRAG/1.0"},
        timeout=(3, 10),
        allow_redirects=False,
        stream=True,
    )
    response.raise_for_status()
    if "text/html" not in response.headers.get("content-type", "").lower():
        raise ValueError("Only HTML websites can be ingested")

    body = bytearray()
    for part in response.iter_content(chunk_size=8192):
        body.extend(part)
        if len(body) > max_bytes:
            raise ValueError("Website response exceeds the allowed size")

    soup = BeautifulSoup(body.decode(response.encoding or "utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return " ".join(soup.stripped_strings)
