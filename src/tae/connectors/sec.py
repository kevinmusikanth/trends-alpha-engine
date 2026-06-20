from __future__ import annotations

import requests


class SecConnector:
    """SEC company facts connector using the free SEC data API."""

    base_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent

    def fetch_company_facts(self, cik: str) -> dict:
        normalized_cik = str(cik).zfill(10)
        response = requests.get(
            self.base_url.format(cik=normalized_cik),
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

