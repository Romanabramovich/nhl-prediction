# ingest/clients/nhl.py
import requests
import pathlib, json, hashlib, time
from typing import Any

CACHE_DIR = pathlib.Path("data/.cache")

class NHLClient:
    def __init__(self, base_url: str = "https://api-web.nhle.com", timeout=15, rate_limit_rps=3):
        self.base_url   = base_url.rstrip("/")
        self.session    = requests.Session()
        self.timeout    = timeout
        self.rate_sleep = 1.0 / rate_limit_rps
        
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
    def _cache_key(self, path: str, params: dict[str, Any] | None) -> pathlib.Path:
        key = f"{path}?{json.dumps(params, sort_keys=True)}"
        hashed_key = hashlib.sha256(key.encode()).hexdigest()[:24]
        
        return CACHE_DIR / f"{hashed_key}.json"
    
    def get(self, path: str, params: dict[str, Any] | None = None, use_cache = True) -> dict:
        url = f"{self.base_url}{path}"
        cache_path = self._cache_key(path, params or {})
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        
    
        # polite retries when fetching
        timeout = 5
        backoff = 1.5
        
        for attempt in range(1, timeout + 1):
            time.sleep(self.rate_sleep)
            
            try:
                response = self.session.get(url, params=params, timeout=self.timeout, headers={
                    "User-Agent": "nhl-prediction (for project; abramovichroman19@gmail.com)"
                })
                
                #error code OK
                if response.status_code == 200:
                    data = response.json()
                    if use_cache:
                        cache_path.write_text(json.dumps(data), encoding="utf-8")
                    return data

                # error code rate limited or service unavailable
                if response.status_code in (429, 503):
                    time.sleep(backoff ** attempt)
                    continue
                    
                response.raise_for_status()
            except requests.RequestException:
                if attempt == timeout:
                    raise
                time.sleep(backoff ** attempt)
        
        raise RuntimeError("runtime error: unreachable")
                
                
        