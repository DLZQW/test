import os
import time

import pandas as pd
import requests


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "stock_data_cache")
TEJ_CACHE_DIR = os.path.join(DATA_DIR, "tej")

if not os.path.exists(TEJ_CACHE_DIR):
    os.makedirs(TEJ_CACHE_DIR)


class TejClient:
    """Lightweight TEJ API adapter.

    The exact endpoint paths and response shape depend on your TEJ subscription.
    Configure them with environment variables so the adapter can be reused across
    different TEJ plans without hard-coding vendor-specific paths.
    """

    def __init__(self, api_key=None, base_url=None, institutional_path=None, margin_path=None, timeout=30):
        self.api_key = api_key or os.getenv("TEJ_API_KEY", "")
        self.base_url = (base_url or os.getenv("TEJ_BASE_URL", "")).rstrip("/")
        self.institutional_path = institutional_path or os.getenv("TEJ_INSTITUTIONAL_PATH", "")
        self.margin_path = margin_path or os.getenv("TEJ_MARGIN_PATH", "")
        self.timeout = timeout

    @property
    def configured(self):
        return bool(self.api_key and self.base_url and self.institutional_path and self.margin_path)

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        }

    def _request_json(self, path, params=None):
        if not self.configured:
            raise ValueError(
                "TEJ is not configured. Set TEJ_API_KEY, TEJ_BASE_URL, TEJ_INSTITUTIONAL_PATH and TEJ_MARGIN_PATH."
            )

        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = requests.get(url, headers=self._headers(), params=params or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _cache_path(self, name):
        return os.path.join(TEJ_CACHE_DIR, name)

    def _read_cache(self, path, max_age_sec=43200):
        if not os.path.exists(path):
            return None

        age = time.time() - os.path.getmtime(path)
        if age > max_age_sec:
            return None

        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def _write_cache(self, df, path):
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return df

    def _unwrap_rows(self, payload):
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "result", "results", "rows"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
        return []

    @staticmethod
    def _normalize_date_series(series):
        return pd.to_datetime(series, errors="coerce").dt.strftime("%Y%m%d")

    def get_institutional_trading(self, stock_id, start_date, end_date, force_update=False):
        cache = self._cache_path(f"{stock_id}_institutional_{start_date}_{end_date}.csv")
        if not force_update:
            cached = self._read_cache(cache)
            if cached is not None:
                return cached

        payload = self._request_json(
            self.institutional_path,
            params={"coid": stock_id, "start_date": start_date, "end_date": end_date},
        )
        rows = self._unwrap_rows(payload)
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = self._normalize_institutional(df, stock_id)
        return self._write_cache(df, cache)

    def get_margin_short(self, stock_id, start_date, end_date, force_update=False):
        cache = self._cache_path(f"{stock_id}_margin_{start_date}_{end_date}.csv")
        if not force_update:
            cached = self._read_cache(cache)
            if cached is not None:
                return cached

        payload = self._request_json(
            self.margin_path,
            params={"coid": stock_id, "start_date": start_date, "end_date": end_date},
        )
        rows = self._unwrap_rows(payload)
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = self._normalize_margin(df, stock_id)
        return self._write_cache(df, cache)

    def _normalize_institutional(self, df, stock_id):
        rename_map = {
            "date": "資料日期",
            "trade_date": "資料日期",
            "coid": "股票代號",
            "stock_id": "股票代號",
            "foreign_net_buy_sell": "外資買賣超",
            "investment_net_buy_sell": "投信買賣超",
            "dealer_net_buy_sell": "自營商買賣超",
            "inst_net_buy_sell": "三大法人買賣超",
            "close": "收盤價",
        }
        df = df.rename(columns=rename_map)

        if "股票代號" not in df.columns:
            df["股票代號"] = str(stock_id).zfill(4)
        df["股票代號"] = df["股票代號"].astype(str).str.zfill(4)

        if "資料日期" in df.columns:
            df["資料日期"] = self._normalize_date_series(df["資料日期"])

        for col in ["外資買賣超", "投信買賣超", "自營商買賣超", "三大法人買賣超", "收盤價"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(subset=["資料日期"]).sort_values("資料日期").reset_index(drop=True)

    def _normalize_margin(self, df, stock_id):
        rename_map = {
            "date": "資料日期",
            "trade_date": "資料日期",
            "coid": "股票代號",
            "stock_id": "股票代號",
            "margin_balance": "融資餘額",
            "margin_change": "融資增減",
            "short_balance": "融券餘額",
            "short_change": "融券增減",
            "short_margin_ratio": "券資比",
            "close": "收盤價",
        }
        df = df.rename(columns=rename_map)

        if "股票代號" not in df.columns:
            df["股票代號"] = str(stock_id).zfill(4)
        df["股票代號"] = df["股票代號"].astype(str).str.zfill(4)

        if "資料日期" in df.columns:
            df["資料日期"] = self._normalize_date_series(df["資料日期"])

        for col in ["融資餘額", "融資增減", "融券餘額", "融券增減", "券資比", "收盤價"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(subset=["資料日期"]).sort_values("資料日期").reset_index(drop=True)
