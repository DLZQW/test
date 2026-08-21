import pandas as pd
import requests
import time
import re
import numpy as np
import yfinance as yf
import io
import contextlib
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings('ignore')

# --- 基礎設定 ---
list_url = "https://norway.twsthr.info/StockHoldersDividendTop.aspx?CID=0&Show=1"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "stock_data_cache")      

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def _quiet_yf_download(*args, **kwargs):
    """Silence yfinance console noise and return dataframe as-is."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return yf.download(*args, **kwargs)

# ==========================================
# 股價與爬蟲輔助功能
# ==========================================
def download_stock_price_history(stock_id, force_update=False):
    """下載單一股票 3 年歷史價格數據並存成 CSV (快取 12 小時)"""
    price_file = os.path.join(DATA_DIR, f"{stock_id}_price_history.csv")
    
    # 檢查快取是否有效
    if os.path.exists(price_file):
        file_age = time.time() - os.path.getmtime(price_file)
        if file_age <= 43200 and not force_update:
            try:
                df = pd.read_csv(price_file, index_col='Date', parse_dates=True)
                if not df.empty:
                    return df
            except:
                pass
    
    # 計算 3 年前的日期
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)
    
    # 嘗試兩種股票代碼後綴
    for suffix in [".TW", ".TWO"]:
        ticker = f"{stock_id}{suffix}"
        try:
            print(f"  ↓ 下載 {ticker} 價格歷史...", end=" ", flush=True)
            data = _quiet_yf_download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False
            )
            
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                    
                # 保留必要欄位
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                data.index.name = 'Date'
                data.to_csv(price_file, encoding='utf-8-sig')
                print(f"✓ 成功({len(data)}筆)")
                return data
        except Exception as e:
            print(f"✗ 失敗")
            continue
    
    print(f"✗ 無法取得")
    return None

def get_next_monday_open_price(stock_id, signal_date):
    """計算並從本地價格歷史取得下週一的開盤價"""
    try:
        signal_dt = datetime.strptime(str(signal_date).replace('-', '').split(' ')[0], "%Y%m%d")
        days_until_monday = (7 - signal_dt.weekday()) % 7
        if days_until_monday == 0: days_until_monday = 7
        next_monday = signal_dt + timedelta(days=days_until_monday)

        # 先確保有下載該股票的價格歷史
        price_df = download_stock_price_history(stock_id)
        if price_df is None or price_df.empty:
            return np.nan
        
        # 從 DataFrame 中查找下週一的開盤價
        next_monday_str = next_monday.strftime("%Y-%m-%d")
        if next_monday_str in price_df.index.strftime("%Y-%m-%d").values:
            idx = price_df.index.strftime("%Y-%m-%d") == next_monday_str
            if idx.any():
                return float(price_df.loc[idx, 'Open'].iloc[0])
        
        return np.nan
    except Exception:
        return np.nan

def get_next_friday_close_price(stock_id, signal_date):
    """計算並從本地價格歷史取得下週五的收盤價，若無則往前查找最近交易日"""
    try:
        signal_dt = datetime.strptime(str(signal_date).replace('-', '').split(' ')[0], "%Y%m%d")
        days_until_monday = (7 - signal_dt.weekday()) % 7
        if days_until_monday == 0: days_until_monday = 7
        next_monday = signal_dt + timedelta(days=days_until_monday)
        next_friday = next_monday + timedelta(days=4)

        # 先確保有下載該股票的價格歷史
        price_df = download_stock_price_history(stock_id)
        if price_df is None or price_df.empty:
            return np.nan
        
        # 先嘗試找下週五，若無則往前查找最多 5 個交易日
        # 但不能往前查超過進場日期 (signal_dt)
        for days_back in range(0, 5):
            target_date = next_friday - timedelta(days=days_back)

            # 若查到的日期早於進場日期，停止往前查找
            if target_date < signal_dt:
                break

            target_date_str = target_date.strftime("%Y-%m-%d")
            
            date_strs = price_df.index.strftime("%Y-%m-%d").values
            if target_date_str in date_strs:
                idx = price_df.index.strftime("%Y-%m-%d") == target_date_str
                if idx.any():
                    return float(price_df.loc[idx, 'Close'].iloc[0])

        return np.nan
    except Exception:
        return np.nan
    
def check_condition_e_with_yfinance(stock_id, signal_date, monday_open_price):
    """條件 E: 從本地價格歷史檢查下週二收盤 > 週一開盤，且週三、週四收盤連續走高。若無則往前查找最近交易日，但不超過進場日期。"""
    try:
        signal_dt = datetime.strptime(str(signal_date).replace('-', '').split(' ')[0], "%Y%m%d")
    except ValueError:
        return False

    days_until_monday = (7 - signal_dt.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7

    next_monday = signal_dt + timedelta(days=days_until_monday)
    next_tuesday = next_monday + timedelta(days=1)
    next_wednesday = next_monday + timedelta(days=2)
    next_thursday = next_monday + timedelta(days=3)
    target_dates = [next_tuesday, next_wednesday, next_thursday]

    # 先確保有下載該股票的價格歷史
    price_df = download_stock_price_history(stock_id)
    if price_df is None or price_df.empty:
        return False
    
    # 轉換索引為日期字符串便於查詢
    price_df['Date_str'] = price_df.index.strftime("%Y-%m-%d")
    
    # 為每個目標日期查找收盤價（往前查找最多 5 天），但不能查到進場日前的價格
    closes = []
    for target_date in target_dates:
        found = False
        for days_back in range(0, 5):
            search_date = target_date - timedelta(days=days_back)

            # 不允許查到進場日期之前的價格
            if search_date < signal_dt:
                break

            search_date_str = search_date.strftime("%Y-%m-%d")
            
            if search_date_str in price_df['Date_str'].values:
                close_price = float(price_df[price_df['Date_str'] == search_date_str]['Close'].iloc[0])
                closes.append(close_price)
                found = True
                break
        
        if not found:
            return False  # 如果某一天怎麼都找不到，就視為條件不成立
    
    tue_close, wed_close, thu_close = closes
    
    # 條件 E: 下週二收盤 > 週一開盤，且週三、週四收盤連續走高
    return tue_close > monday_open_price and wed_close > tue_close and thu_close > wed_close

# ==========================================
# 爬蟲引擎 (含 12 小時智慧快取)
# ==========================================
def get_stock_ids(url=list_url, force_update=False):
    """取得全市場股票代號清單"""
    list_file = os.path.join(DATA_DIR, "stock_list.txt")
    
    if os.path.exists(list_file):
        file_age = time.time() - os.path.getmtime(list_file)
        if file_age > 43200: force_update = True
            
    if not force_update and os.path.exists(list_file):
        with open(list_file, 'r', encoding='utf-8') as f: return [line.strip() for line in f.readlines()]
    try:
        r = requests.get(url, headers=headers)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'lxml')
        stock_ids = [match.group(1) for a in soup.find_all('a', href=True) if (match := re.search(r'STOCK=(\d{4,6})', a['href'])) and len(match.group(1)) == 4]
        stock_ids = sorted(list(set(stock_ids)))
        if stock_ids:
            with open(list_file, 'w', encoding='utf-8') as f:
                for sid in stock_ids: f.write(f"{sid}\n")
        return stock_ids
    except: return []

def get_individual_stock_data(stock_id, force_update=False):
    """取得單一股票籌碼資料"""
    file_path = os.path.join(DATA_DIR, f"{stock_id}.csv")
    
    if os.path.exists(file_path):
        file_age = time.time() - os.path.getmtime(file_path)
        if file_age > 43200: 
            force_update = True 
            
    if not force_update and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if '>1000張百分比' in df.columns:
                df['股票代號'] = df['股票代號'].astype(str).str.zfill(4)
                df['資料日期'] = df['資料日期'].astype(str)
                return df
            force_update = True
        except: pass

    url = f"https://norway.twsthr.info/StockHolders.aspx?stock={stock_id}"
    try:
        r = requests.get(url, headers=headers)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'lxml')
        data_rows = []
        for tr in soup.find_all('tr'):
            numbers = re.findall(r'\d+\.?\d*', tr.get_text(separator=' ').replace(',', ''))
            if len(numbers) >= 13 and re.match(r'^202\d{5}$', numbers[0]):
                try:
                    row = {
                        '資料日期': numbers[0], '總張數' : numbers[1], '總股東人數': numbers[2], 
                        '平均張數/人': numbers[3], '>1000張百分比': numbers[4], '>400張百分比': numbers[5], '收盤價': numbers[12]
                    }
                    if 0 < float(row['收盤價']) < 100000: data_rows.append(row)
                except: continue
        
        if not data_rows: return None
        df = pd.DataFrame(data_rows).drop_duplicates(subset=['資料日期'])
        for col in ['總張數','總股東人數', '平均張數/人', '>1000張百分比', '>400張百分比', '收盤價']: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        df.insert(0, '股票代號', str(stock_id).zfill(4))
        df = df.sort_values('資料日期', ascending=True).reset_index(drop=True)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return df
    except: return None
