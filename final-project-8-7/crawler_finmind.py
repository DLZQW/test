import pandas as pd
from FinMind.data import DataLoader
import time
import os
import sys
from datetime import timedelta

# 建立快取資料夾
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "stock_data_cache")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def _convert_to_weekly(df_daily):
    """內部函式：將日級 DataFrame 轉換為週級資料"""
    if df_daily.empty:
        return pd.DataFrame()
        
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily.set_index('date', inplace=True)
    
    # 拆分外資與信用交易進行降頻
    df_foreign = df_daily[['Foreign_Buy', 'Foreign_Sell']].copy()
    df_margin = df_daily[['MarginPurchaseBalance', 'ShortSaleBalance']].copy()
    
    # 外資買賣超用「加總」
    weekly_foreign = df_foreign.resample('W-FRI').sum()
    weekly_foreign.rename(columns={'Foreign_Buy': 'Foreign_Buy_Sum', 'Foreign_Sell': 'Foreign_Sell_Sum'}, inplace=True)
    
    # 融資券餘額用「最後一天」
    weekly_margin = df_margin.resample('W-FRI').last()
    
    # 合併並整理格式
    df_weekly = pd.merge(weekly_foreign, weekly_margin, left_index=True, right_index=True, how='outer')
    df_weekly.reset_index(inplace=True)
    
    return df_weekly

def fetch_weekly_chip_data(stock_id, start_date, end_date):
    """
    抓取 FinMind 籌碼資料。
    具備增量更新機制 (儲存日級快取)，並回傳週級資料供回測使用。
    """
    file_path = os.path.join(DATA_DIR, f"{stock_id}_finmind_daily.csv")
    dl = DataLoader()
    
    fetch_start_date = pd.to_datetime(start_date)
    target_end_date = pd.to_datetime(end_date)
    df_existing = pd.DataFrame()
    
    # 1. 檢查並讀取本地日級快取
    if os.path.exists(file_path):
        # 🌟 取得檔案最後修改時間，計算經過了幾秒
        file_age = time.time() - os.path.getmtime(file_path)
        
        try:
            df_existing = pd.read_csv(file_path, parse_dates=['date'])
            if not df_existing.empty:
                last_cached_date = df_existing['date'].max()
                
                # 🌟 修復關鍵：如果快取已經涵蓋目標日期，【或】檔案是 12 小時內 (43200秒) 剛更新的
                # 就不再去向 API 要資料，直接放行 (解決假日造成的無窮迴圈)
                if last_cached_date >= target_end_date or file_age <= 43200:
                    mask = (df_existing['date'] >= fetch_start_date) & (df_existing['date'] <= target_end_date)
                    return _convert_to_weekly(df_existing.loc[mask].copy())
                
                # 若需要更新，將抓取起始日設為快取最後一天的「隔天」
                fetch_start_date = last_cached_date + timedelta(days=1)
        except Exception as e:
            print(f"  讀取快取失敗 ({e})，將重新抓取完整區間...")
            df_existing = pd.DataFrame()

    fetch_start_str = fetch_start_date.strftime('%Y-%m-%d')
    target_end_str = target_end_date.strftime('%Y-%m-%d')
    
    # 如果開始日期已經大於結束日期，代表無須抓取 (保險機制)
    if fetch_start_date > target_end_date:
        return _convert_to_weekly(df_existing)

    # 2. 呼叫 API 抓取增量資料 (包含自動休眠)
    max_retries = 3
    retry_count = 0
    df_new_daily = pd.DataFrame()
    
    while retry_count < max_retries:
        try:
            print(f"  ↓ 抓取 {stock_id} 增量資料 ({fetch_start_str} 至 {target_end_str})...", end=" ", flush=True)
            
            df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=fetch_start_str, end_date=target_end_str)
            df_margin = dl.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=fetch_start_str, end_date=target_end_str)
            
            # 若無新資料，直接使用現有資料
            if df_inst.empty and df_margin.empty:
                print("✓ 無新資料需更新")
                df_new_daily = pd.DataFrame()
                break
                
            # 清理新資料格式
            if not df_inst.empty:
                df_inst = df_inst[df_inst['name'] == 'Foreign_Investor'][['date', 'buy', 'sell']].copy()
                df_inst.rename(columns={'buy': 'Foreign_Buy', 'sell': 'Foreign_Sell'}, inplace=True)
                df_inst['date'] = pd.to_datetime(df_inst['date'])
            else:
                df_inst = pd.DataFrame(columns=['date', 'Foreign_Buy', 'Foreign_Sell'])
                
            if not df_margin.empty:
                df_margin = df_margin[['date', 'MarginPurchaseTodayBalance', 'ShortSaleTodayBalance']].copy()
                df_margin.rename(columns={'MarginPurchaseTodayBalance': 'MarginPurchaseBalance', 'ShortSaleTodayBalance': 'ShortSaleBalance'}, inplace=True)
                df_margin['date'] = pd.to_datetime(df_margin['date'])
            else:
                df_margin = pd.DataFrame(columns=['date', 'MarginPurchaseBalance', 'ShortSaleBalance'])

            # 合併新的外資與融資券資料
            df_new_daily = pd.merge(df_inst, df_margin, on='date', how='outer')
            print("✓ 抓取成功")
            time.sleep(0.5)  # 基礎保護延遲
            break

        except Exception as e:
            error_msg = str(e).lower()
            if "upper limit" in error_msg or "limit" in error_msg or "429" in error_msg:
                retry_count += 1
                sleep_minutes = 15
                print(f"\n⚠️ 觸發 API 上限，準備休眠 {sleep_minutes} 分鐘後重試 (第 {retry_count}/{max_retries} 次)...")
                for remaining in range(sleep_minutes * 60, 0, -1):
                    sys.stdout.write(f"\r⏳ 倒數計時: {remaining} 秒 ")
                    sys.stdout.flush()
                    time.sleep(1)
                print("\n🔄 休眠結束，繼續執行！")
            else:
                print(f"\n⚠️ 抓取發生錯誤: {e}")
                break
                
    # 3. 資料合併與儲存
    if not df_new_daily.empty:
        if not df_existing.empty:
            df_final = pd.concat([df_existing, df_new_daily], ignore_index=True)
        else:
            df_final = df_new_daily
            
        # 確保依日期排序並去除重複項，處理完後寫入 CSV
        df_final.drop_duplicates(subset=['date'], keep='last', inplace=True)
        df_final.sort_values('date', inplace=True)
        df_final.to_csv(file_path, index=False, encoding='utf-8-sig')
    else:
        df_final = df_existing

    # 4. 輸出指定日期區間的週級資料
    if not df_final.empty:
        mask = (df_final['date'] >= pd.to_datetime(start_date)) & (df_final['date'] <= pd.to_datetime(end_date))
        return _convert_to_weekly(df_final.loc[mask].copy())
    
    return pd.DataFrame()

# 測試執行模組
if __name__ == "__main__":
    test_df = fetch_weekly_chip_data('2330', '2023-01-01', '2026-08-04')
    print(test_df.head())