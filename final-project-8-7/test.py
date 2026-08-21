import crawler
import backtest
import predict_cbc
import train_ml
from tabulate import tabulate
import re
import unicodedata

def get_display_width(text):
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)

# ==========================================
# 🌟 參數設定區 (將 GA 最佳化參數與訓練參數分離)
# ==========================================
# 模式一用：寬鬆參數 (目的是收集包含獲利與虧損的平衡樣本)
TRAIN_PARAMS = {
    "continuous_weeks": 2,
    "min_growth": -0.05,           # 容許微幅衰退
    "last_week_threshold": -0.05,  # 容許微幅衰退
    "pop_decline_threshold": -0.05,# 容許微幅衰退
    "large_corr_thresh": 0.0,      # 放寬大戶相關係數
    "retail_corr_thresh": 0.0,     # 放寬散戶相關係數
    "avg_corr_thresh": 0.0,        # 放寬均張相關係數
    "skip_cond_e": True            # 🌟 新增：訓練模式略過條件 E，收集失敗案例
}

# 模式二用：GA 最佳化參數 (嚴格實戰條件)
STRICT_PARAMS = {
    "continuous_weeks": 2,
    "min_growth": 0.0041,
    "last_week_threshold": 1.05,
    "pop_decline_threshold": 0.907,
    "skip_cond_e": False           # 🌟 實戰模式必須開啟條件 E
}

def main():
    print("========================================")
    print(" 🌟 歡迎使用【籌碼面量化分析系統】🌟")
    print("========================================")
    
    stock_list = crawler.get_stock_ids(crawler.list_url)
    total_available = len(stock_list)

    if total_available == 0:
        print("❌ 沒抓到股票清單，程式結束。")
        return

    print(f"✅ 成功取得 {total_available} 檔股票清單。")
    print("--------------------------------")
    print("1. 前 10 個 (快速測試)")
    print("2. 前 50 個 (建議)")
    print(f"3. 全部 ({total_available} 個)")
    print("4. 自訂範圍")
    print("--------------------------------")

    choice = input("👉 請輸入掃描範圍 (1/2/3/4): ").strip()
    start_index, end_index = 0, 10

    if choice == '2':
        end_index = min(50, total_available)
    elif choice == '3':
        end_index = total_available
    elif choice == '4':
        try:
            start_index = int(input("👉 從第幾檔開始? (預設 0): ").strip() or 0)
            count = int(input("👉 要抓幾檔? (預設 10): ").strip() or 10)
            end_index = min(start_index + count, total_available)
        except:
            start_index, end_index = 0, min(10, total_available)

    target_list = stock_list[start_index:end_index]

    # 🌟 修改選單結構
    print("\n--------------------------------")
    print("請選擇要執行的功能：")
    print("1. 🧠 【模式一】收集訓練資料 (寬鬆參數) 並訓練機器學習模型")
    print("2. 📊 【模式二】執行實戰嚴格回測 (GA 最佳化參數)")
    print("3. 🔮 【模式三】執行下週飆股預測 (未來的實戰應用)")
    print("--------------------------------")
    mode = input("👉 請選擇 (1/2/3): ").strip()

    print(f"\n🚀 準備處理 {len(target_list)} 檔股票...\n")

    # ==========================================
    # 模式一：收集訓練資料並更新模型
    # ==========================================
    if mode == '1':
        print("🔍 啟動訓練模式：使用寬鬆參數收集平衡樣本...")
        # 傳入 TRAIN_PARAMS 且 is_training=True
        trades_df = backtest.run_all_analysis(target_list, params=TRAIN_PARAMS, is_training=True)
        
        if not trades_df.empty:
            print("\n✅ 資料收集完畢，準備訓練模型...")
            backtest.export_ml_data()
            print("\n" + "=" * 90)
            print("🧠 開始訓練/更新機器學習模型...")
            print("=" * 90)
            train_ml.main()
        else:
            print("\n⚠️ 沒有符合任何寬鬆條件的訊號。")

    # ==========================================
    # 模式二：嚴格條件實戰回測
    # ==========================================
    elif mode == '2':
        print("🔍 啟動回測模式：使用 GA 最佳化參數進行嚴格篩選...")
        # 傳入 STRICT_PARAMS 且 is_training=False
        trades_df = backtest.run_all_analysis(target_list, params=STRICT_PARAMS, is_training=False)
        
        if not trades_df.empty:
            display_df = trades_df.fillna({'下週收盤出場價': '等待開獎', '週報酬%': '等待開獎'})
            print("\n" + "=" * 90)
            print("📈 籌碼策略回測結果 (實戰嚴格參數)")
            print("=" * 90)
            print(tabulate(display_df, headers='keys', tablefmt='simple', showindex=False))
            
            completed_trades = trades_df.dropna(subset=['週報酬%'])
            if not completed_trades.empty:
                win_rate = (completed_trades['週報酬%'] > 0).mean() * 1003
                avg_return = completed_trades['週報酬%'].mean()
                print(f"\n【總體績效統計】")
                print(f"勝率: {win_rate:.2f}% | 平均週報酬: {avg_return:.2f}% | 訊號總數: {len(completed_trades)} (已結算)")
            
            print("\n✅ 實戰回測結束。此模式不會覆蓋機器學習模型。")
        else:
            print("\n⚠️ 嚴格參數下，沒有出現任何符合條件的回測訊號。")

    # ==========================================
    # 模式三：未來預測
    # ==========================================
    elif mode == '3':
        # 🌟 關鍵修改：將 STRICT_PARAMS 傳給預測模組
        recommend_df = predict_cbc.get_next_week_recommendations(target_list, STRICT_PARAMS)
        
        # 🌟 新增：強制換行，避免被前面的 end="\r" 吃掉輸出
        print("\n")
        
        # 接下來是你原本的排版程式碼
        print("\n" + "=" * 110)
        if not recommend_df.empty:
            print("🎯 掃描完畢！發現以下【下週實戰推薦清單】：")
            print("=" * 110)
            
            display_df = recommend_df.drop(columns=['歷史走勢明細'])
            print(tabulate(display_df, headers='keys', tablefmt='simple', showindex=False))
            
            print("\n" + "=" * 110)
            print("📜 【歷史相似走勢 - 深度明細解析】")
            print("=" * 110)
            
            for idx, row in recommend_df.iterrows():
                print(f"🔸 股票代號: 【 {row['代號']} 】 | 綜合建議: {row['建議']}")
                if row['歷史走勢明細'] == "無":
                    print("   └─ 歷史上尚無完全相同之訊號可供比對。")
                else:
                    trades = row['歷史走勢明細'].split('\n')
                    for trade_str in trades:
                        match = re.search(r'(.*軌跡: )(.*) (\(.*)', trade_str)
                        if match:
                            prefix = "   └─ " + match.group(1)
                            trajectory_str = match.group(2)
                            status_str = " " + match.group(3)
                            
                            weeks = trajectory_str.split(', ')
                            indent_width = get_display_width(prefix)
                            indent_spaces = " " * indent_width
                            
                            chunk_size = 8
                            lines = []
                            for i in range(0, len(weeks), chunk_size):
                                lines.append(", ".join(weeks[i:i+chunk_size]))
                            
                            formatted_trajectory = f",\n{indent_spaces}".join(lines)
                            print(f"{prefix}{formatted_trajectory}{status_str}")
                        else:
                            print(f"   └─ {trade_str}")
                print("-" * 110)
                
            print("\n💡 判讀教學：")
            print("若「綜合建議」顯示為『❌ 回測不佳』，代表此股票過去發生相同訊號時，")
            print("多半會立刻遭遇連續兩週下跌的停損出場，或歷史平均報酬為負，請避開陷阱。")
        else:
            print("⚠️ 掃描完畢，目前的清單中【沒有】剛好在最新一週觸發進場訊號的股票。")

if __name__ == "__main__":
    main()
