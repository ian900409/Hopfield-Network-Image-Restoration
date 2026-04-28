import numpy as np
import os
import hopfield_core as core
import data_utils as utils
import sys

# 參數配置 (根據 GUI 選擇動態調整)

# 定義不同數據集的尺寸和檔案名
DATA_CONFIG = {
    "BASIC": {
        "ROWS": 12, 
        "COLS": 9,
        "TRAIN_FILE": "./Hopfield_dataset/Basic_Training.txt",
        "TEST_FILE": "./Hopfield_dataset/Basic_Testing.txt",
    },
    "BONUS": {
        "ROWS": 10,
        "COLS": 10,
        "TRAIN_FILE": "./Hopfield_dataset/Bonus_Training.txt",
        "TEST_FILE": "./Hopfield_dataset/Bonus_Testing.txt",
    }
}

# 負責處理所有數據載入、訓練和回憶邏輯，作為 Qt 介面的後端接口
class HopfieldGUIConnector:  
    def __init__(self):
        self.W = None
        self.training_bipolar = None
        self.testing_bipolar = None
        self.rows = 0
        self.cols = 0
        self.current_dataset = None
        
    # 載入數據並訓練網路
    def load_and_train(self, dataset_key, rule):
        config = DATA_CONFIG.get(dataset_key.upper())
        if not config:
            raise ValueError("無效的數據集選擇。")

        self.current_dataset = dataset_key.upper()
        self.rows = config['ROWS']
        self.cols = config['COLS']
        
        # 1. 載入訓練數據
        print(f"載入 {self.rows}x{self.cols} 訓練數據...")
        training_uni = utils.load_data(config['TRAIN_FILE'], self.rows, self.cols)
        if training_uni is None:
            raise FileNotFoundError(f"無法載入訓練檔案: {config['TRAIN_FILE']}")

        self.training_bipolar = utils.bipolarize(training_uni)
        print(f"成功載入 {self.training_bipolar.shape[1]} 個訓練模式。")

        # 2. 訓練：計算權重矩陣 W
        self.W = core.get_trained_weights(self.training_bipolar, rule)
        print(f"權重矩陣 W 計算完成 (大小：{self.W.shape[0]} x {self.W.shape[0]})。")
        
        # 3. 載入測試數據
        testing_uni = utils.load_data(config['TEST_FILE'], self.rows, self.cols)
        if testing_uni is not None:
            self.testing_bipolar = utils.bipolarize(testing_uni)
        else:
            self.testing_bipolar = None
            
        # 返回訓練模式數量和測試模式數量，供 GUI 更新下拉選單
        num_train = self.training_bipolar.shape[1]
        num_test = self.testing_bipolar.shape[1] if self.testing_bipolar is not None else 0
        
        return self.training_bipolar, num_train, num_test

    # 根據測試索引進行單次回憶
    def recall_test_pattern(self, test_index, noise_percent=0):
        if self.W is None:
            raise RuntimeError("請先訓練網路。")
        if self.testing_bipolar is None:
            raise RuntimeError("未載入測試資料。")

        # 檢查索引是否有效
        if test_index < 0 or test_index >= self.testing_bipolar.shape[1]:
             raise IndexError("測試模式索引超出範圍。")
             
        # 1. 從測試集取得初始輸入
        # 由於暫無 add_noise，我們直接使用測試集中的模式作為輸入
        initial_input = self.testing_bipolar[:, test_index].copy()
        noisy_input = initial_input # 0% 雜訊
        
        # 2. 回憶
        final_state, convergence_iter = core.recall_pattern(self.W, noisy_input)
        
        # 3. 驗證
        is_memory, matched_index = core.verify_spurious_state(final_state, self.training_bipolar)
        
        # 返回結果，供 GUI 繪製
        return noisy_input, final_state, is_memory, matched_index, convergence_iter

## 主程序
if __name__ == "__main__":
    
    # 預設參數
    DEFAULT_DATASET = "BONUS"
    DEFAULT_RULE = "PINV"
    DEFAULT_TEST_INDEX = 0  # 預設測試第一個模式 (index 0)
    
    dataset_choice = sys.argv[1].upper() if len(sys.argv) > 1 else DEFAULT_DATASET
    rule_choice = sys.argv[2].upper() if len(sys.argv) > 2 else DEFAULT_RULE
    try:
        test_index = int(sys.argv[3]) - 1 if len(sys.argv) > 3 else DEFAULT_TEST_INDEX
    except ValueError:
        print("警告：測試索引無效，使用預設值 1。")
        test_index = DEFAULT_TEST_INDEX
        
    
    print(f"\n--- 🚀 開始驗證：{dataset_choice} 數據集, {rule_choice} 規則 ---")
    
    connector = HopfieldGUIConnector()
    
    try:
        # 1. 載入並訓練
        train_patterns, num_train, num_test = connector.load_and_train(dataset_choice, rule_choice)
        
        # 2. 測試單個模式 (模擬 GUI 點擊)
        print(f"\n--- 模擬回憶測試 #{test_index + 1} ---")
        
        # 確保測試索引在範圍內
        if test_index >= num_test:
            test_index = 0
            print("警告：測試索引超出範圍，改為測試第一個模式。")
            
        noisy_input, final_state, is_memory, matched_index = connector.recall_test_pattern(
            test_index=test_index, 
            noise_percent=0 # 暫時沒有雜訊
        )
        
        # 3. 顯示結果
        print(f"輸入模式尺寸: {noisy_input.shape}")
        print(f"回憶狀態: {'✅ 成功' if is_memory else '❌ 失敗/虛偽記憶'} (匹配記憶 #{matched_index + 1 if is_memory else '無'})")
        
    except Exception as e:
        print(f"\n致命錯誤發生：{e}")
        print("請確保資料檔案路徑正確，並檢查相關模組的邏輯。")