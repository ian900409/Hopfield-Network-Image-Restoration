import numpy as np
import os
import string
import random

## 1. 數據轉換
# 將 (0, 1) 轉換為 (+1, -1)
def bipolarize(patterns):
    return 2 * patterns - 1

# 將 (-1, +1) 轉換回 (0, 1)
def unipolarize(bipolar_pattern):
    return (bipolar_pattern + 1) // 2

## 2. 數據載入
# 讀取 ROWS 行數據，並忽略最後的單行分隔符
def load_data(file_path, rows, cols):
    # 實際讀取的行數：ROWS (有效數據行) + 1 (單行分隔符)
    READ_ROWS = rows + 1
    TOTAL_READ_SIZE = READ_ROWS * cols
    
    patterns_list = []
    current_pattern_data = []
    
    if not os.path.exists(file_path):
        # 錯誤處理將在主程式碼中處理
        return None
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            
            # 清洗輸入行：處理 Unicode 和不可見字符
            line = line.replace('\xa0', ' ').replace('\t', ' ')
            line_cleaned = ''.join(filter(lambda x: x in string.printable, line.strip('\n')))
            line_stripped = line_cleaned.strip()
            is_empty_data_line = not line_stripped
            
            row_data = [0] * cols
            
            if not is_empty_data_line:
                # 逐字符解析前 COLS 個字符
                for i in range(cols):
                    try:
                        char = line_cleaned[i]
                        if char == '1':
                            row_data[i] = 1
                    except IndexError:
                        break

            # 累積數據點
            current_pattern_data.extend(row_data)
            
            # 檢查模式是否完成 (總點數達到 TOTAL_READ_SIZE)
            if len(current_pattern_data) == TOTAL_READ_SIZE:
                
                # 忽略最後 cols 個點 (分隔符)
                valid_pattern_data = np.array(current_pattern_data)[:rows*cols]
                patterns_list.append(valid_pattern_data)
                
                current_pattern_data = [] # 清空
            
        if not patterns_list:
            return None
        
        # 返回 N x M 矩陣 (N: 神經元數, M: 模式數)
        return np.stack(patterns_list).T

    except Exception as e:
        # 返回 None 讓主程式碼處理
        print(f"錯誤：{e}")
        return None

# 3. 雜訊生成
# def add_noise(bipolar_pattern, noise_percentage):
#     """ 將指定百分比的雜訊加入雙極性模式中。 """
#     N = bipolar_pattern.size
    
#     num_flips = int(N * noise_percentage / 100)
    
#     if num_flips == 0 and noise_percentage > 0:
#         # 確保即使百分比很低，也至少翻轉一個點
#         num_flips = 1
        
#     noise_indices = np.random.choice(N, size=num_flips, replace=False)
    
#     noisy_pattern = bipolar_pattern.copy()
#     noisy_pattern[noise_indices] *= -1 
    
#     return noisy_pattern