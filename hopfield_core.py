import numpy as np
import random
from numpy.linalg import pinv

## 1. 訓練階段：權重矩陣計算
# 使用 Hebbian Rule 計算權重矩陣 W
def train_hopfield_hebb(bipolar_patterns):
    N = bipolar_patterns.shape[0]
    M = bipolar_patterns.shape[1]

    W = np.zeros((N, N))
    for p in range(M):
        s_p = bipolar_patterns[:, p].reshape(-1, 1)
        W += s_p @ s_p.T

    # 設置對角線元素為0 (Wii = 0)
    np.fill_diagonal(W, 0)
    return W

# 使用 Pseudo-Inverse Rule 計算權重矩陣 W
def train_hopfield_pseudoinverse(bipolar_patterns):
    N = bipolar_patterns.shape[0]
    # M = bipolar_patterns.shape[1]

    S = bipolar_patterns

    try:
        # 計算 S 的偽逆 S_pinv (M x N 矩陣)
        S_pinv = pinv(S) 
    except np.linalg.LinAlgError as e:
        print(f"pseudo-inverse計算失敗：{e}")
        return np.zeros((N, N))
        
    # 計算 W = S @ S_pinv (N x N 矩陣)
    W = S @ S_pinv

    # 設置對角線元素為0 (Wii = 0) - *在偽逆法則中，允許 Wii != 0 通常效果更好*
    # np.fill_diagonal(W, 0)
    return W

# 根據選擇的規則返回權重矩陣
def get_trained_weights(bipolar_patterns, rule="PINV"):
    if rule.upper() == "HEBB":
        return train_hopfield_hebb(bipolar_patterns)
    elif rule.upper() == "PINV":
        return train_hopfield_pseudoinverse(bipolar_patterns)
    else:
        raise ValueError("無效的訓練規則。請選擇 'HEBB' 或 'PINV'。")

## 2. 回憶階段：狀態迭代收斂
# 使用異步更新來回憶模式，直到收斂
def recall_pattern(W, initial_state, max_iter=1000):
    N = W.shape[0]
    current_state = initial_state.copy()
    
    # 異步更新
    for iteration in range(max_iter):
        previous_state = current_state.copy()
        update_order = list(range(N))
        random.shuffle(update_order)

        for i in update_order:
            # 計算淨輸入
            net_input = W[i, :] @ current_state

            # 激活函數：Sign(h_i)
            if net_input > 0:
                current_state[i] = 1
            elif net_input < 0:
                current_state[i] = -1
            # 否則 (net_input == 0) 狀態保持不變

        if np.array_equal(current_state, previous_state):
            print(f"收斂於第{iteration + 1}輪迭代")
            return current_state, iteration + 1
            
    return current_state, max_iter

## 3. 驗證：虛偽記憶檢查
# 檢查最終狀態是否為已儲存的真實記憶
def verify_spurious_state(final_state, training_bipolar):
    M = training_bipolar.shape[1]
    
    for j in range(M):
        memory_Pj = training_bipolar[:, j]
        
        # 檢查輸出是否與任一記憶模式完全相同
        if np.array_equal(final_state, memory_Pj):
            return True, j
            
    # 如果迴圈結束都未匹配，則視為偽記憶
    return False, -1