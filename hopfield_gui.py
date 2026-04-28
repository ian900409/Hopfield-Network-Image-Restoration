import sys
# 📌 導入語句已修改為 PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QWidget, QVBoxLayout, 
    QGridLayout, QGraphicsView, QGraphicsScene, QStyle, QComboBox,
    QRadioButton, QPushButton, QGroupBox, QLabel, QFrame, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QBrush
from PyQt5.QtCore import Qt, QSize, QRectF
from PyQt5 import uic
import numpy as np
import os

# 導入後端接口
from main import HopfieldGUIConnector, DATA_CONFIG
from data_utils import unipolarize


class HopfieldGUI(QMainWindow):
    def __init__(self, ui_file_name = "hopfield.ui"):
        super().__init__()
        uic.loadUi(ui_file_name, self)
        
        self.connector = HopfieldGUIConnector()
        self.current_patterns_count = 0
        self.current_rows = 0
        self.current_cols = 0
        
        # 設置初始介面元件的預設值和狀態
        self.setup_initial_ui_values()
        
        # 連接訊號：元件現在是 self 的屬性
        self.connect_signals()

    # 初始化 UI 元件的值，確保元件是可用的
    def setup_initial_ui_values(self):
        # 確保下拉選單的值在載入後被設置
        if hasattr(self, 'comboBox_dataset'):
            self.comboBox_dataset.blockSignals(True) # 阻擋信號
            self.comboBox_dataset.clear()
            self.comboBox_dataset.addItem("BASIC")
            self.comboBox_dataset.addItem("BONUS")
            self.comboBox_dataset.setCurrentIndex(0) 
            self.comboBox_dataset.setEnabled(True)
            self.comboBox_dataset.blockSignals(False) # 恢復信號
        
        # 設置預設規則
        if hasattr(self, 'radio_pinv'):
            self.radio_pinv.setChecked(True) 
        
        if hasattr(self, 'comboBox_test_select'):
            self.comboBox_test_select.setEnabled(False)
        
        if hasattr(self, 'groupBox_recall'):
            self.groupBox_recall.setTitle("回憶結果")

    # 連接 UI 元件的訊號到對應的槽函數
    def connect_signals(self):
        # 控制面板連接
        self.pushButton_train.clicked.connect(self.handle_train_click)
        
        # 回憶結果區連接
        self.comboBox_test_select.currentIndexChanged.connect(self.handle_recall_test)
        
    ## 槽函數：訓練與數據載入
    # 點擊 '開始訓練' 按鈕時執行
    def handle_train_click(self):
        try:
            # 1. 讀取用戶選擇
            dataset_key = self.comboBox_dataset.currentText()
            
            # 判斷訓練規則
            if self.radio_hebb.isChecked():
                rule = "HEBB"
            elif self.radio_pinv.isChecked():
                rule = "PINV"
            else:
                QMessageBox.warning(self, "錯誤", "請選擇訓練規則 (Hebb Rule 或 Pseudo-Inverse)。")
                return

            # 2. 調用後端接口進行載入和訓練
            train_bipolar, num_train, num_test = self.connector.load_and_train(dataset_key, rule)
            
            self.current_patterns_count = num_train
            self.current_rows = self.connector.rows
            self.current_cols = self.connector.cols

            # 3. 更新記憶展示區
            self.display_memories(train_bipolar)
            
            # 4. 更新回憶測試下拉選單
            self.update_test_selector(num_test)
            
            QMessageBox.information(self, "訓練成功", f"網路已訓練完成 ({rule} 規則)。\n記憶模式數: {num_train}，測試模式數: {num_test}。")
            
        except Exception as e:
            QMessageBox.critical(self, "訓練錯誤", f"訓練或載入數據失敗：\n{e}")

    ## 槽函數：回憶與結果顯示
    # 當選擇測試模式時執行回憶 (接收 currentIndexChanged 傳來的 index)
    def handle_recall_test(self, index): 
        if self.connector.W is None:
            return # 尚未訓練
            
        # 直接使用傳入的 index
        test_index = index 
        if test_index == -1:
            return # 下拉選單為空

        try:
            # 暫時將 noise_percent 設為 0
            noisy_input, final_state, is_memory, matched_index, convergence_iter = self.connector.recall_test_pattern(
                test_index=test_index, 
                noise_percent=0 
            )
            
            # 1. 顯示輸入圖形
            self.display_pattern_on_view(self.graphicsView_input, noisy_input, self.current_rows, self.current_cols)
            
            # 2. 顯示回憶結果
            self.display_pattern_on_view(self.graphicsView_output, final_state, self.current_rows, self.current_cols)
            
            # 3. 更新回憶結果訊息
            if convergence_iter < 1000:
                self.label_status_convergence.setText(f"✅ 收斂於第 {convergence_iter} 輪迭代")
            else:
                self.label_status_convergence.setText("未收斂 (Max Iter)")

            if is_memory:
                self.label_status_match.setText(f"✅ 訓練記憶 #{matched_index + 1}")
            else:
                self.label_status_match.setText("❌ 虛偽/錯誤收斂")
            
        except Exception as e:
            QMessageBox.critical(self, "回憶錯誤", f"回憶測試失敗：\n{e}")
            
    ## UI 更新
    # 根據載入的測試模式數量更新下拉選單
    def update_test_selector(self, num_test):
        self.comboBox_test_select.clear()
        if num_test > 0:
            items = [f"測試模式 #{i + 1}" for i in range(num_test)]
            self.comboBox_test_select.addItems(items)
            self.comboBox_test_select.setEnabled(True)
        else:
            self.comboBox_test_select.setEnabled(False)
            
    # 在 widget_memories 中繪製所有記憶模式 (使用QLabel + QPixmap)
    def display_memories(self, train_bipolar):
        if not hasattr(self, 'widget_memories'):
            return
            
        # 清除舊的佈局和內容 
        if self.widget_memories.layout():
            QWidget().setLayout(self.widget_memories.layout())
        
        M = train_bipolar.shape[1]

        # 設置尺寸參數 (這是給 QLabel 設置的最小尺寸)
        MEM_SIZE = 100 # 讓每個圖案顯示為 100x100 像素
        PLOT_COLS = min(M, 5)
        
        grid_layout = QGridLayout(self.widget_memories)
        grid_layout.setSpacing(5) 
        
        for i in range(M):
            pattern_vector = train_bipolar[:, i]
            
            # 創建 QPixmap 圖像
            pixmap = create_pixmap_from_pattern(
                pattern_vector, self.current_rows, self.current_cols, size=MEM_SIZE
            )
            
            # 創建 QLabel 來顯示圖像
            img_label = QLabel(self.widget_memories)
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignCenter) # 圖像居中
            img_label.setFixedSize(QSize(MEM_SIZE, MEM_SIZE)) # 鎖定 QLabel 尺寸
            
            # 創建編號 Label
            label = QLabel(f"#{i + 1}")
            label.setAlignment(Qt.AlignCenter)

            # 創建一個垂直佈局 VBox
            vbox = QVBoxLayout()
            vbox.addWidget(label)
            vbox.addWidget(img_label) # 顯示圖像的 Label
            
            container = QWidget(self.widget_memories)
            container.setLayout(vbox)

            row = i // PLOT_COLS
            col = i % PLOT_COLS
            grid_layout.addWidget(container, row, col)

        self.widget_memories.setLayout(grid_layout)


    # 繪製單個模式到指定的 QGraphicsView 上
    def display_pattern_on_view(self, view: QGraphicsView, pattern_vector, rows, cols):
        # 消除 QGraphicsView 內建的邊界和樣式
        view.setFrameStyle(QFrame.NoFrame) # 移除邊框
        view.setLineWidth(0) # 確保邊框寬度為零

        pattern_unipolar = unipolarize(pattern_vector).reshape(rows, cols)
        
        scene = QGraphicsScene()
        view.setScene(scene)
    
        cell_size = 10
        
        for r in range(rows):
            for c in range(cols):
                value = pattern_unipolar[r, c]
                color = QColor(Qt.white) if value == 1 else QColor(Qt.black)
                brush = QBrush(color)
                
                rect = QRectF(c * cell_size, r * cell_size, cell_size, cell_size)
                # 使用 QColor() 建立畫筆
                scene.addRect(rect, QColor(Qt.gray), brush)

        # 設置 Scene 的邊界 (確保內容是 100x100)
        SCENE_WIDTH = cols * cell_size
        SCENE_HEIGHT = rows * cell_size
        scene.setSceneRect(0, 0, SCENE_WIDTH, SCENE_HEIGHT)
                
        view.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

# 將雙極性模式轉換為 QPixmap
def create_pixmap_from_pattern(pattern_vector, rows, cols, size=100):
    """
    Args:
        pattern_vector: Bipolar (-1, 1) vector.
        rows, cols: 矩陣尺寸。
        size: 期望的輸出像素尺寸 (例如 100x100)。
    """
    # 轉換回 (0, 1)
    pattern_unipolar = unipolarize(pattern_vector).reshape(rows, cols)
    
    # 將 (0, 1) 轉換為 (0, 255) 灰度圖像數據
    # 0 (黑色) -> 0
    # 1 (白色) -> 255
    img_data = pattern_unipolar * 255 
    img_data = img_data.astype(np.uint8)
    
    # 轉換為 QImage
    height, width = img_data.shape
    q_image = QImage(img_data.data, width, height, width, QImage.Format_Grayscale8)
    
    # 轉換為 QPixmap，並縮放至目標大小
    pixmap = QPixmap.fromImage(q_image)
    
    # 縮放至目標尺寸
    return pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.FastTransformation)

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        ui_file = "hopfield.ui"
        # 檢查 UI 文件是否存在
        if not os.path.exists(ui_file):
            print(f"錯誤：找不到 UI 文件 '{ui_file}'。請將 UI 文件放在執行目錄中。")
            sys.exit(1)
        window = HopfieldGUI()
        window.show()
        sys.exit(app.exec_())
    
    except Exception as e:
        # 捕捉所有未被處理的異常並打印
        print(f"*** 程式啟動失敗或發生未捕捉的異常 ***")
        print(f"錯誤類型: {type(e).__name__}")
        print(f"錯誤訊息: {e}")
        # 嘗試顯示一個彈窗，如果 Qt 系統允許
        try:
             QMessageBox.critical(None, "致命錯誤", f"程式啟動失敗，請檢查控制台。\n錯誤: {e}")
        except:
             pass
        sys.exit(1)