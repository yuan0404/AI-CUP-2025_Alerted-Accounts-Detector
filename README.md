# AI 警示帳戶偵測系統

本專案為 **[AI CUP 2025 玉山人工智慧公開挑戰賽－AI偵探出任務，精準揪出警示帳戶！](https://tbrain.trendmicro.com.tw/Competitions/Details/40)** 的參賽作品。  
專案目標是從帳戶交易資料中偵測潛在警示帳戶，主要功能包括特徵建構、警示檢測，以及結果視覺化與輸出。

**隊伍編號**：TEAM_8666  
**隊長姓名**：趙堉安  
**隊員姓名**：金以凡、劉千慈、陳映竹

## 專案結構

- **Data/**：存放專案所需資料
  - `README.md`：資料夾說明
- **Preprocess/**：警示帳戶特徵建構
  - `features.py`：計算帳戶交易特徵
  - `features.jpg`：生成的特徵分布比較圖
  - `README.md`：資料夾說明
- **Model/**：警示帳戶偵測模型
  - `detector.py`：進行警示帳戶偵測
  - `detector.jpg`：生成的警示分數分布圖
  - `README.md`：資料夾說明
- `main.py`：主程式
- `result.csv`：最終警示帳戶檢測結果
- `requirements.txt`：套件需求
- `README.md`：專案說明

## 使用方式

### 1. 環境需求

- Python 版本：3.12.2
- 套件安裝：
  ```bash
  pip install -r requirements.txt
  ```

### 2. 資料準備

將交易資料放入 `Data` 資料夾，包含：

- `acct_transaction.csv`：帳戶交易資料
- `acct_alert.csv`：已知警示帳戶清單
- `acct_predict.csv`：待偵測帳戶清單

### 3. 執行程式

- 執行主程式：
  ```bash
  python main.py
  ```

### 4. 實驗結果

- `Preprocess/features.jpg`：特徵分布比較圖
- `Model/detector.jpg`：警示分數分布圖
- `result.csv`：警示帳戶檢測結果

## 程式說明（main.py）

- 功能：專案主程式，負責整合完整流程
- 具體步驟：
    1. 載入交易資料、已知警示帳戶清單及待偵測帳戶清單
    2. 為警示帳戶與待偵測帳戶計算交易特徵
    3. 設定模型參數（`config`）
        - `n_runs`：PU Bagging 迭代次數
        - `n_estimators`：隨機森林樹數
        - `max_depth`：樹的最大深度
        - `class_weight`：類別權重
        - `threshold`：判定警示帳戶的分數百分位
    4. 執行警示帳戶偵測模型，並對帳戶進行警示分數評估
    5. 輸出最終檢測結果（`result.csv`）
