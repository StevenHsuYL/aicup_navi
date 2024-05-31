# aicup_navi
以生成式AI建構無人機於自然環境偵察時所需之導航資訊競賽 II － 導航資料生成競賽
Generative-AI Navigation Information Competition

## 背景說明
主題說明：[AICUP 競賽網頁](<https://tbrain.trendmicro.com.tw/Competitions/Details/35>)

如需執行本專案程式碼，需先至競賽網頁下載訓練及測試資料集。下載完成後，將檔案儲存於 Google Drive 根目錄下的 `AICUP_2024` 資料夾內。

### 執行環境
此專案須於 Google Colab 環境執行，並自行選擇適合的 GPU 做為運算資源。

執行過程中會需要由從使用者的 Google Drive 下載資料集，請事先將資料集儲存於 Drive 資料夾。

儲存路徑（Colab 顯示路徑）：

```
"/content/drive/MyDrive/AICUP_2024/35_Competition 2_Training dataset_V3.zip"
"/content/drive/MyDrive/AICUP_2024/35_Competition 2_public testing dataset.zip"
"/content/drive/MyDrive/AICUP_2024/35_Competition 2_Private Test Dataset.zip"
```

## 執行步驟
以下說明本專案程式碼之執行流程。執行步驟包括資料前處理 (pre-processing)、模型訓練、資料後處理 (post-processing)、模型驗證 (validation)、測試 (inferecne) 等階段。

### 1. 匯入模組及資料集
* 執行 `Download training dataset` 的 code cells，連結 Google Drive 並下載練資料集。

* 執行 `Import Packages` code cell，匯入所需的模組。

### 2. 資料前處理 (pre-processing)
此步驟分為兩個階段：標註影像 (label image) 前處理，以及輸入影像 (input image) 前處理。

* 標註影像 (label image) 前處理：將訓練資料集中的 label image，由四個白色邊界所構成的標註資料，轉換為內部填滿白色的標註方式，以提升訓練成效。處理結果會儲存於 `/content/Training_dataset/label_img_new/`
* 輸入影像 (input image) 前處理：將訓練資料集中的 input image，以傳統方法大致提取河流及道路的特徵，並標註於輸入影像中，以提升訓練成效。處理結果會儲存於 `/content/Training_dataset/img_new/`
* 前處理完成後，執行第三個 code cell 確認處理後的影像數量（輸出 4321 表示數量正確）。
  ```
  !ls -l '/content/Training_dataset/label_img_new/' | wc -l
  !ls -l '/content/Training_dataset/img_new/'  | wc -l
  ```

### 3. 資料設定
* 由於本專案中，河流及道路的模型是分開訓練的，所以一次訓練、驗證及測試的過程，只會使用其中一種類型的影像資料，參數必須在此設定。河流的參數為 `'_RI'`，道路的參數為 `'_RO'`。在此先設定為 `'_RI'`。

  `category = '_RI'`
* 資料參數設定後，執行 code cell，將訓練資料集中的河流影像，分為訓練及驗證集，比例為 80% 及 20%。
* 執行 `Dataset, DataLoader, and Transforms` 的 code cells，定義 Dataset 的資料擴增及轉換方式。
  
### 4. 定義損失函數 (loss function)
* 使用 binary cross entropy `torch.nn.BCEWithLogitsLoss()` 做為訓練時的損失函數。

  `pos_weight = torch.ones([1, 128, 128]) * 0.5` 是為了減少 false positive rate，提高 precision。

  請參考：[BCEWithLogitsLoss](<https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html>)


### 5. 模型訓練
* 定義訓練函式：使用 **Adam** 做為 optimizer，並使用 learning rate scheduler 增加訓練成效。
* 套用 **SegFormer** 做為訓練模型，並載入預訓練權重 (pretrained weights)，以進行影像分割任務的訓練。

  `model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")`

  請參考：[SegFormer](<https://huggingface.co/docs/transformers/model_doc/segformer>)
* 載入資料集之 `DataLoader`，設定 `epoch_num=30, lr=1e-4`，對模型進行微調 (fine tuning)。

  `train(model, tr_loader, val_loader, save_path, epoch_num=30, lr=1e-4)`
* 河流資料的模型權重會儲存於 `'/content/models_RI/best_model.pth'`。

### 6. 模型推論 (inference)
* 下載測試資料集，並對輸入影像進行前處理。處理結果會儲存於 `/content/ts/img_new/`
* 以下執行順序很重要，正確執行完才能得到測試資料的推論結果：
  * 先設定 `Test = False`，使用 validation set 來尋找最理想的後處理 (post-processing) 參數：
  * 載入訓練權重，並對 validation set 執行 block mask prediction。
  * 搜尋最好的 Gaussian kernel size 及 binary threshold，這是為了讓後處理完的 Mean F-measure (FM) score 可以達到最佳的結果。
  * 以搜尋後的參數，對 validation set 執行後處理，並計算 Mean FM score。此時顯示的兩個分數，第一個是 block mask 的分數，第二個是 edge mask 的分數，也就是競賽的 label 形式。
    ```
    Validation set FM score:
    {'meanfm(block)': 0.971661746570068}
    {'meanfm(edge)': 0.7705276127383758}
    ```
  * 算完分數後，回到先前的 code cell，將設定改為 `Test = True`，對 test set 進行推論。
  * 跳過搜尋參數的步驟，套用先前最佳的 Gaussian kernel size 及 binary threshold 參數，對測試集的預測結果進行後處理，將 block mask 轉換為 edge mask。
    
* 執行至此，已完成測試集中河流影像的推論。
* 回到步驟 3，將類別改為 `'_RO'`，重新執行步驟 3~6，訓練道路影像的模型，即可得到測試集中道路影像資料的推論。

  `category = '_RO'`

### 7. 輸出測試結果
* 如果測試集中兩種類型（河流及道路）的影像資料都已經推論完成，即可執行最後一個 code cell，將推論結果輸出。
* 檔案壓縮前須確認輸出影像數量和輸入影像數量相同。
  ```
  !ls -l '/content/ts/img/' | wc -l
  !ls -l '/content/ts/result/' | wc -l
  ```

## 補充說明
由於競賽評分使用 Mean F-measure，且 `beta = 0.3`，所以須降低偽陽性率，提升 precision，以得到較好的 FM score，但是此輸出方式線條較暗，使人眼較不易直接觀察影像結果。

在實際應用時，若需清晰的輸出模式，可在後處理時設定 `application = True`，此種輸出會有較清楚的線條可供人眼辨識。

## 作者

TEAM_5447 / **Khóo Hō-Sûn**

## License

This project is licensed under the GNU AGL License - see the [LICENSE](LICENSE) file for details.
