# Speak_Docker
本次是使用Python 3.9.0 環境結合OpenAI開發的口說教學助理
1.先決條件
     具備OpenAI的key
2.配著環境變數
     

## Docker
### 方法一：構建映像檔並啟動Docker
```bsah
docker compose up --build
```
### 方法二：
#### 2-1構建映像檔
```bash
docker build -t 
```
#### 2-2啟動docker
```bash

```
### 本地構建
### 使用git clone方式將檔案拉到桌面
```bash
git clone https://github.com/ChasDi/Speak_Docker.git
```
### 1.激活虛擬環境：
1-1.建立虛擬環境
``` bash
python -m venv .venv
```
mac版本：
``` bash
source .venv/bin/activate 
```
windows版本：
``` bash
.venv/Scripts/activate
```
### 2.更新pip：
 ``` bash
pip install --upgrade pip
```
### 3.下載依賴檔：
``` bash
pip install -r requirements.txt
```
### 4.執行：
```bash
python ./AniTalker/code/webgui_copy.py
```
