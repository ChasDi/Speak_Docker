# Speak_Docker
## Python版本＝3.9.0
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
cd Scripts
activate
cd ..
```
### 2.更新pip：
 ``` bash
pip python -m pip install --upgrade pip
```
### 3.下載依賴檔：
``` bash
pip install -r requirements.txt
```
### 4.構建映像檔並啟動Docker：
``` bash
docker compose up --build
```
