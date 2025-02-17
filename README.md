# Speak_Docker
本次是使用Python 3.9.0 環境結合OpenAI開發的口說教學學堂
### 1.先決條件
具備OpenAI的key

首先，下載GIT LFS，並初始化
mac版本：
```bash
brew install git-lfs
git lfs install
```
windows版本：
官網下載https://git-lfs.com/
```bash
git lfs install
```


### 2.本地構建
使用git clone方式將檔案拉到桌面
```bash
git clone https://github.com/ChasDi/Speak_Docker.git
cd Speak_Docker
```

### 3.配置
首先，需要配置位於`AniTalker/.env`中，OpenAI的環境變數：
```bash
OPEN_API_KEY="apikey"
```

# 下載模型檔
mac版本：
```bsah
git clone https://huggingface.co/taocode/anitalker_ckpts
mv anitalker_ckpts ./AniTalker
cd ./AniTalker
mv anitalker_ckpts ckpts
```
windows版本：
```bsah
git clone https://huggingface.co/taocode/anitalker_ckpts
move anitalker_ckpts ./AniTalker
cd AniTalker
#重新命名檔名
ren anitalker_ckpts ckpts
cd ..
```

# Docker
### 方法一：構建映像檔並啟動Docker
```bsah
docker compose up --build
```
### 方法二：
#### 2-1構建映像檔
```bash
docker build -t image_name .
```
#### 2-2啟動docker
```bash
docker run -p 5000:5000 image_name
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
python ./AniTalker/code/webgui_copy2.py
```
