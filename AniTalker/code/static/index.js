$(document).ready(function() {
    $('#startTalk').on('click', startRecording);
    $('#stopTalk').on('click', stopRecording);
    $('#submit').on('click', submitTalk);
    $('.box img').on('click', uploadimg);
    $('input[type=file]').on('change', setImg);
    let showimg = "../static/images/aiface2.png"
    let defaultimg = "./aiface2.png"
    $('.box img').attr("src",showimg);
    $('.box video').hide();

    let recognition;
    let isRecording = false;
    
    function setImg(){
        img = $('input[type=file]')[0].files[0];
        if(img){
            const reader = new FileReader();
            reader.onload = (e) =>{
                console.log(e.target.result)
                $('.box img').attr("src",e.target.result);
            }
            reader.readAsDataURL(img);
        }
    }

    function uploadimg(){
        $('input[type=file]').click();
    }

    function startRecording() {
        if (!isRecording) {
            recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = 'en';
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;

            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                $('#talked').val(function(index, value) {
                    return value + transcript + '\n';
                });
            };

            recognition.onerror = function(event) {
                console.error('Speech recognition error detected: ' + event.error);
            };

            recognition.onend = function() {
                isRecording = false;
            };

            recognition.start();
            isRecording = true;
        }
    }

    // function chinese(){
    //     let url = "http://127.0.0.1:5000/api/chinese";

    // }

    function stopRecording() {
        if (isRecording && recognition) {
            recognition.stop();
            isRecording = false;
        }
    }

    // function translateText(text) {
    //     const targetUrl = 'https://api.mymemory.translated.net/get';
    //     const url = new URL(targetUrl);
    //     url.searchParams.append("q", text);
    //     url.searchParams.append("langpair", `en|zh`);
    
    //     return new Promise((resolve, reject) => {
    //         $.ajax({
    //             url: url.toString(),
    //             method: "GET",
    //             headers: {
    //                 "Content-Type": "application/json"
    //             },
    //             success: function(response) {
    //                 if (response.responseData) {
    //                     resolve(response.responseData.translatedText);
    //                 } else {
    //                     reject("無效的翻譯結果");
    //                 }
    //             },
    //             error: function(xhr, status, error) {
    //                 console.error("翻譯請求失敗：", error);
    //                 reject(`翻譯請求失敗 - 狀態碼：${xhr.status}`);
    //             }
    //         });
    //     });
    // }
    

    
    // 翻譯函數，將英文翻譯為中文
function translateText(text) {
    const targetUrl = 'https://api.mymemory.translated.net/get';
    const langpair = 'en|zh'; // 翻譯從英文到中文

    const url = `${targetUrl}?q=${encodeURIComponent(text)}&langpair=${langpair}`;

    return fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.responseStatus === 200) {
                return data.responseData.translatedText; // 返回翻譯後的文本
            } else {
                throw new Error("Translation failed");
            }
        })
        .catch(error => {
            console.error('Translation Error:', error);
            return "Translation failed"; // 若翻譯失敗，返回錯誤信息
        });
}

    
    
    function submitTalk() {
        let img = defaultimg;
        let url = "http://127.0.0.1:5000/api/generate_video";
        fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                img: img,
                text: $('#talked').val()
            })
        })
        .then(response => response.json())
        .then(data => {
            if(data.status != 200){
                alert(data.msg);
                returns
            }
            let str = "AIres:"+data.result+"\n";
            // str += "\n中文:" + translateText(data.result);
            $('#res').val(str);
            $('.box img').hide();
            $('.box video').show();
            $('.box video').attr("src",data.video_path);
            $('.box video')[0].play();
       // 執行翻譯操作
       return translateText(data.result); // 傳遞 AI 回應給翻譯函數
    })
    .then(translatedText => {
        // 在翻譯完成後更新翻譯結果到 #res
        let currentStr = $('#res').val(); // 獲取當前的值
        currentStr += "\n中文: " + translatedText; // 添加翻譯結果
        $('#res').val(currentStr); // 更新到 #res
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

    // function submitTalk() {
    //     let img = defaultimg;
    //     let url = "http://127.0.0.1:5000/api/generate_video";
    //     let translateUrl = "http://127.0.0.1:5000/api/translate";
    
    //     // 第一步：生成視頻
    //     fetch(url, {
    //         method: "POST",
    //         headers: {
    //             "Content-Type": "application/json"
    //         },
    //         body: JSON.stringify({
    //             img: img,
    //             text: $('#talked').val()
    //         })
    //     })
    //     .then(response => response.json())
    //     .then(data => {
    //         if (data.status != 200) {
    //             alert(data.msg);
    //             return;
    //         }
    
    //         let str = "AIres:" + data.result + "\n";
    //         $('#res').val(str); // 更新結果到 #res
    //         $('.box img').hide();
    //         $('.box video').show();
    //         $('.box video').attr("src", data.video_path);
    //         $('.box video')[0].play();
    
    //         // 第二步：翻譯
    //         return fetch(translateUrl, {
    //             method: "POST",
    //             headers: {
    //                 "Content-Type": "application/json"
    //             },
    //             body: JSON.stringify({
    //                 text: data.result // 翻譯 AI 的結果
    //             })
    //         });
    //     })
    //     .then(response => response.json())
    //     .then(data => {
    //         if (data.status != 200) {
    //             alert(data.msg);
    //             return;
    //         }
    
    //         // 更新翻譯結果到 #res
    //         let currentStr = $('#res').val(); // 獲取當前的值
    //         currentStr += "\n中文:" + data.translatedText; // 添加翻譯結果
    //         $('#res').val(currentStr); // 更新到 #res
    //     })
    //     .catch(error => {
    //         console.error('Error:', error);
    //     });
    // }
    
    



        

});
