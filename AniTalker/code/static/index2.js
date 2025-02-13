$(document).ready(function () {
    $('#talkButton').on('click', toggleRecording);
    $('.avatar-box').on('click', uploadimg);
    $('input[type=file]').on('change', setImg);

    $('.btn').hover(
        function () { $(this).find('i').addClass('fa-bounce'); },
        function () { $(this).find('i').removeClass('fa-bounce'); }
    );
    let showing = "../static/images/aiface2.png"
    let defaultimg = "./aiface2.png";
    $('.avatar-box img').attr("src", defaultimg);
    let recognition;
    let isRecording = false;
    let currentTranscript = '';

    function setImg() {
        let img = $('input[type=file]')[0].files[0];
        if (img) {
            const reader = new FileReader();
            reader.onload = (e) => {
                $('.avatar-box img').attr("src", e.target.result);
            }
            reader.readAsDataURL(img);
        }
    }

    function uploadimg() {
        $('input[type=file]').click();
    }

    function toggleRecording() {
        const $button = $('#talkButton');
        const $icon = $button.find('i');

        if (!isRecording) {
            $button.removeClass('btn-success').addClass('btn-danger');
            $icon.removeClass('fa-microphone').addClass('fa-stop');
            $button.find('span').text('停止說話');

            recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = 'en';
            recognition.interimResults = true;
            recognition.maxAlternatives = 1;

            recognition.onresult = function (event) {
                let transcript = '';
                for (let i = 0; i < event.results.length; i++) {
                    const result = event.results[i];
                    if (result.isFinal) {
                        transcript += result[0].transcript;
                    }
                }
                if (transcript) {
                    currentTranscript = transcript;
                    $('#messageInput').val(currentTranscript);
                }
            };

            recognition.onerror = function (event) {
                console.error('語音識別錯誤: ' + event.error);
                alert('語音識別發生錯誤，請重試。');
                resetButton();
            };

            recognition.onend = function () {
                if (isRecording) {
                    recognition.start();
                }
            };


            recognition.start();
            isRecording = true;

        } else {
            if (recognition) {
                recognition.stop();
            }
            if (currentTranscript) {
                submitTalk();
            }
            resetButton();
            currentTranscript = '';
        }
    }

    function resetButton() {
        const $button = $('#talkButton');
        const $icon = $button.find('i');

        isRecording = false;
        $button.removeClass('btn-danger').addClass('btn-success');
        $icon.removeClass('fa-stop').addClass('fa-microphone');
        $button.find('span').text('開始說話');
    }

    function addMessage(text, isUser = false) {
        const time = new Date().toLocaleTimeString();
        const avatarSrc = isUser ? './static/images/tm.jpg' : '../static/images/aiface2.png';
        const messageHtml = `
            <div class="chat-message ${isUser ? 'user' : 'ai'}">
                <div class="message-avatar">
                    <img src="${avatarSrc}" >
                </div>
                <div class="message-content">
                    <div class="original-text">${text}</div>
                </div>
                <div class="message-time">${time}</div>
            </div>
        `;
        $('#chatContainer').append(messageHtml);
        $('#chatContainer').scrollTop($('#chatContainer')[0].scrollHeight);
    }

    function submitTalk() {
        let userText = $('#messageInput').val();
        if (userText.trim() === '') return;

        addMessage(userText, true);
        $('#messageInput').val('');

        let img = defaultimg;
        let url = "http://127.0.0.1:5000/api/generate_video";

        fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                img: img,
                text: userText
            })
        })
            .then(response => response.json())
            .then(data => {
                if (data.status != 200) {
                    alert(data.msg);
                    return Promise.reject('API error');
                }

                addMessage(data.result);

                $('.box img').remove();
                $('#video').show();
                $('#video').attr("src", data.video_path);
                $('#video')[0].play();
            })
    }
});