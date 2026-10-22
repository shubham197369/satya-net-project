document.addEventListener('DOMContentLoaded', () => {
    const textTabBtn = document.getElementById('textTabBtn');
    const imageTabBtn = document.getElementById('imageTabBtn');
    const audioTabBtn = document.getElementById('audioTabBtn'); // Audio Tab Button

    const textSection = document.getElementById('text-section');
    const imageSection = document.getElementById('image-section');
    const audioSection = document.getElementById('audio-section'); // Audio Section Div
    const resultDiv = document.getElementById('result');

    // Tab Switching Logic (Text, Image & Audio)
    if (textTabBtn && imageTabBtn) {
        textTabBtn.addEventListener('click', () => {
            textTabBtn.classList.add('active');
            imageTabBtn.classList.remove('active');
            if(audioTabBtn) audioTabBtn.classList.remove('active');
            
            textSection.classList.add('active');
            imageSection.classList.remove('active');
            if(audioSection) audioSection.classList.remove('active');
            resultDiv.innerText = "Result yahan dikhega...";
        });

        imageTabBtn.addEventListener('click', () => {
            imageTabBtn.classList.add('active');
            textTabBtn.classList.remove('active');
            if(audioTabBtn) audioTabBtn.classList.remove('active');

            imageSection.classList.add('active');
            textSection.classList.remove('active');
            if(audioSection) audioSection.classList.remove('active');
            resultDiv.innerText = "Result yahan dikhega...";
        });

        if (audioTabBtn) {
            audioTabBtn.addEventListener('click', () => {
                audioTabBtn.classList.add('active');
                textTabBtn.classList.remove('active');
                imageTabBtn.classList.remove('active');

                audioSection.classList.add('active');
                textSection.classList.remove('active');
                imageSection.classList.remove('active');
                resultDiv.innerText = "Result yahan dikhega...";
            });
        }
    }

    // Text Verification Logic
    const checkBtn = document.getElementById('checkBtn');
    if (checkBtn) {
        checkBtn.addEventListener('click', async () => {
            const text = document.getElementById('userInput').value;
            
            if (!text) {
                resultDiv.innerHTML = "<span style='color: red;'>Kripya kuch text likhein!</span>";
                return;
            }

            resultDiv.innerHTML = "Checking with AI RAG Engine...";

            try {
                const response = await fetch('http://127.0.0.1:8000/verify-claim', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();
                
                resultDiv.innerHTML = `
                    <b>Verdict:</b> ${data.verdict}<br><br>
                    <b>Explanation:</b> ${data.explanation}
                `;
            } catch (error) {
                resultDiv.innerHTML = "<span style='color: red;'>Server se connect nahi ho paya! Check karein ki backend chal raha hai ya nahi.</span>";
            }
        });
    }

    // Image Verification Logic
    const checkImageBtn = document.getElementById('checkImageBtn');
    if (checkImageBtn) {
        checkImageBtn.addEventListener('click', async () => {
            const fileInput = document.getElementById('imageFile');
            
            if (!fileInput || fileInput.files.length === 0) {
                resultDiv.innerHTML = "<span style='color: red;'>Kripya pehle ek image select karein!</span>";
                return;
            }
            
            const formData = new FormData();
            formData.append("image_file", fileInput.files[0]);
            
            resultDiv.innerHTML = "Analyzing image and checking deepfakes...";
            
            try {
                const response = await fetch('http://127.0.0.1:8000/verify-image', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                resultDiv.innerHTML = `
                    <b>Verdict:</b> ${data.final_verdict}<br><br>
                    <b>Details:</b> ${data.final_explanation}
                `;
            } catch (error) {
                resultDiv.innerHTML = "<span style='color: red;'>Server se connect nahi ho paya! Check karein ki backend chal raha hai ya nahi.</span>";
            }
        });
    }

    // --- NEW AUDIO VERIFICATION LOGIC ---
    const checkAudioBtn = document.getElementById('checkAudioBtn');
    if (checkAudioBtn) {
        checkAudioBtn.addEventListener('click', async () => {
            const audioFileInput = document.getElementById('audioFile');
            
            if (!audioFileInput || audioFileInput.files.length === 0) {
                resultDiv.innerHTML = "<span style='color: red;'>Kripya pehle ek audio file select karein!</span>";
                return;
            }
            
            const formData = new FormData();
            formData.append("audio_file", audioFileInput.files[0]);
            
            resultDiv.innerHTML = "Analyzing audio voice clip for deepfakes...";
            
            try {
                const response = await fetch('http://127.0.0.1:8000/verify-audio', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                resultDiv.innerHTML = `
                    <b>Verdict:</b> ${data.final_verdict}<br><br>
                    <b>Details:</b> ${data.final_explanation}
                `;
            } catch (error) {
                resultDiv.innerHTML = "<span style='color: red;'>Server se connect nahi ho paya! Check karein ki backend chal raha hai ya nahi.</span>";
            }
        });
    }
});