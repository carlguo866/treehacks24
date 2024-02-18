document.addEventListener('DOMContentLoaded', function() {
    const startRecordingBtn = document.getElementById('startRecording');
    const stopRecordingBtn = document.getElementById('stopRecording');
    const modal = document.getElementById('modal');
    const closeBtn = document.querySelector('.close');
    let mediaRecorder;
    let audioChunks = [];

    // Function to query the ScamLLM model
    async function queryScamLLM(transcribedText) {
        const API_TOKEN = 'Your_Hugging_Face_API_Token'; // Replace with your actual Hugging Face API token
        const response = await fetch("https://api-inference.huggingface.co/models/phishbot/ScamLLM", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${API_TOKEN}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({inputs: transcribedText})
        });
        const result = await response.json();
        return result;
    }

    // Function to show the modal with scam likely message
    const showScamModal = () => {
        modal.style.display = 'block';
        modal.innerHTML = '<p>Scam Likely Detected!</p>';
    };

    // Function to close the modal
    const closeModal = () => {
        modal.style.display = 'none';
    };

    // Add event listener for the close button on the modal
    closeBtn.addEventListener('click', closeModal);

    // Start recording when the startRecordingBtn is clicked
    startRecordingBtn.addEventListener('click', async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Microphone access is not supported by this browser.');
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.start();
        } catch (err) {
            console.error('Error accessing the microphone', err);
        }
    });

    // Stop recording and process the audio when the stopRecordingBtn is clicked
    stopRecordingBtn.addEventListener('click', () => {
        if (mediaRecorder) {
            mediaRecorder.stop();

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                await sendAudioToServer(audioBlob);
            };
        }
    });

    // Send audio to the server for transcription with Whisper
    async function sendAudioToServer(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob);

        // Replace 'YOUR_SERVER_ENDPOINT' with your actual server endpoint for processing audio with Whisper
        const response = await fetch('YOUR_SERVER_ENDPOINT', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            console.error('Failed to send audio to server');
            return;
        }

        const { transcription } = await response.json();
        checkForScams(transcription);
    }

    // Check the transcription for potential scams
    async function checkForScams(transcribedText) {
        const scamResult = await queryScamLLM(transcribedText);
        if (scamResult && scamResult.length > 0 && scamResult[0].label === "LABEL_1" && scamResult[0].score > 0.6) {
            showScamModal();
        }
    }
});
