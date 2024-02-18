document.addEventListener('DOMContentLoaded', function() {
    const startRecordingBtn = document.getElementById('startRecording');
    const stopRecordingBtn = document.getElementById('stopRecording');
    const modal = document.getElementById('modal');
    const closeBtn = document.querySelector('.close');
    require('dotenv').config();
    let mediaRecorder;
    let audioChunks = [];

    // Function to query the ScamLLM model
    async function queryScamLLM(transcribedText) {
        const API_TOKEN = process.env.API_TOKEN; // Replace with your actual Hugging Face API token
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

    // Add event listener to start recording when the button is clicked
    startRecordingBtn.addEventListener('click', async () => {
        // Your recording logic here
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
        // Convert Blob to base64 as OpenAI's API might require the audio in base64 format
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = async () => {
            const base64Audio = reader.result.split(',')[1]; // Remove the Data URL part
    
            const OPENAI_API_URL = 'https://api.openai.com/v1/whisper'; // Hypothetical URL
            const OPENAI_API_KEY = process.env.OPENAI_API_KEY; // Securely manage this token
    
            try {
                const response = await fetch(OPENAI_API_URL, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${OPENAI_API_KEY}`,
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        model: "whisper-1", // Specify the model you wish to use
                        audio: base64Audio,
                    }),
                });
    
                if (!response.ok) {
                    console.error('Failed to transcribe audio with Whisper');
                    return;
                }
    
                const { text } = await response.json(); // Adjust based on the actual API response
                console.log(text); // Do something with the transcription
            } catch (error) {
                console.error('Error transcribing audio with Whisper:', error);
            }
        };
    }
    

    // Check the transcription for potential scams
    async function checkForScams(transcribedText) {
        const scamResult = await queryScamLLM(transcribedText);
        if (scamResult && scamResult.length > 0 && scamResult[0].label === "LABEL_1" && scamResult[0].score > 0.6) {
            showScamModal();
        }
    }
});
