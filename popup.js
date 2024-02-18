document.addEventListener('DOMContentLoaded', function() {
    const modal = document.getElementById('modal');
    const closeBtn = document.querySelector('.close');

    // Simulating spam detection
    const detectSpamAudio = () => {
        // Add your logic to detect spam audio here
        return true; // For demonstration purposes
    };

    // Function to show the modal
    const showModal = () => {
        modal.style.display = 'block';
    };

    // Function to close the modal
    const closeModal = () => {
        modal.style.display = 'none';
    };

    // Event listener for close button
    closeBtn.addEventListener('click', closeModal);

    // Simulate listening for audio and detecting spam
    setInterval(() => {
        if (detectSpamAudio()) {
            showModal();
        }
    }, 5000); // Check every 5 seconds (adjust as needed)
});