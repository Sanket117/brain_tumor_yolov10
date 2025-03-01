// Display welcome message in chatbot on first load with animation
window.addEventListener('DOMContentLoaded', function () {
    const chatOutput = document.getElementById('chat-output');

    // Create and append the welcome message
    const welcomeMessage = document.createElement('p');
    welcomeMessage.classList.add('bot-message', 'animated-message');
    chatOutput.appendChild(welcomeMessage);

    // Add typing effect and fade-in animation for the welcome message
    const welcomeText = "Hello, I'm Jaggu. I'm an AI assistant. How can I help you today? 😄";
    typeTextWithAnimation(welcomeMessage, welcomeText, 50);
});

// Function to handle form submission for predicting brain tumor
document.getElementById('upload-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const fileInput = document.getElementById('file');
    const loadingIndicator = document.getElementById('loading');
    const uploadedImage = document.getElementById('uploaded-image');
    const resultImage = document.getElementById('result-image');
    const captionElement = document.getElementById('caption');

    if (fileInput.files.length === 0) {
        captionElement.innerHTML = `<div class="alert alert-warning">Please upload a file before submitting.</div>`;
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    loadingIndicator.style.display = 'block'; // Show loading indicator

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        loadingIndicator.style.display = 'none'; // Hide loading indicator

        if (response.ok) {
            const resultData = await response.json();

            // Display the uploaded image
            uploadedImage.src = URL.createObjectURL(fileInput.files[0]);
            uploadedImage.style.display = 'block';

            // Display the result image
            resultImage.src = resultData.annotated_image;
            resultImage.style.display = 'block';

            // Display the AI-generated analysis
            captionElement.innerHTML = `
                <h3>YOLOv10 Summary</h3>
                <p>${resultData.yolo_summary}</p>
                <h3>AI Analysis</h3>
                <p>${resultData.ai_analysis}</p>
            `;
        } else {
            const error = await response.json();
            captionElement.innerHTML = `<div class="alert alert-danger">${error.error}</div>`;
        }
    } catch (error) {
        loadingIndicator.style.display = 'none';
        console.error('Error:', error);
        captionElement.innerHTML = `<div class="alert alert-danger">An error occurred while processing the file.</div>`;
    }
});

// Function to handle chatbot interactions
document.getElementById('send-btn').addEventListener('click', async function (event) {
    event.preventDefault();

    const chatInput = document.getElementById('chat-input');
    const chatOutput = document.getElementById('chat-output');
    const message = chatInput.value.trim();

    if (message === '') return;

    // Append user message to chat
    addMessage(message, 'user-message');
    chatInput.value = ''; // Clear input field

    const typingIndicator = document.createElement('p');
    typingIndicator.classList.add('typing-indicator');
    typingIndicator.textContent = 'Typing...';
    chatOutput.appendChild(typingIndicator);

    try {
        const response = await fetch('/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        });

        typingIndicator.remove();

        if (response.ok) {
            const data = await response.json();
            addMessage(data.response, 'bot-message');
        } else {
            const error = await response.json();
            addMessage(`Error: ${error.error}`, 'bot-message');
        }
    } catch (error) {
        typingIndicator.remove();
        console.error('Error:', error);
        addMessage(`Error: ${error.message}`, 'bot-message');
    }
});

// Helper function to add a message to the chat
function addMessage(text, className) {
    const chatOutput = document.getElementById('chat-output');
    const messageElement = document.createElement('p');
    messageElement.classList.add(className, 'animated-message'); // Add animation class
    messageElement.textContent = '';
    chatOutput.appendChild(messageElement);
    typeTextWithAnimation(messageElement, text, 50);
}

// Helper function to type out text dynamically with animation
function typeTextWithAnimation(element, text, typingSpeed = 50) {
    let i = 0;
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, typingSpeed);
        }
    }
    type();
}
