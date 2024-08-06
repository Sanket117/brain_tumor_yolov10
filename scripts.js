// Function to handle form submission for predicting brain tumor
document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const formData = new FormData();
    formData.append('file', document.getElementById('file').files[0]);

    document.getElementById('loading').style.display = 'block';  // Show loading indicator

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    document.getElementById('loading').style.display = 'none';  // Hide loading indicator

    if (response.ok) {
        const imageBlob = await response.blob();
        const imageObjectURL = URL.createObjectURL(imageBlob);

        // Display the uploaded image
        document.getElementById('uploaded-image').src = URL.createObjectURL(document.getElementById('file').files[0]);

        // Display the result image
        document.getElementById('result-image').src = imageObjectURL;

        // Optionally, display the caption
        const resultData = await response.json();
        document.getElementById('caption').innerHTML = `<div class="alert alert-info">${resultData.caption}</div>`;
    } else {
        const error = await response.json();
        document.getElementById('result').innerHTML = `<div class="alert alert-danger">${error.error}</div>`;
    }
});

// Function to handle chatbot interactions
document.getElementById('send-btn').addEventListener('click', async function(event) {
    event.preventDefault();
    const message = document.getElementById('chat-input').value.trim();

    if (message === '') return;

    // Append user message to chat
    addMessage(message, 'user-message');
    document.getElementById('chat-input').value = ''; // Clear input field

    const typingIndicator = document.createElement('p');
    typingIndicator.classList.add('typing-indicator');
    typingIndicator.textContent = 'Typing...';
    document.getElementById('chat-output').appendChild(typingIndicator);

    try {
        const response = await fetch('/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
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

function addMessage(text, className) {
    const messageElement = document.createElement('p');
    messageElement.classList.add(className);
    messageElement.textContent = '';
    document.getElementById('chat-output').appendChild(messageElement);
    typeText(messageElement, text);
}

function typeText(element, text) {
    let i = 0;
    const typingSpeed = 50; // Adjust typing speed
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, typingSpeed);
        }
    }
    type();
}
