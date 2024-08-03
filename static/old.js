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

    const response = await fetch('/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
    });

    if (response.ok) {
        const data = await response.json();
        document.getElementById('chat-output').innerHTML += `<p><strong>User:</strong> ${message}</p>`;
        document.getElementById('chat-output').innerHTML += `<p><strong>Chatbot:</strong> ${data.response}</p>`;
        document.getElementById('chat-input').value = '';
    } else {
        const error = await response.json();
        document.getElementById('chat-output').innerHTML += `<p><strong>Error:</strong> ${error.error}</p>`;
    }
});
