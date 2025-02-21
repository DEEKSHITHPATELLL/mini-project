document.addEventListener('DOMContentLoaded', function() {
    const predictionText = document.getElementById('prediction-text');
    const clearBtn = document.getElementById('clear-btn');
    const gestureHistory = document.getElementById('gesture-history');
    let predictions = [];

    // Function to update prediction text with appropriate styling
    function updatePrediction(text) {
        if (text && text !== predictionText.textContent) {
            predictionText.textContent = text;
            
            // Add appropriate styling based on the message type
            if (text.includes("not recognized") || text.includes("No hand detected")) {
                predictionText.classList.remove('success');
                predictionText.classList.add('warning');
            } else if (text.includes("Confidence")) {
                predictionText.classList.remove('warning');
                predictionText.classList.add('success');
            }
            
            // Only add successful predictions to history
            if (text.includes("Confidence")) {
                addToHistory(text);
            }
        }
    }

    // Function to add prediction to history
    function addToHistory(text) {
        const timestamp = new Date().toLocaleTimeString();
        const historyItem = document.createElement('p');
        historyItem.textContent = `${timestamp}: ${text}`;
        gestureHistory.insertBefore(historyItem, gestureHistory.firstChild);

        // Keep only last 10 predictions
        predictions.unshift(text);
        if (predictions.length > 10) {
            predictions.pop();
            if (gestureHistory.lastChild) {
                gestureHistory.removeChild(gestureHistory.lastChild);
            }
        }
    }

    // Clear button functionality
    clearBtn.addEventListener('click', function() {
        predictionText.textContent = 'Waiting for gesture...';
        predictionText.classList.remove('success', 'warning');
        predictions = [];
        gestureHistory.innerHTML = '';
    });

    // Fetch predictions from backend
    function fetchPrediction() {
        fetch('/get_prediction')
            .then(response => response.json())
            .then(data => {
                if (data.prediction) {
                    updatePrediction(data.prediction);
                }
            })
            .catch(error => console.error('Error fetching prediction:', error));
    }

    // Poll for predictions every second
    setInterval(fetchPrediction, 1000);
});
