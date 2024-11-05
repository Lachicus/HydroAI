function previewImage(event) {
    const imagePreviewContainer = document.getElementById("imagePreviewContainer");
    const imagePreview = document.getElementById("imagePreview");

    imagePreview.src = URL.createObjectURL(event.target.files[0]);
    imagePreviewContainer.style.display = "block";
}

async function predictImage() {
    const modelSelect = document.getElementById("modelSelect").value;
    const fileInput = document.getElementById("imageUpload").files[0];
    if (!fileInput) {
        alert("Please upload an image");
        return;
    }

    // Show loading modal
    const loadingModal = document.getElementById("loadingModal");
    loadingModal.style.display = "flex"; // Display modal with flex to center it
    document.getElementById("resultsContainer").style.display = "none"; // Hide results initially

    const formData = new FormData();
    formData.append("file", fileInput);
    formData.append("model", modelSelect);

    try {
        const response = await fetch('/predict', {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();

        // Update the results in the UI
        document.getElementById("waterLevel").textContent = result.water_level || "N/A";
        document.getElementById("floodLabel").textContent = result.flood_label || "N/A";
        document.getElementById("allowedVehicles").textContent = result.allowed_vehicles.join(", ") || "N/A";
    } catch (error) {
        console.error('Error during prediction:', error);
        alert('An error occurred while processing your request.');
    } finally {
        // Hide loading modal and show results
        loadingModal.style.display = "none";
        document.getElementById("resultsContainer").style.display = "block"; // Show results again
    }
}
