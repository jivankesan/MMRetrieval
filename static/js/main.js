document.getElementById('uploadBtn').addEventListener('click', async () => {
    const files = document.getElementById('fileInput').files;
    const formData = new FormData();

    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    try {
        const response = await axios.post('/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
        alert(response.data.status);
    } catch (error) {
        console.error(error);
    }
});

document.getElementById('searchBtn').addEventListener('click', async () => {
    const query = document.getElementById('searchQuery').value;

    try {
        const response = await axios.post('/search', { query });
        const results = response.data;

        let resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '';  // Clear previous results

        results.forEach(result => {
            let resultItem = document.createElement('div');
            resultItem.className = 'box';
            resultItem.innerHTML = `
                <p><strong>Filename:</strong> ${result.filename}</p>
                <p><strong>Action:</strong> ${result.action}</p>
                <p><strong>Start Time:</strong> ${result.start_time}</p>
                <p><strong>End Time:</strong> ${result.end_time}</p>
                <a href="/uploads/${result.filename}" target="_blank">Download Video</a>
            `;
            resultsDiv.appendChild(resultItem);
        });
    } catch (error) {
        console.error(error);
    }
});