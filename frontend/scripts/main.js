document.addEventListener("DOMContentLoaded", () => {
    // --- 1. Get references ---
    const chatWindow = document.getElementById("chat-window");
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message-input");
    const sendButton = document.getElementById("send-button");
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const uploadButton = document.getElementById("upload-button");
    const uploadStatus = document.getElementById("upload-status");
    const llmSelect = document.getElementById("llm-select");
    const apiKeyInput = document.getElementById("api-key-input");
    
    const UPLOAD_URL = "http://127.0.0.1:8000/upload";
    const INGEST_URL = "http://127.0.0.1:8000/ingest";
    const CHAT_URL = "http://127.0.0.1:8000/chat";

    // --- 2. Handle Upload Form Submission ---
    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const file = fileInput.files[0];
        if (!file) {
            alert("Please select a file to upload.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        // --- Step 1: Upload the file ---
        uploadStatus.textContent = `Step 1/2: Uploading file "${file.name}"...`;
        uploadStatus.className = "upload-status processing";
        uploadButton.disabled = true;

        try {
            const uploadResponse = await fetch(UPLOAD_URL, {
                method: "POST",
                body: formData,
            });

            if (!uploadResponse.ok) {
                const errorData = await uploadResponse.json();
                throw new Error(errorData.detail || "File upload failed.");
            }
            const uploadData = await uploadResponse.json();
            console.log(uploadData.message);

            // --- Step 2: Run ingestion ---
            uploadStatus.textContent = "Step 2/2: Processing file... This may take a moment. The server will restart.";
            
            const ingestResponse = await fetch(INGEST_URL, {
                method: "POST"
            });
            
            if (!ingestResponse.ok) {
                const errorData = await ingestResponse.json();
                throw new Error(errorData.detail || "File processing (ingestion) failed.");
            }
            
            const ingestData = await ingestResponse.json();
            console.log(ingestData.message);

            // Success!
            uploadStatus.textContent = `File "${file.name}" processed! Server is restarting. Please wait a moment before chatting.`;
            uploadStatus.className = "upload-status success";
            
            // Add a bot message
            addMessageToChat(`New file "${file.name}" has been added. The library is now updated.`, "bot");

        } catch (error) {
            uploadStatus.textContent = `Error: ${error.message}`;
            uploadStatus.className = "upload-status error";
            console.error("Upload/Ingest error:", error);
        } finally {
            uploadButton.disabled = false;
        }
    });

    // --- 3. Handle Chat Form Submission (NO CHANGE) ---
    chatForm.addEventListener("submit", (event) => {
        event.preventDefault();
        const query = messageInput.value.trim();
        if (!query) return;

        const llmChoice = llmSelect.value;
        const apiKey = apiKeyInput.value.trim();
        
        if (llmChoice === "openai" && !apiKey) {
            alert("Please enter your OpenAI API key to use this model.");
            return;
        }

        addMessageToChat(query, "user");
        addMessageToChat("Thinking...", "loading");
        
        messageInput.value = "";
        sendButton.disabled = true;

        callChatAPI(query, llmChoice, apiKey);
    });

    // --- 4. Function to call the FastAPI backend (/chat) (NO CHANGE) ---
    async function callChatAPI(query, llmChoice, apiKey) {
        try {
            const response = await fetch(CHAT_URL, { 
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    llm_choice: llmChoice,
                    api_key: apiKey || null,
                }),
            });

            removeLoadingMessage();

            if (!response.ok) {
                const errorData = await response.json();
                addMessageToChat(`Error: ${errorData.detail || 'Something went wrong'}`, "bot");
            } else {
                const data = await response.json();
                addMessageToChat(data.answer, "bot");
            }

        } catch (error) {
            removeLoadingMessage();
            addMessageToChat("Network error. Is the backend server running?", "bot");
            console.error("Fetch error:", error);
        } finally {
            sendButton.disabled = false;
            messageInput.focus();
        }
    }

    // --- 5. Helper function to add a new message (NO CHANGE) ---
    function addMessageToChat(text, sender) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("chat-message", sender);
        const textElement = document.createElement("p");
        textElement.textContent = text;
        messageElement.appendChild(textElement);
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // --- 6. Helper function to remove the "Thinking..." message (NO CHANGE) ---
    function removeLoadingMessage() {
        const loadingMessage = document.querySelector(".chat-message.loading");
        if (loadingMessage) {
            chatWindow.removeChild(loadingMessage);
        }
    }
});