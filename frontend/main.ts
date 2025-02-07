// When the user clicks the submit button or presses Enter, create a new conversation message pair.
document.getElementById("submit-btn")?.addEventListener("click", async function (e: Event) {
  e.preventDefault();
  
  const queryInput = document.getElementById("query") as HTMLTextAreaElement;
  const queryText = queryInput.value.trim();
  if (!queryText) {
    alert("Please enter a valid query.");
    return;
  }

  // Create a message pair container
  const messagePair = document.createElement("div");
  messagePair.className = "message-pair";

  // Create the user query bubble (aligned right)
  const queryBubble = document.createElement("div");
  queryBubble.className = "chat-bubble query-bubble";
  queryBubble.innerText = queryText;

  // Add an edit button to the user query bubble
  const editBtn = document.createElement("button");
  editBtn.className = "edit-button";
  editBtn.innerText = "Edit";
  editBtn.addEventListener("click", () => {
    const newText = prompt("Edit your query:", queryBubble.innerText);
    if (newText !== null && newText.trim() !== "") {
      queryBubble.innerText = newText;
      queryBubble.appendChild(editBtn); // reattach edit button
    }
  });
  queryBubble.appendChild(editBtn);

  // Create the response bubble (aligned left)
  const responseBubble = document.createElement("div");
  responseBubble.className = "chat-bubble response-bubble";

  // Append the query and response bubbles to the message pair
  messagePair.appendChild(queryBubble);
  messagePair.appendChild(responseBubble);

  // Append the message pair to the conversation container
  const conversation = document.getElementById("conversation");
  if (conversation) {
    // **Show the chat container after the first query is added**
    conversation.style.display = 'flex'; // Make the container visible
    conversation.appendChild(messagePair);
    conversation.scrollTop = conversation.scrollHeight;
  }

  // **Hide the header after the first query is sent**
  const header = document.querySelector(".header"); // Target the header element
  if (header) {
    header.style.display = "none"; // Hide the header
  }

  // Clear the query input for the next message
  queryInput.value = "";

  // Send the query to the backend and update the response bubble
  await sendQueryAndUpdateResponse(queryText, responseBubble);
});

// Event listener for "Enter" key to submit the query
document.getElementById("query")?.addEventListener("keydown", function (e: KeyboardEvent) {
  if (e.key === "Enter" && !e.shiftKey) { // Enter key (without Shift)
    e.preventDefault(); // Prevent new line in textarea
    document.getElementById("submit-btn")?.click(); // Trigger button click
  }
});

async function sendQueryAndUpdateResponse(query: string, responseBubble: HTMLDivElement): Promise<void> {
  responseBubble.innerText = "Loading...";
  try {
    const modelButton = document.getElementById("model-select") as HTMLButtonElement;
    const modelNameElem = modelButton.querySelector(".model-name") as HTMLDivElement;
    const model = modelNameElem.innerText;

    const contextButton = document.getElementById("context-select") as HTMLButtonElement;
    const contextNameElem = contextButton.querySelector(".context-name") as HTMLDivElement;
    const context = contextNameElem.innerText;

    const res = await fetch("http://127.0.0.1:8000/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, model, additional_context: context })
    });
    if (!res.ok) {
      throw new Error("Server error: " + res.statusText);
    }

    const data = await res.json();
    // Assuming the response structure: data.result.responses[0][2]
    const responseText: string = data.result.responses[0][2];

    // Gradually type out the response—preserving spaces and line breaks.
    responseBubble.innerText = "";
    let index = 0;
    const intervalId = setInterval(() => {
      responseBubble.innerText += responseText[index];
      index++;
      if (index === responseText.length) {
        clearInterval(intervalId);
      }
    }, 10);
  } catch (error: any) {
    responseBubble.innerText = "Error: " + error.message;
  }
}

// --- Model Selector Dropdown Logic ---
document.getElementById("model-select")?.addEventListener("click", function () {
  const modelDropdown = document.querySelector(".model-dropdown") as HTMLDivElement;
  const rect = this.getBoundingClientRect(); // Get button position

  // Check if there's enough space below to display the dropdown
  const spaceBelow = window.innerHeight - rect.bottom;
  const dropdownHeight = modelDropdown.offsetHeight;

  if (spaceBelow >= dropdownHeight) {
    modelDropdown.style.top = "100%"; // Position it below the button
    modelDropdown.style.bottom = "auto"; // Reset bottom
  } else {
    modelDropdown.style.bottom = `${window.innerHeight - rect.top}px`; // Position it above the button
    modelDropdown.style.top = "auto"; // Reset top
  }

  // Toggle dropdown visibility
  modelDropdown.style.display = modelDropdown.style.display === "block" ? "none" : "block";
});

document.querySelectorAll(".model-option").forEach((option) => {
  option.addEventListener("click", function () {
    const selectedModel = option.getAttribute("data-model");
    if (selectedModel) {
      const modelName = document.querySelector("#model-select .model-name") as HTMLDivElement;
      modelName.innerText = selectedModel;
    }
    const modelDropdown = document.querySelector(".model-dropdown") as HTMLDivElement;
    modelDropdown.style.display = "none";
  });
});

// --- Context Selector Dropdown Logic ---
document.getElementById("context-select")?.addEventListener("click", function () {
  const contextDropdown = document.querySelector(".context-dropdown") as HTMLDivElement;
  contextDropdown.style.display = contextDropdown.style.display === "block" ? "none" : "block";
});

document.querySelectorAll(".context-option").forEach((option) => {
  option.addEventListener("click", function () {
    const selectedContext = option.getAttribute("data-context");
    if (selectedContext) {
      const contextName = document.querySelector("#context-select .context-name") as HTMLDivElement;
      contextName.innerText = selectedContext;
    }
    const contextDropdown = document.querySelector(".context-dropdown") as HTMLDivElement;
    contextDropdown.style.display = "none";
  });
});
