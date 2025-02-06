document.getElementById("submit-btn")!.addEventListener("click", async function (e: Event) {
  e.preventDefault();
  
  const query = (document.getElementById("query") as HTMLTextAreaElement).value;
  const model = (document.getElementById("model-select") as HTMLSelectElement).value;
  const loadingDiv = document.getElementById("loading") as HTMLDivElement;
  const responseDiv = document.getElementById("response") as HTMLDivElement;

  // Debug log to check query and model
  console.log("Query:", query);
  console.log("Model:", model);

  if (!query.trim()) {
    responseDiv.innerText = "Please enter a valid query.";
    return;
  }

  // Show loading indicator and clear previous response.
  loadingDiv.style.display = "block";
  responseDiv.innerText = "";

  try {
    // Debugging the fetch request
    console.log("Sending request to backend...");
    const res = await fetch("http://127.0.0.1:8000/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        query: query,
        model: model // Send the selected model to the backend
      })
    });

    if (!res.ok) {
      throw new Error("Failed to fetch from the server: " + res.statusText);
    }
    
    const data = await res.json();
    // Extract the response text from the backend's result
    const responseText = data.result.responses[0][2];

    // Gradually print the response text
    let index = 0;
    const intervalId = setInterval(() => {
      responseDiv.innerText += responseText[index];
      index++;
      if (index === responseText.length) {
        clearInterval(intervalId);
      }
    }, 25);  // Adjust the typing speed here (in milliseconds)

  } catch (error) {
    console.error("Error during fetch:", error); // Log the error to the console
    responseDiv.innerText = "Error: " + (error as Error).message;
  } finally {
    loadingDiv.style.display = "none";
  }
});
