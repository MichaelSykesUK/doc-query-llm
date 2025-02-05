import numpy as np
import os
import csv
import pdfplumber
import requests
import json
import re
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List, Tuple

# Ollama API URL (local)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Initialize the Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define available Ollama models
available_models = [
    "llama3.1:8b",  # Meta
    "llama3:8b",  # Meta
    "llama2:7b",  # Meta
    "gemma2:9b",  # Google - poor
    "phi4:14b",  # Microsoft
    "mistral:7b",  # Mistral AI
    "mistral-nemo:12b",  # Mistral AI
]


# Persistent storage for document embeddings
EMBEDDINGS_FILE = "output/document_embeddings.pkl"
CHUNK_MAPPING_FILE = "output/chunk_mappings.pkl"
CHUNK_TEXTS_FILE = "output/chunk_texts.pkl"

if not os.path.exists("output"):
    os.mkdir("output")


def load_embeddings() -> Tuple[List[np.ndarray], List[Tuple[str, int]], List[str]]:
    """
    Load stored document embeddings, chunk-to-doc map, and chunk texts.
    """
    if (
        os.path.exists(EMBEDDINGS_FILE)
        and os.path.exists(CHUNK_MAPPING_FILE)
        and os.path.exists(CHUNK_TEXTS_FILE)
    ):
        with open(EMBEDDINGS_FILE, "rb") as emb_file, open(
            CHUNK_MAPPING_FILE, "rb"
        ) as map_file, open(CHUNK_TEXTS_FILE, "rb") as text_file:
            embeddings = pickle.load(emb_file)
            chunk_to_doc_map = pickle.load(map_file)
            chunk_texts = pickle.load(text_file)
            return embeddings, chunk_to_doc_map, chunk_texts
    return [], [], []


def save_embeddings(
    embeddings: List[np.ndarray],
    chunk_to_doc_map: List[Tuple[str, int]],
    chunk_texts: List[str],
) -> None:
    """
    Save document embeddings, chunk-to-doc map, and chunk texts.
    """
    with open(EMBEDDINGS_FILE, "wb") as emb_file, open(
        CHUNK_MAPPING_FILE, "wb"
    ) as map_file, open(CHUNK_TEXTS_FILE, "wb") as text_file:
        pickle.dump(embeddings, emb_file)
        pickle.dump(chunk_to_doc_map, map_file)
        pickle.dump(chunk_texts, text_file)
    print("Embeddings saved successfully.")


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract text from a PDF file, saving raw text to 'text_raw_{filename}.txt'
    and processed text to 'text_processed_{filename}.txt'.

    The processed text preserves the original carriage returns so that the line
    structure remains similar to the raw text.
    """
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return None

    # Define phrases to filter out lines we don't want.
    unwanted_phrases = {
        "date:",
        "issue:",
        "page:",
        "©",
        "information in this document is the property",
        "ltd.",
        "express written consent",
    }

    def is_unwanted_line(line: str) -> bool:
        """Return True if the line contains unwanted keywords or patterns."""
        line_lower = line.lower()
        if any(phrase in line_lower for phrase in unwanted_phrases):
            return True
        # Also remove lines that contain numeric references with keywords.
        return any(
            keyword in line_lower and any(char.isdigit()
                                          for char in line_lower)
            for keyword in ["date", "issue", "page"]
        )

    raw_text_list = []
    processed_text_list = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_text_list.append(text)

                processed_lines = []
                for line in text.splitlines():
                    if not is_unwanted_line(line):
                        # Remove groups of two or more periods.
                        cleaned_line = re.sub(r"\.{2,}", "", line)
                        # Cleanup extra spaces.
                        cleaned_line = re.sub(
                            r"\s+", " ", cleaned_line).strip()
                        processed_lines.append(cleaned_line)
                # Preserve the carriage returns.
                processed_text = "\n".join(processed_lines)
                processed_text_list.append(processed_text)

    # Extract only the file name from the full path.
    pdf_filename = os.path.basename(pdf_path)

    # Write raw text preserving original carriage returns.
    with open(f"output/text_raw_{pdf_filename}.txt", "w", encoding="utf-8") as raw_file:
        raw_file.write("\n".join(raw_text_list))

    # Write processed text preserving line breaks.
    with open(
        f"output/text_processed_{pdf_filename}.txt", "w", encoding="utf-8"
    ) as processed_file:
        processed_file.write("\n".join(processed_text_list))

    return "\n".join(processed_text_list)


def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    sentences = re.split(r"(?<!\w\.\w)(?<=\. )", text)
    chunks = []
    chunk = []

    for index, sentence in enumerate(sentences, start=1):
        chunk.append(sentence)
        if len(" ".join(chunk).split()) >= chunk_size:
            if not chunk[-1].endswith("."):
                chunk.pop()
            chunks.append(" ".join(chunk))
            chunk = []

    if chunk:
        chunks.append(" ".join(chunk))

    print(f"Total chunks: {len(chunks)}")
    return chunks


def embed_documents(
    pdf_files: List[str], folder_path: str
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Embed the text chunks of the PDFs in the folder, and return the embeddings and other relevant data.
    """
    all_chunks_text = []
    all_chunks_embeddings = []
    chunk_to_doc_map = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Processing document: {pdf_file}")

        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"No text extracted from the PDF: {pdf_path}")
            continue

        chunks = chunk_text(text)
        all_chunks_text.extend(chunks)

        chunk_embeddings = model.encode(chunks)
        all_chunks_embeddings.extend(chunk_embeddings)

        chunk_to_doc_map.extend([(pdf_path, i)
                                for i in range(len(chunk_embeddings))])

    save_embeddings(all_chunks_embeddings, chunk_to_doc_map, all_chunks_text)

    return all_chunks_embeddings, chunk_to_doc_map, all_chunks_text


def select_top_chunks(
    similarities: np.ndarray,
    chunk_to_doc_map: List[Tuple[str, int]],
    chunk_texts: List[str],
    top_n: int,
    similarity_threshold: float,
) -> Tuple[str, set, List[List]]:
    """
    Select the top N chunks based on cosine similarity, preserving document order.
    """
    selected_indices = [
        idx for idx, sim in enumerate(similarities) if sim > similarity_threshold
    ]

    if len(selected_indices) < top_n:
        remaining_indices = sorted(
            (idx for idx in range(len(similarities))
             if idx not in selected_indices),
            key=lambda idx: similarities[idx],
            reverse=True,
        )
        selected_indices.extend(
            remaining_indices[: top_n - len(selected_indices)])

    selected_indices.sort()

    relevant_texts = "\n\n".join(
        f"From document, {os.path.basename(chunk_to_doc_map[idx][0])}:\n{
            chunk_texts[idx]}"
        for idx in selected_indices
    )
    relevant_documents = {chunk_to_doc_map[idx][0] for idx in selected_indices}

    if not relevant_documents:
        relevant_documents = {"No documents met the similarity threshold."}

    rank_map = {
        idx: rank + 1
        for rank, idx in enumerate(
            sorted(
                range(len(similarities)), key=lambda i: similarities[i], reverse=True
            )
        )
    }
    selection_map = {idx: "Yes" for idx in selected_indices}

    similarity_results = []
    for idx, similarity in enumerate(similarities):
        doc_path, chunk_num = chunk_to_doc_map[idx]
        cosine_rank = rank_map.get(idx)
        picked = selection_map.get(idx, "No")
        similarity_results.append(
            [doc_path, chunk_num, similarity, cosine_rank, picked]
        )

    # Alternate approach using list comprehension
    # similarity_results = [
    #     [chunk_to_doc_map[idx][0], chunk_num, similarities[idx], rank + 1, "Yes"]
    #     for rank, idx in enumerate(sorted(selected_indices, key=lambda i: similarities[i], reverse=True))
    # ]

    return relevant_texts, relevant_documents, similarity_results


def save_similarities_to_csv(
    results: List[List], filename: str = "output/similarities.csv"
) -> None:
    """
    Save the similarities results to a CSV file.
    """
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Document",
                "Chunk Number",
                "Cosine Similarity",
                "Cosine Rank",
                "Picked for LLM",
            ]
        )
        writer.writerows(results)
    print(f"Similarities saved to {filename}")


def save_text_to_file(text: str, filename: str = "relevant_text.txt") -> None:
    """
    Save the provided text to a text file.
    """
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Text successfully saved to '{filename}'.")


def compute_similarities(
    query_embedding: np.ndarray,
    document_embeddings: List[np.ndarray],
    chunk_to_doc_map,
    chunk_texts,
    top_n: int = 5,
    similarity_threshold: float = 0.4,
) -> Optional[str]:
    """
    Compute cosine similarities between the query and document embeddings.
    """
    print("Computing cosine similarities between query and document embeddings...")
    similarities = cosine_similarity(
        query_embedding, document_embeddings).flatten()

    print("Selecting top chunks based on similarities...")
    relevant_texts, relevant_documents, similarity_results = select_top_chunks(
        similarities, chunk_to_doc_map, chunk_texts, top_n, similarity_threshold
    )

    save_similarities_to_csv(similarity_results)

    condition_statement = (
        "Chunks with cosine similarity > "
        f"{similarity_threshold}, or top {
            top_n} chunks if fewer than {top_n} exceed the threshold"
    )
    print("\n--- Query Summary ---")
    print(f"Condition for selection: {condition_statement}")
    for idx in similarity_results:
        doc_path, chunk_num, similarity, rank, picked = idx
        if picked == "Yes":
            print(
                f"Chunk {chunk_num} from '{doc_path}' with Rank {
                    rank} (Cosine Similarity: {similarity:.4f})"
            )

    print("\nMost relevant documents:")
    for doc in relevant_documents:
        print(f"{doc}")

    return relevant_texts


def select_models():
    """
    Prompt the user to select one or more models from the available list.
    """
    print("\nAvailable models:")
    for i, m in enumerate(available_models, 1):
        print(f"{i}. {m}")
    model_choices = input(
        "Select one or more models by number (comma separated, default is 1): "
    )
    if model_choices.strip() == "":
        selected_indices = [0]
    else:
        try:
            selected_indices = [
                int(x.strip()) - 1
                for x in model_choices.split(",")
                if x.strip().isdigit()
            ]
        except Exception as e:
            print(f"Error parsing input, defaulting to model 1. Error: {e}")
            selected_indices = [0]
    selected_models = [
        available_models[i] for i in selected_indices if 0 <= i < len(available_models)
    ]
    print("Selected models:", selected_models)
    return selected_models


def gather_emeddings(folder_path):
    document_embeddings, chunk_to_doc_map, chunk_texts = load_embeddings()
    if not document_embeddings:
        if not os.path.isdir(folder_path):
            print(f"The specified folder does not exist: {folder_path}")
            return None
        pdf_files = [f for f in os.listdir(
            folder_path) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print("No PDF files found in the folder.")
            return None

        print("Embedding documents with Sentence-BERT model...")
        document_embeddings, chunk_to_doc_map, chunk_texts = embed_documents(
            pdf_files, folder_path
        )

    return document_embeddings, chunk_to_doc_map, chunk_texts


def query_ollama(model_name: str, prompt: str) -> Optional[str]:
    """
    Query the Ollama API with a given text chunk and query.
    """
    payload = {"model": model_name, "prompt": prompt}
    response = requests.post(OLLAMA_API_URL, json=payload)

    if response.status_code == 200:
        raw_data = response.text.splitlines()
        output = ""
        for line in raw_data:
            try:
                json_response = json.loads(line)
                if json_response.get("done") is False:
                    output += json_response.get("response", "")
                elif json_response.get("done") is True:
                    output += json_response.get("response", "")
                    break
            except ValueError as e:
                print("Error parsing JSON:", e)
                return None
        return output.strip()
    else:
        print(f"Ollama API error: {response.status_code}")
        return None


def run_query_loop(document_embeddings, chunk_to_doc_map, chunk_texts):

    all_responses = []

    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        additional_context = input(
            "Enter additional context (or press Enter to skip): "
        )

        if additional_context:
            query = f"In the context of {additional_context}, {
                user_query}. Frame the response with a bullet point summary, quote relevant sentences, and provided references to key document sections."
        else:
            query = f"{
                user_query}. Frame the response with a bullet point summary, quote relevant sentences, and provided references to key document sections."

        # Embed query using Sentence-BERT model.
        print("Embedding query with Sentence-BERT model...")
        query_embedding = model.encode([query])

        # Compute cosine similarity between query and document embeddings.
        relevant_texts = compute_similarities(
            query_embedding, document_embeddings, chunk_to_doc_map, chunk_texts
        )

        prompt = (
            f"Relevant text:\n{relevant_texts}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        save_text_to_file(prompt, "output/prompt")

        # Query each selected model.
        for model_name in selected_models:
            print(f"\nQuerying model: {model_name}...")
            response = query_ollama(model_name, prompt)

            if response:

                response_embedding = model.encode([response])
                response_similarity = cosine_similarity(
                    query_embedding, response_embedding).flatten()[0]
                print(f"Cosine similarity between query and response: {
                      response_similarity:.4f}")
                print("Response:\n")
                print(response)

            else:
                print("No response generated for the query.")

            all_responses.append(
                (query, model_name, response, response_similarity))

        # Ask if the user wants an additional response from another model.
        another = input(
            "\nWould you like to try another model response? (yes/no): ")
        while another.lower() in ["yes", "y"]:
            print("\nInitial selected models:")
            for idx, model_name in enumerate(selected_models, start=1):
                print(f"{idx}. {model_name}")
            chosen_model_input = input(
                "Enter the model number from the initial selection: "
            )
            try:
                chosen_index = int(chosen_model_input) - 1
                if 0 <= chosen_index < len(selected_models):
                    chosen_model = selected_models[chosen_index]
                    print(
                        f"\n--- Querying additional response from model: {
                            chosen_model} ---"
                    )
                    response = query_ollama(
                        relevant_texts,
                        query,
                        chosen_model,
                        context=additional_context,
                    )
                    all_responses.append(
                        (query, chosen_model,
                         response, response_similarity)
                    )
                else:
                    print("Invalid model number. Please try again.")
            except Exception as e:
                print("Error processing model selection:", e)
            another = input(
                "Would you like to try another model response? (yes/no): ")

    # Sort responses by their last element (assumed similarity score) in descending order
    all_responses = sorted(all_responses, key=lambda x: x[-1], reverse=True)

    # Select the best performing model based on the highest similarity
    best_model = all_responses[0][1]
    print(f"\nBest performing model: {best_model}")

    # Concatenate all model responses and re-query the best model with them
    print("Re-querying the best model with all responses...")
    response_texts = "\n".join([x[2] for x in all_responses])
    prompt = (
        f"Relevant text:\n{response_texts}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    save_text_to_file(prompt, "output/super_prompt")

    response = query_ollama(best_model, prompt)

    print(f"\nBest model response:\n{response}")

    # Compute cosine similarity between the query and the best model response
    best_model_similarity = cosine_similarity(
        query_embedding, model.encode([response])
    ).flatten()[0]

    print(f"Cosine similarity between query and best model response: {
          best_model_similarity:.4f}")

    # Append the best model result to the list of responses
    all_responses.append(
        (query, best_model+"_super", response, best_model_similarity))

    # Export all responses to a text file.
    export_filename = "output/model_responses.txt"
    with open(export_filename, "w", encoding="utf-8") as f:
        for query_text, model_used, response_text, response_similarity in all_responses:
            f.write("Query: " + query_text + "\n")
            f.write("Model: " + model_used + "\n")
            f.write(f"Cosine of similarity between prompt and response: {
                    response_similarity:.4f}\n")
            f.write(f"Response:{response_similarity}\n" + response_text + "\n")
            f.write("-" * 40 + "\n")
    print(f"\nAll responses exported to '{export_filename}'.")


if __name__ == "__main__":
    # Ask the user to choose one or more models.
    selected_models = select_models()

    # Provide the path to the folder containing PDF files.
    folder_path = "docs"

    # load embeddings if available.
    document_embeddings, chunk_to_doc_map, chunk_texts = gather_emeddings(
        folder_path)

    # Run LLM query loop.
    run_query_loop(document_embeddings, chunk_to_doc_map, chunk_texts)
