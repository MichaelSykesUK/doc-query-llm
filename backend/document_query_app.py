import os
import re
import csv
import json
import pickle
import requests
import numpy as np
import pdfplumber
from typing import Optional, List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# PDF Processing
# =============================================================================


class PDFProcessor:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.unwanted_phrases = {
            "date:",
            "issue:",
            "page:",
            "©",
            "information in this document is the property",
            "ltd.",
            "express written consent",
        }

    def is_unwanted_line(self, line: str) -> bool:
        line_lower = line.lower()
        if any(phrase in line_lower for phrase in self.unwanted_phrases):
            return True
        # Also remove lines that contain numeric references with keywords.
        return any(
            keyword in line_lower and any(char.isdigit()
                                          for char in line_lower)
            for keyword in ["date", "issue", "page"]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[List[Dict]]:
        """
        Extracts text from the PDF page by page.
        Returns a list of dictionaries with keys:
          - doc_name: the name of the document (PDF file basename)
          - page: the page number (1-indexed)
          - raw_text: the unprocessed text from the page
          - processed_text: the cleaned-up text from the page
        """
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return None

        pages = []
        raw_text_list = []
        processed_text_list = []

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    raw_text_list.append(text)
                    processed_lines = []
                    for line in text.splitlines():
                        if not self.is_unwanted_line(line):
                            # Remove groups of two or more periods.
                            cleaned_line = re.sub(r"\.{2,}", "", line)
                            # Cleanup extra spaces.
                            cleaned_line = re.sub(
                                r"\s+", " ", cleaned_line).strip()
                            processed_lines.append(cleaned_line)
                    processed_text = "\n".join(processed_lines)
                    processed_text_list.append(processed_text)
                    pages.append({
                        "doc_name": os.path.basename(pdf_path),
                        "page": i,
                        "raw_text": text,
                        "processed_text": processed_text,
                    })

        pdf_filename = os.path.basename(pdf_path)
        print(pdf_filename)
        # Save raw text (joined with double newlines)
        with open(
            os.path.join(self.output_dir, f"text_raw_{pdf_filename}.txt"),
            "w",
            encoding="utf-8",
        ) as raw_file:
            raw_file.write("\n\n".join(raw_text_list))
        # Save processed text
        with open(
            os.path.join(self.output_dir, f"text_processed_{
                         pdf_filename}.txt"),
            "w",
            encoding="utf-8",
        ) as processed_file:
            processed_file.write("\n\n".join(processed_text_list))

        return pages

    def chunk_pages(self, pages: List[Dict], chunk_size: int = 100, mode: str = "page") -> List[Dict]:
        """
        Chunks each page's processed text according to the selected mode.

        Currently supports:
          - "page": each page is treated as a single chunk.
          - "character": splits the text into chunks of fixed character length.

        Returns a list of dictionaries with keys:
          - doc_name: document name (basename)
          - page: page number from which the chunk came
          - chunk_text: the text for the chunk
        """
        chunks = []
        for page in pages:
            doc_name = page["doc_name"]
            page_num = page["page"]
            text = page["processed_text"]

            if mode == "page":
                # Each page becomes one chunk.
                chunks.append({
                    "doc_name": doc_name,
                    "page": page_num,
                    "chunk_text": text,
                })
            elif mode == "character":
                # Chunk by fixed number of characters.
                for i in range(0, len(text), chunk_size):
                    chunk_text = text[i: i + chunk_size]
                    chunks.append({
                        "doc_name": doc_name,
                        "page": page_num,
                        "chunk_text": chunk_text,
                    })
            else:
                # Fallback: treat each page as a single chunk.
                chunks.append({
                    "doc_name": doc_name,
                    "page": page_num,
                    "chunk_text": text,
                })
        print(f"Total chunks created: {len(chunks)}")
        return chunks

# =============================================================================
# Embedding Management
# =============================================================================


class EmbeddingManager:
    def __init__(self, model: SentenceTransformer, output_dir: str = "output"):
        self.model = model
        self.output_dir = output_dir
        self.embeddings_file = os.path.join(
            self.output_dir, "document_embeddings.pkl")
        self.chunk_mapping_file = os.path.join(
            self.output_dir, "chunk_mappings.pkl")
        self.chunk_texts_file = os.path.join(
            self.output_dir, "chunk_texts.pkl")

    def load_embeddings(self) -> Tuple[List[np.ndarray], List[Tuple[str, int]], List[str]]:
        if (
            os.path.exists(self.embeddings_file)
            and os.path.exists(self.chunk_mapping_file)
            and os.path.exists(self.chunk_texts_file)
        ):
            with open(self.embeddings_file, "rb") as emb_file, \
                    open(self.chunk_mapping_file, "rb") as map_file, \
                    open(self.chunk_texts_file, "rb") as text_file:
                embeddings = pickle.load(emb_file)
                chunk_to_doc_map = pickle.load(map_file)
                chunk_texts = pickle.load(text_file)
            return embeddings, chunk_to_doc_map, chunk_texts
        return [], [], []

    def save_embeddings(
        self,
        embeddings: List[np.ndarray],
        chunk_to_doc_map: List[Tuple[str, int]],
        chunk_texts: List[str],
    ) -> None:
        with open(self.embeddings_file, "wb") as emb_file, \
                open(self.chunk_mapping_file, "wb") as map_file, \
                open(self.chunk_texts_file, "wb") as text_file:
            pickle.dump(embeddings, emb_file)
            pickle.dump(chunk_to_doc_map, map_file)
            pickle.dump(chunk_texts, text_file)
        print("Embeddings saved successfully.")

    def embed_documents(
        self,
        pdf_files: List[str],
        folder_path: str,
        pdf_processor: PDFProcessor,
        chunk_mode: str = "page",  # using "page" mode here
        chunk_size: int = 100
    ) -> Tuple[List[np.ndarray], List[Tuple[str, int]], List[str]]:
        all_chunks_text = []
        all_chunks_embeddings = []
        chunk_to_doc_map = []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            print(f"Processing document: {pdf_file}")

            pages = pdf_processor.extract_text_from_pdf(pdf_path)
            if not pages:
                print(f"No text extracted from the PDF: {pdf_path}")
                continue

            # Create chunks from the pages using the desired mode.
            chunks = pdf_processor.chunk_pages(
                pages, chunk_size=chunk_size, mode=chunk_mode)
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]
            all_chunks_text.extend(chunk_texts)
            chunk_to_doc_map.extend(
                [(chunk["doc_name"], chunk["page"]) for chunk in chunks])
            chunk_embeddings = self.model.encode(chunk_texts)
            all_chunks_embeddings.extend(chunk_embeddings)

        self.save_embeddings(all_chunks_embeddings,
                             chunk_to_doc_map, all_chunks_text)
        return all_chunks_embeddings, chunk_to_doc_map, all_chunks_text

    def gather_embeddings(
        self, folder_path: str, pdf_processor: PDFProcessor
    ) -> Tuple[List[np.ndarray], List[Tuple[str, int]], List[str]]:
        document_embeddings, chunk_to_doc_map, chunk_texts = self.load_embeddings()
        if not document_embeddings:
            if not os.path.isdir(folder_path):
                print(f"The specified folder does not exist: {folder_path}")
                return None, None, None
            pdf_files = [f for f in os.listdir(
                folder_path) if f.lower().endswith(".pdf")]
            if not pdf_files:
                print("No PDF files found in the folder.")
                return None, None, None

            print("Embedding documents with Sentence-BERT model...")
            document_embeddings, chunk_to_doc_map, chunk_texts = self.embed_documents(
                pdf_files, folder_path, pdf_processor, chunk_mode="page", chunk_size=100
            )

        return document_embeddings, chunk_to_doc_map, chunk_texts

# =============================================================================
# Ollama API Client
# =============================================================================


class OllamaClient:
    def __init__(self, api_url: str = "http://localhost:11434/api/generate"):
        self.api_url = api_url

    def query(self, model_name: str, prompt: str) -> Optional[str]:
        payload = {"model": model_name, "prompt": prompt}
        response = requests.post(self.api_url, json=payload)

        if response.status_code == 200:
            raw_data = response.text.splitlines()
            output = ""
            for line in raw_data:
                try:
                    json_response = json.loads(line)
                    output += json_response.get("response", "")
                    if json_response.get("done", False):
                        break
                except ValueError as e:
                    print("Error parsing JSON:", e)
                    return None
            return output.strip()
        else:
            print(f"Ollama API error: {response.status_code}")
            return None

# =============================================================================
# Response Exporter
# =============================================================================


class ResponseExporter:
    @staticmethod
    def save_similarities_to_csv(
        results: List[List], filename: str = "output/similarities.csv"
    ) -> None:
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Document",
                    "Page",
                    "Chunk Number",
                    "Cosine Similarity",
                    "Cosine Rank",
                    "Picked for LLM",
                ]
            )
            writer.writerows(results)
        print(f"Similarities saved to {filename}")

    @staticmethod
    def save_text_to_file(text: str, filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Text successfully saved to '{filename}'.")

    @staticmethod
    def export_responses(
        all_responses: List[Tuple[str, str, str, float]],
        filename: str = "output/model_responses.txt",
    ) -> None:
        with open(filename, "w", encoding="utf-8") as f:
            for (query_text, model_used, response_text, response_similarity) in all_responses:
                f.write("Query: " + query_text + "\n")
                f.write("Model: " + model_used + "\n")
                f.write(f"Cosine similarity between query and response: {
                        response_similarity:.4f}\n")
                f.write("Response:\n" + response_text + "\n")
                f.write("-" * 40 + "\n")
        print(f"\nAll responses exported to '{filename}'")

# =============================================================================
# Query Processing
# =============================================================================


class QueryProcessor:
    def __init__(
        self,
        model: SentenceTransformer,
        embedding_manager: EmbeddingManager,
        ollama_client: OllamaClient,
        available_models: List[str],
    ):
        self.model = model
        self.embedding_manager = embedding_manager
        self.ollama_client = ollama_client
        self.available_models = available_models

    def compute_similarities(
        self,
        query_embedding: np.ndarray,
        document_embeddings: List[np.ndarray],
        chunk_to_doc_map: List[Tuple[str, int]],
        chunk_texts: List[str],
        top_n: int = 10,
        similarity_threshold: float = 0.5,
    ) -> str:
        print("Computing cosine similarities between query and document embeddings...")
        similarities = cosine_similarity(
            query_embedding, document_embeddings).flatten()

        selected_indices = [idx for idx, sim in enumerate(
            similarities) if sim > similarity_threshold]
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

        # Build relevant texts string including document name and page number.
        relevant_texts = "\n\n".join(
            f"From document '{chunk_to_doc_map[idx][0]}' (Page {chunk_to_doc_map[idx][1]}):\n{
                chunk_texts[idx]}"
            for idx in selected_indices
        )

        # Save the most relevant texts to a file.
        ResponseExporter.save_text_to_file(
            relevant_texts, "output/relevant_texts.txt")

        # Create similarity results and export to CSV.
        rank_map = {
            idx: rank + 1
            for rank, idx in enumerate(sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True))
        }
        selection_map = {idx: "Yes" for idx in selected_indices}
        similarity_results = []
        for idx, sim in enumerate(similarities):
            doc_name, page_num = chunk_to_doc_map[idx]
            cosine_rank = rank_map.get(idx)
            picked = selection_map.get(idx, "No")
            similarity_results.append(
                [doc_name, page_num, idx, sim, cosine_rank, picked])

        ResponseExporter.save_similarities_to_csv(similarity_results)

        print("\n--- Query Summary ---")
        print(f"Condition for selection: Chunks with cosine similarity > {
              similarity_threshold}, or top {top_n} chunks if fewer than {top_n} exceed the threshold")
        for item in similarity_results:
            doc_name, page_num, chunk_num, sim, rank, picked = item
            if picked == "Yes":
                print(f"Chunk {chunk_num} from '{doc_name}' (Page {page_num}) with Rank {
                      rank} (Cosine Similarity: {sim:.4f})")
        print("\nMost relevant documents:")
        relevant_documents = {
            chunk_to_doc_map[idx][0] for idx in selected_indices}
        for doc in relevant_documents:
            print(doc)

        return relevant_texts

    def query_models(
        self,
        prompt: str,
        query: str,
        query_embedding: np.ndarray,
        selected_models: List[str],
    ) -> List[Tuple[str, str, str, float]]:
        all_responses = []
        for model_name in selected_models:
            print(f"\nQuerying model: {model_name}...")
            response = self.ollama_client.query(model_name, prompt)
            if response:
                response_embedding = self.model.encode([response])
                response_similarity = float(cosine_similarity(
                    query_embedding, response_embedding).flatten()[0])
                print(f"Cosine similarity between query and response: {
                      response_similarity:.4f}")
                print("Response:\n", response)
            else:
                print("No response generated for the query.")
                response_similarity = 0.0
            all_responses.append(
                (query, model_name, response, response_similarity))
        return all_responses

    def run_super_query(
        self,
        query: str,
        query_embedding: np.ndarray,
        all_responses: List[Tuple[str, str, str, float]],
    ) -> List[Tuple[str, str, str, float]]:
        # Sort responses by similarity in descending order.
        all_responses = sorted(
            all_responses, key=lambda x: x[-1], reverse=True)
        best_model = all_responses[0][1]
        print(f"\nBest performing model: {best_model}")

        # Concatenate all model responses and re-query the best model.
        response_texts = "\n".join([x[2] for x in all_responses])
        prompt = f"Relevant text:\n{
            response_texts}\n\nQuestion: {query}\n\nAnswer:"
        ResponseExporter.save_text_to_file(
            prompt, os.path.join("output", "super_prompt.txt"))
        response = self.ollama_client.query(best_model, prompt)

        print(f"\nBest model response:\n{response}")

        best_model_similarity = float(cosine_similarity(
            query_embedding, self.model.encode([response])).flatten()[0])
        print(f"Cosine similarity between query and best model response: {
              best_model_similarity:.4f}")
        all_responses.append(
            (query, best_model + "_super", response, best_model_similarity))
        return all_responses

# =============================================================================
# Main Application (API-Friendly)
# =============================================================================


class DocumentQueryApp:
    def __init__(self, docs_folder: str = "docs", output_dir: str = "output"):
        self.docs_folder = docs_folder
        self.output_dir = output_dir
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.available_models = [
            "llama3.1:8b",
            "llama3:8b",
            "llama2:7b",
            "gemma2:9b",
            "phi4:14b",
            "mistral:7b",
            "mistral-nemo:12b",
        ]
        self.pdf_processor = PDFProcessor(output_dir=self.output_dir)
        self.embedding_manager = EmbeddingManager(
            self.model, output_dir=self.output_dir)
        self.ollama_client = OllamaClient()
        self.query_processor = QueryProcessor(
            self.model,
            self.embedding_manager,
            self.ollama_client,
            self.available_models,
        )

    def process_query(
        self, query: str, additional_context: str, model: Optional[List[str]] = None
    ) -> dict:
        # If a model is provided as a string (from the front end), convert it to a list.
        if model and isinstance(model, str):
            selected_models = [m.strip()
                               for m in model.split(",") if m.strip()]
        else:
            selected_models = model if model else [
                self.query_processor.available_models[0]]

        extra_context = (
            ". Structure the response as follows, a concise summary, "
            "followed by bulleted key takeaways, followed by key document references. "
            "Where possible list the relevant document sections and page numbers."
        )

        # Build the full query text from inputs.
        full_query = (
            f"In the context of {additional_context}, {query} {extra_context}"
            if additional_context
            else f"{query} {extra_context}"
        )

        # Get the query embedding.
        query_embedding = self.model.encode([full_query])

        # Gather or update embeddings.
        document_embeddings, chunk_to_doc_map, chunk_texts = self.embedding_manager.gather_embeddings(
            self.docs_folder, self.pdf_processor
        )
        if document_embeddings is None:
            return {"error": "No document embeddings available."}

        # Compute similarities and build the prompt.
        relevant_texts = self.query_processor.compute_similarities(
            query_embedding, document_embeddings, chunk_to_doc_map, chunk_texts
        )
        prompt = f"Relevant text:\n{
            relevant_texts}\n\nQuestion: {full_query}\n\nAnswer:"

        # Query selected models.
        all_responses = self.query_processor.query_models(
            prompt, full_query, query_embedding, selected_models)

        # Optionally, run the super query to combine responses.
        # all_responses = self.query_processor.run_super_query(query, query_embedding, all_responses)

        # Export and return the responses.
        ResponseExporter.export_responses(all_responses)

        return {"query": query, "responses": all_responses}
