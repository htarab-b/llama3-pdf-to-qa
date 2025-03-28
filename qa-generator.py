import ollama
import json
from PyPDF2 import PdfReader
import os

def extract_text_from_pdf(pdf_path, pages_per_chunk=2):
    """Extracts text from a PDF file in chunks of pages."""
    reader = PdfReader(pdf_path)
    chunks = []
    text = []
    
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
        
        if (i + 1) % pages_per_chunk == 0 or (i + 1) == len(reader.pages):
            chunks.append("\n".join(text))
            text = []  # Reset text buffer
            print(f"Extracted chunk {len(chunks)} with {pages_per_chunk} pages.")

    os.system("clear")
    print(chunks)
    return chunks

def structure_text_using_llama(text):
    """Uses LLaMA 3 to restructure raw text into properly formatted sentences."""
    prompt = f"""The following text is extracted from a PDF and needs restructuring.

    Text:
    {text}

    Please restructure the text into clear, properly formatted sentences while preserving the meaning.
    If the last sentence seems incomplete, leave it at the end so it can be continued in the next part.
    
    Output only the structured text without extra explanation.
    """

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    os.system("clear")
    print(response["message"]["content"])
    return response["message"]["content"]

def summarize_text(text):
    """Summarizes extracted content before generating Q&A."""
    prompt = f"""Summarize the following text while keeping key technical details:
    
    {text}
    
    Output only the summarized content."""
    
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    os.system("clear")
    print(response["message"]["content"])
    return response["message"]["content"]

def generate_qa_pairs(content):
    """Generates all possible Q&A pairs using LLaMA 3 for structured content."""
    prompt = f"""Generate all possible question-answer pairs based on the following text. Try to generate the maximum number of questions and answers by covering the same content through different angles.:
    
    {content}
    
    Format:
    Q: <Generated Question>
    A: <Generated Answer>
    
    Only output the Q and A without extra text.
    """

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    response_text = response["message"]["content"]

    # Parse the response into Q&A pairs
    qa_pairs = []
    qa_sections = response_text.split("Q:")
    for section in qa_sections[1:]:  # Skip the first empty split
        if "A:" in section:
            parts = section.split("A:")
            question = parts[0].strip()
            answer = parts[1].strip()

            qa_pairs.append({
                "prompt": question,
                "response": answer
            })
    os.system("clear")
    print(qa_pairs)
    return qa_pairs

def save_to_jsonl(data, filename):
    """Saves the generated Q&A pairs in JSONL format."""
    with open(filename, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

# Load PDF and extract text
pdf_path = "./QA Generator/Liebherr-LTM-1130-operators-manual-21-1686.pdf"
chunks = extract_text_from_pdf(pdf_path)

# Process text and generate Q&A pairs
structured_chunks = [structure_text_using_llama(chunk) for chunk in chunks]
summarized_chunks = [summarize_text(chunk) for chunk in structured_chunks]
qa_data = []

for chunk in summarized_chunks:
    qa_data.extend(generate_qa_pairs(chunk))

# Save output
save_to_jsonl(qa_data, "./QA Generator/train.jsonl")
print(f"QA pairs saved to train.jsonl with {len(qa_data)} entries.")
