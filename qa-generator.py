import ollama
import json
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path, pages_per_chunk=5):
    """Extracts text from a PDF file in chunks of pages."""
    reader = PdfReader(pdf_path)
    chunks = []
    text = []
    
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
        
        # Process chunk when it reaches `pages_per_chunk` or at the end
        if (i + 1) % pages_per_chunk == 0 or (i + 1) == len(reader.pages):
            chunks.append("\n".join(text))
            text = []  # Reset text buffer
            print(f"Extracted chunk {len(chunks)} with {pages_per_chunk} pages.")

    return chunks

def structure_text_using_llama(text):
    """Uses LLaMA 3 to structure raw text into properly formatted sentences."""
    prompt = f"""The following text is extracted from a PDF and needs restructuring.

    Text:
    {text}

    Please restructure the text into clear, properly formatted sentences while preserving the meaning.
    If the last sentence seems incomplete, leave it at the end so it can be continued in the next part.
    
    Output only the structured text without extra explanation.
    """

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    print (response["message"]["content"])
    return response["message"]["content"]

def generate_qa_pairs(content):
    """Generates Q&A pairs using LLaMA 3 for structured content."""
    qa_pairs = []
    prompt = f"""Generate all possible questions and answers based on the following text:
    
    {content}
    
    Format:
    Q: <Generated Question>
    A: <Generated Answer>
    
    Output only the Q and A without extra text.
    """

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    response_text = response["message"]["content"]

    # Parse the response into Q&A pairs
    qa_sections = response_text.split("Q:")
    for section in qa_sections[1:]:  # Skip the first split part as it will be empty
        if "A:" in section:
            parts = section.split("A:")
            question = parts[0].strip()
            answer = parts[1].strip()

            qa_pairs.append({
                "instruction": question,
                "input": "",
                "output": answer
            })

    print (qa_pairs)
    return qa_pairs

def save_to_jsonl(data, filename="./QA Generator/train.jsonl"):
    """Saves the generated Q&A pairs in JSONL format."""
    with open(filename, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

# Load PDF and extract text in chunks
pdf_path = "./QA Generator/Liebherr-LTM-1130-operators-manual-21-1686.pdf"

chunks = extract_text_from_pdf(pdf_path)
qa_data = []
incomplete_sentence = ""

for i, chunk in enumerate(chunks):
    # If there's an incomplete sentence from the previous chunk, prepend it
    if incomplete_sentence:
        chunk = incomplete_sentence + " " + chunk

    # Get structured sentences from LLaMA 3
    structured_text = structure_text_using_llama(chunk)

    # Check if the last sentence is incomplete
    structured_sentences = structured_text.split("\n")
    last_sentence = structured_sentences[-1].strip()
    
    # If it looks incomplete, store it for the next chunk
    if len(last_sentence) < 20 or not last_sentence.endswith((".", "?", "!")):
        incomplete_sentence = last_sentence
        structured_sentences = structured_sentences[:-1]  # Remove the incomplete part
    else:
        incomplete_sentence = ""  # Reset if sentence is complete

    final_text = "\n".join(structured_sentences)

    # Generate Q&A pairs from the structured text
    qa_data.extend(generate_qa_pairs(final_text))
    print(f"Finished generating Q&A for chunk {i+1}/{len(chunks)}")

# Save the output
save_to_jsonl(qa_data)
print(f"QA pairs saved to train.jsonl with {len(qa_data)} entries.")