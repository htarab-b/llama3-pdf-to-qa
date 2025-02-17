import ollama
import json
import spacy
from PyPDF2 import PdfReader

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = []
    
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
        print(f"Finished extracting text from page {i+1}")

    return "\n".join(text)

def chunk_large_text(text, chunk_size=500_000):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

def split_text_spacy(text):
    chunks = chunk_large_text(text)
    sentences = []
    
    for i, chunk in enumerate(chunks):
        doc = nlp(chunk)
        sentences.extend([sent.text.strip() for sent in doc.sents])
        print(f"Finished processing chunk {i+1}/{len(chunks)}")

    return sentences

def generate_qa_pairs(content):
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

    # Split the response into multiple Q&A pairs
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

    return qa_pairs

def save_to_jsonl(data, filename="./QA Generator/train.jsonl"):
    with open(filename, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

pdf_path = "./QA Generator/Liebherr-LTM-1130-operators-manual-21-1686.pdf"

text = extract_text_from_pdf(pdf_path)
sentences = split_text_spacy(text)
chunks = [" ".join(sentences[i:i+10]) for i in range(0, len(sentences), 10)] 

qa_data = []
for i, chunk in enumerate(chunks):
    qa_data.extend(generate_qa_pairs(chunk))
    print(f"Finished generating Q&A for chunk {i+1}/{len(chunks)}")

save_to_jsonl(qa_data)
print(f"QA pairs saved to train.jsonl with {len(qa_data)} entries.")