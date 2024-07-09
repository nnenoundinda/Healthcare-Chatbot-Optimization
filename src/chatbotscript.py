import os
import json
import gradio as gr
import csv
from pathlib import Path
import fitz  # PyMuPDF for PDF text extraction
from embedchain import App
from bert_score import score
from difflib import SequenceMatcher

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "API-KEY"

# Define the path to the folder containing PDFs
pdf_folder_path = Path('/content/drive/MyDrive/A&N_Research/PDF/')

# Function to extract and index PDF content
def index_pdfs(pdf_folder):
    pdf_index = {}
    for pdf_file in pdf_folder.glob('*.pdf'):
        with fitz.open(pdf_file) as doc:
            text = []
            for page in doc:
                text.append(page.get_text())
            pdf_index[pdf_file.stem] = ' '.join(text)
    return pdf_index

pdf_index = index_pdfs(pdf_folder_path)

# Function to get the most relevant context for a question from indexed PDFs
def get_relevant_context(question, pdf_index):
    best_context = ""
    best_score = 0
    for text in pdf_index.values():
        score = sum(question.lower().count(word) for word in text.lower().split())
        if score > best_score:
            best_score = score
            best_context = text
    return best_context

# Initialize embedchain apps for each model version
app_gpt_3_5 = App.from_config(
    config={
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.5,
                "max_tokens": 1000,
                "top_p": 1,
                "stream": False
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-ada-002"
            }
        }
    }
)

app_gpt_4 = App.from_config(
    config={
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4-turbo",
                "temperature": 0.5,
                "max_tokens": 1000,
                "top_p": 1,
                "stream": False
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-ada-002"
            }
        }
    }
)

# Function to load ground truth data from a JSON file
def load_ground_truth(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to calculate the BERT F1 score
def calculate_bert_score(prediction, reference):
    _, _, bert_scores = score([prediction], [reference], lang='en', model_type='bert-base-uncased')
    return bert_scores.mean().item()

# Function to calculate the Accuracy of responses



# Define the CSV file path
csv_file_path = '/content/drive/MyDrive/A&N_Research/final_answers.csv'

# Function to update CSV with results
def update_csv(question, gpt35_response, gpt35_score, gpt4_response, gpt4_score):
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if os.stat(csv_file_path).st_size == 0:  # Check if file is empty
            writer.writerow(['Question', 'GPT-3.5 Response', 'GPT-3.5 Score', 'GPT-4 Response', 'GPT-4 Score'])
        writer.writerow([question, gpt35_response, gpt35_score, gpt4_response, gpt4_score])

# Chatbot interaction function
def chatbot_interaction(question, json_file_path):
    relevant_context = get_relevant_context(question, pdf_index)
    response_gpt_3_5 = app_gpt_3_5.query(question + "\n\n" + relevant_context)
    response_gpt_4 = app_gpt_4.query(question + "\n\n" + relevant_context)
    ground_truth_data = load_ground_truth(json_file_path)
    most_similar_question = max(ground_truth_data, key=lambda x: SequenceMatcher(None, question, x['question']).ratio(), default=None)

    if most_similar_question:
        reference_answer = most_similar_question['answer']
        gpt_3_5_score = calculate_bert_score(response_gpt_3_5, reference_answer)
        gpt_4_score = calculate_bert_score(response_gpt_4, reference_answer)
        update_csv(question, response_gpt_3_5, gpt_3_5_score, response_gpt_4, gpt_4_score)

        return (f"Response from GPT-3.5: {response_gpt_3_5}\nScore: {gpt_3_5_score:.2f}\n\n"
                f"Response from GPT-4: {response_gpt_4}\nScore: {gpt_4_score:.2f}")
    else:
        return "No similar ground truth question found."

# Launch Gradio interface
demo = gr.Interface(
    fn=lambda q: chatbot_interaction(q, '/content/drive/MyDrive/A&N_Research/GroundTruth/final_ground_truth_answers_revised.json'),
    inputs="text",
    outputs="text",
    title="Kidney Cancer Research Chatbot",
    description="Ask a question about kidney cancer to receive answers from GPT-3.5 and GPT-4, along with BERT F1 scores."
)
demo.launch()
