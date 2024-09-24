import os
import getpass

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from time import sleep
import json

from openai import OpenAI

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your Openai API key: ")

client = OpenAI()

# Directory containing PDFs
pdf_dir = 'PDFs'
fine_tune_data_dir = 'fine_tune_data'
fine_tune_models_file = 'models.jsonl'
fine_tune_jobs_file = 'fine_tune_jobs.jsonl'
base_model = "gpt-4o-mini-2024-07-18"

text_splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small"))

def prepare_finetune_data(documents:list[Document])->list:
    # Split the text into chunks
    print("Splitting Document")
    chunks = text_splitter.split_documents(documents)
    print("Document Splitting complete")
    
    # Create the structured data for fine-tuning
    data = []
    for i in range(len(chunks) - 1):
        entry = {
            "messages": [
                {"role": "system", "content": "write the next part of the book"},
                {"role": "user", "content": chunks[i].page_content},
                {"role": "assistant", "content": chunks[i+1].page_content}
            ]
        }
        data.append(entry)
    
    return data

def generate_next_part(model:str, last_part:str):
    try:
        conversation_history = [
            {"role": "system", "content": "write the next part of the book"},
                  {"role": "user", "content": last_part}
        ]
        while input("Continue? (y/n) ").strip().lower() == 'y':
            completion = client.chat.completions.create(
                model=model,
                messages=conversation_history
            )
            last_part = completion.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": last_part})
            print("\n")
            print(last_part)
            print("\n")
        exit()
    except Exception as e:
        print(e)

pending_models = []
models_available = []
with open(fine_tune_jobs_file, 'r') as file:
    for line in file:
        if not line.strip():
            continue
        entry = json.loads(line)
        model = client.fine_tuning.jobs.retrieve(entry.get("id")).fine_tuned_model
        if model != None:
            models_available.append({"model":model,"last_part":entry.get("last_part")})
        else:
            pending_models.append(model)

if models_available:
    print("Available Models:\n")
    for idx, model in enumerate(models_available):
        print(f"{idx + 1}: {model['model']}")

    try:
        selected_index = int(input("Enter the index number of the Model you want to select(n to train a new pdf): "))
        if 1 <= selected_index < len(models_available) + 1:
            selected_model = models_available[selected_index - 1]
            print("Generating Next Part")
            generate_next_part(selected_model["model"], selected_model["last_part"])
        else:
            print("Invalid index number.")
            exit()
    except Exception as e:
        print(e)
        pass

if pending_models:
    print("Some models are still training")
    if input("y to add new pdf, n to exit: ") == "n":
        exit()

# Get list of PDFs in the directory
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

# Check if there are any PDFs in the directory
if not pdf_files:
    print("No PDF files found in the directory.")
else:
    # List all the PDFs with index numbers
    print("Available PDFs:\n")
    for idx, pdf in enumerate(pdf_files):
        print(f"{idx + 1}: {pdf}")

    # Ask the user to input the index number to select a PDF
    try:
        selected_index = int(input("Enter the index number of the PDF you want to select: "))

        # Check if the index is valid
        if 1 <= selected_index < len(pdf_files) + 1:
            selected_pdf = pdf_files[selected_index - 1]
            selected_pdf_path = os.path.join(pdf_dir, selected_pdf)
            print(f"\nYou selected: {selected_pdf_path}")

            # Load the PDF using PyPDFLoader
            print("Loading PDF")
            loader = PyPDFLoader(selected_pdf_path)
            document = loader.load()
            
            # Training on 20% of the novel because fine-tuning is expensive
            half_point = len(document) // 5
            half_document = document[:half_point]
            # Prepare the fine-tuning data
            print("Preparing Data for Fine-tuning")
            fine_tuning_data = prepare_finetune_data(half_document)

            last_part = fine_tuning_data[-1]["messages"][-1]["content"]
            # Save it to a .jsonl file
            training_file = f"{fine_tune_data_dir}/{selected_pdf}.jsonl"
            print("Creating a jsonl file", training_file)
            with open(training_file, 'w') as jsonl_file:
                for entry in fine_tuning_data:
                    jsonl_file.write(json.dumps(entry) + '\n')
            print("Adding jsonl file to openai")
            openai_file = client.files.create(
              file=open(training_file, "rb"),
              purpose="fine-tune"
            )
            print("Starting a fine-tuning Job")
            fine_tine_job = client.fine_tuning.jobs.create(
              training_file=openai_file.id, 
              model=base_model,
              suffix=selected_pdf.replace(" ", "_")
            )
            with open(fine_tune_jobs_file, 'a') as models_file:
                data = {
                    "id": fine_tine_job.id, 
                    "last_part": last_part
                }
                models_file.write(json.dumps(data) + '\n')
            while client.fine_tuning.jobs.retrieve(fine_tine_job.id).fine_tuned_model == None:
                print("Waiting for fine-tuning to complete, this can take some time, you can exit and continue again later")
                sleep(15)
            fine_tune_model_name = client.fine_tuning.jobs.retrieve(fine_tine_job.id).fine_tuned_model

            print("Generating Next Part")
            generate_next_part(fine_tune_model_name, last_part)

        else:
            print("Invalid index number.")
    except ValueError:
        print("Please enter a valid number.")


