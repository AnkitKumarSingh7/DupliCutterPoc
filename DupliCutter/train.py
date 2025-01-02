import os
from openai import AzureOpenAI

# Configuration
CONFIG = {
    "folders_to_track": [
        "../react-demo/src"
    ]
}

endpoint = os.getenv("ENDPOINT_URL", "https://aep-portal-hackathon-openai.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "TestDeployment0621")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "API_key")

# Initialize Azure OpenAI client with key-based authentication
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

def gather_files(folders):
    """Gathers all files from the given list of folders."""
    all_files = []
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                all_files.append(full_path)
    return all_files

def train_model_on_files(files):
    """Trains the model on the given list of files."""
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            # Prepare chat prompt
            chat_prompt = [
                {"role": "system", "content": "You are an AI assistant trained to analyze and refactor code."},
                {"role": "user", "content": f"Here is the content of the file:\n\n{content}"}
            ]
            # Call the Azure OpenAI API
            completion = client.chat.completions.create(
                model=deployment,
                messages=chat_prompt,
                max_tokens=800,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            print(f"Trained on file: {file}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")

def main():
    # Gather all files from the configured folders
    base_path = os.getcwd()  # Assuming this script runs from the root of the project
    folders = [os.path.join(base_path, folder) for folder in CONFIG["folders_to_track"]]
    all_files = gather_files(folders)

    # Log files being processed
    print("Files to be processed for training:")
    for file in all_files:
        print(file)

    # Train model on the gathered files
    train_model_on_files(all_files)

if __name__ == "__main__":
    main()
