import os
import subprocess
import json
from openai import AzureOpenAI
from scipy.spatial.distance import cosine
import logging

# Azure OpenAI Configuration
endpoint = os.getenv("ENDPOINT_URL", "https://aep-portal-hackathon-openai.openai.azure.com/")
embedding_model = os.getenv("DEPLOYMENT_NAME", "text-embedding-3-small-2")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "ROVN0oYw5UmCyykCSrQDMZ8aQ1cePL8W7ET1nJ7B0fM3mCSLcybkJQQJ99ALACYeBjFXJ3w3AAABACOG6hLG")
gpt_model = "gpt-4"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

def get_git_diff():
    """Gets the git diff of the latest commit."""
    try:
        diff_output = subprocess.check_output(["git", "diff", "--cached"], universal_newlines=True)
        return diff_output
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e}")
        return None

def is_text_file(filepath):
    """Check if a file is likely a text file by its extension."""
    file_extensions = ['.tsx','.jsx', '.js']
    # Add more text-based extensions as needed
    return any(filepath.endswith(ext) for ext in file_extensions)

def get_repo_files():
    """Get all files tracked by Git, filtering out non-text files."""
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to list repository files.")
    
    # Extract list of tracked files
    tracked_files = result.stdout.strip().split("\n")
    
    # Filter out non-text files
    text_files = [file for file in tracked_files if is_text_file(file)]
    
    print("Tracked text files in the repository:")
    for file in text_files:
        print(file)
    
    return text_files

def read_file_content(filepath):
    """Read the content of a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def calculate_chunk_size(content_size, max_token_limit=120000, num_chunks=40):
    """
    Dynamically calculates chunk size for splitting content.
    Args:
        content_size (int): The size of the content (in characters or tokens).
        max_token_limit (int): The maximum token limit allowed (default: 120000).
        num_chunks (int): Approximate number of chunks to split into (default: 40).
    Returns:
        int: The calculated chunk size.
    """
    return max(min(content_size // num_chunks, max_token_limit // 2), 3000)

def split_into_chunks(text, max_token_limit=120000, num_chunks=40):
    """
    Split text into smaller chunks within the token limit for embeddings.
    """
    chunk_size = calculate_chunk_size(len(text), max_token_limit, num_chunks)
    lines = text.splitlines()
    chunks = []
    current_chunk = []

    for line in lines:
        if len(" ".join(current_chunk) + " " + line) <= chunk_size:
            current_chunk.append(line)
        else:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
    
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks

def generate_embedding(text):
    """
    Generate embeddings for a given text. If the text is too large, split it into chunks.
    """
    chunks = split_into_chunks(text)
    chunk_embeddings = []

    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model=embedding_model,
        )
        # Access the embedding data correctly
        chunk_embeddings.append(response.data[0].embedding)
    
    # Average the embeddings to create a single vector representation
    avg_embedding = [sum(x) / len(chunks) for x in zip(*chunk_embeddings)]
    return avg_embedding

def is_new_file(filepath):
    """Check if the file is new by seeing if it exists in the previous commit."""
    try:
        subprocess.check_output(["git", "show", f"HEAD^:{filepath}"], stderr=subprocess.STDOUT)
        return False
    except subprocess.CalledProcessError:
        return True

def detect_redundancy(latest_changes, repo_files):
    """Detect redundancy in repository files."""
    redundancy_report = []

    for file in repo_files:
        if is_new_file(file):
            logging.info(f"Skipping new file {file} as it has no previous commit history.")
            continue

        try:
            # Get the previous version of the file from the previous commit
            file_content = subprocess.check_output(
                ["git", "show", f"HEAD^:{file}"], universal_newlines=True
            )
            # latest_file_content = read_file_content(file)
            latest_embedding = generate_embedding(latest_changes)
            file_embedding = generate_embedding(file_content)

            similarity = 1 - cosine(latest_embedding, file_embedding)
            logging.info(f"Similarity between the current and previous state of {file}: {similarity}")

            if similarity > 0.7:  # Threshold for redundancy detection
                redundancy_report.append({
                    "file": file,
                    "similarity": similarity,
                    "content": file_content,
                })
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")
            continue

    return redundancy_report

def analyze_changes(changes):
    """Analyze changes and generate suggestions using GPT-4."""
    prompt = (
        "You are an AI assistant reviewing code changes. Provide structured suggestions like this:\n\n"
        "File: <File Name>\nIssue: <Brief description of the issue>\n"
        "What Should Be Done: <Action to resolve the issue>\nWhy It Should Be Done: <Reasoning>\n\n"
        "---\n\nChanges:\n" + changes
    )
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.7,
    )
    return response.choices[0].message.content

def group_similar_files(redundancy_report, threshold=0.005):
    """
    Groups files based on similarity. Files with similarity above the threshold
    (difference in similarity <= threshold) are grouped together for a combined refactoring suggestion.
    """
    grouped_files = []
    current_group = []
    previous_similarity = None

    # Iterate through the redundancy report and extract file and similarity values
    for item in redundancy_report:
        file = item['file']
        similarity = item['similarity']

        # Group files with similar similarity scores
        if previous_similarity is None or abs(similarity - previous_similarity) <= threshold:
            current_group.append(file)
        else:
            if current_group:
                grouped_files.append(current_group)
            current_group = [file]

        previous_similarity = similarity

    # Add the last group if any files were added
    if current_group:
        grouped_files.append(current_group)
        print(f"Grouped files: {current_group}")

    return grouped_files


def generate_combined_refactor_suggestions(grouped_files, redundancy_report, latest_changes):
    """Generate combined refactoring suggestions for groups of similar redundant code."""
    suggestions = []
    logging.info("Starting to generate combined refactor suggestions...")

    # Iterate through each group of similar files
    for group in grouped_files:
        logging.info(f"Processing group of files: {group}")
        
        # Combine content of all files in the group
        combined_content = ""
        for file in group:
            file_redundancy = next(item for item in redundancy_report if item['file'] == file)
            combined_content += f"\n--- Content from {file} ---\n{file_redundancy['content']}"

        # Append the latest changes to the combined content for reference
        combined_content += f"\n--- Latest Changes ---\n{latest_changes}"

        # Split the combined content into chunks
        chunks = split_into_chunks(combined_content)
        logging.info(f"Split combined content into {len(chunks)} chunk(s) for group: {group}")

        for idx, chunk in enumerate(chunks):
            prompt = (
                f"The following code snippets are redundant with others in the repository:\n\n"
                f"Code Chunk {idx + 1}:\n{chunk}\n\n"
                "Suggest how to refactor these redundant codes into a reusable function or improve the structure."
            )
            try:
                logging.info(f"Generating combined refactor suggestion for chunk {idx + 1} of group {group}...")
                response = client.chat.completions.create(
                    model=gpt_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7,
                )
                suggestion = response.choices[0].message.content
                suggestions.append({
                    "group": group,
                    "chunk": idx + 1,
                    "refactor_suggestion": suggestion,
                })
                logging.info(f"Successfully generated refactor suggestion for chunk {idx + 1} of group {group}")
            except Exception as e:
                logging.error(f"Error generating refactor suggestion for chunk {idx + 1} of group {group}: {e}")

    logging.info("Combined refactor suggestions generation completed.")
    return suggestions

def generate_report(changes, suggestions, redundancy_report, refactor_suggestions):
    """Generate a combined report in text and JSON format."""
    report = {
        "commit_changes_feedback": suggestions,
        "redundancy_report": redundancy_report,
        "refactor_suggestions": refactor_suggestions,
    }

    # Save JSON report
    with open("repository_analysis_report.json", "w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=4)

    # Save Text Report
    with open("repository_analysis_report.txt", "w", encoding="utf-8") as text_file:
        text_file.write("### Repository Analysis Report ###\n\n")
        text_file.write(f"### Feedback on Latest Commit Changes ###\n\n{suggestions}\n\n")
        text_file.write("### Redundancy Report ###\n")
        for item in redundancy_report:
            text_file.write(f"File: {item['file']}\nSimilarity: {item['similarity']:.2f}\n\n")
        text_file.write("### Refactoring Suggestions ###\n")
        # for suggestion in refactor_suggestions:
        #     text_file.write(f"File: {suggestion['file']}\nSuggestion: {suggestion['refactor_suggestion']}\n\n")
        for suggestion in refactor_suggestions:
            try:
                # Check if 'file' and 'refactor_suggestion' keys exist in the suggestion
                file = suggestion.get('file', 'No file specified')  # Default to 'No file specified' if missing
                refactor_suggestion = suggestion.get('refactor_suggestion', 'No suggestion provided')  # Default message

                # Write the suggestion to the report
                text_file.write(f"File: {file}\nSuggestion: {refactor_suggestion}\n\n")
            except KeyError as e:
                # Log and handle the error
                print(f"KeyError: {e} in refactor suggestion: {suggestion}")
                text_file.write("Error: Missing key in refactor suggestion.\n\n")

    print("Reports generated: repository_analysis_report.json and repository_analysis_report.txt")

def main():
    """Main function to analyze the repository."""
    changes = get_git_diff()
    if not changes:
        print("No changes found in the latest commit.")
        return

    print("Analyzing changes...")
    suggestions = analyze_changes(changes)

    print("Scanning repository for redundancy...")
    repo_files = get_repo_files()
    redundancy_report = detect_redundancy(changes, repo_files)

    # Group similar files based on similarity threshold
    print("Grouping similar files...")
    grouped_files = group_similar_files(redundancy_report, threshold=0.005)

    print("Generating refactor suggestions...")
    # refactor_suggestions = generate_refactor_suggestions(redundancy_report)
    refactor_suggestions = generate_combined_refactor_suggestions(grouped_files, redundancy_report, changes)

    print("Generating final report...")
    generate_report(changes, suggestions, redundancy_report, refactor_suggestions)

if __name__ == "__main__":
    main()
