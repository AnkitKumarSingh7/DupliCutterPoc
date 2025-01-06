import os
import subprocess
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
    """
    Gets the git diff of the latest staged changes.

    This function retrieves the differences between the files in the staging area and the latest committed state.

    Returns:
        str: The git diff output as a string, or None if an error occurs.

    Exceptions:
        Handles CalledProcessError if the git command fails (e.g., no repository, no staged changes).
    """
    try:
        # Run the `git diff --cached` command to get the staged changes
        # `universal_newlines=True` ensures the output is returned as a string
        diff_output = subprocess.check_output(["git", "diff", "--cached"], universal_newlines=True)
        return diff_output
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e}")
        return None

def analyze_changes(changes):
    """
    Analyze changes and generate suggestions using GPT-4.
    
    Parameters:
    - changes (str): A string containing the code changes to be analyzed.

    Returns:
    - str: A structured response containing suggestions for improvement.
    """
    prompt = (
        "You are an AI assistant reviewing code changes. Provide structured suggestions like this:\n\n"
        "File: <File Name>\nIssue: <Brief description of the issue>\n"
        "What Should Be Done: <Action to resolve the issue>\nWhy It Should Be Done: <Reasoning>\n\n"
        "---\n\nChanges:\n" + changes
    )
    # Call the GPT model with the provided prompt and parameters.
    response = client.chat.completions.create(
        model=gpt_model,  # Specify the GPT model to use.
        messages=[{"role": "user", "content": prompt}],  # Provide the prompt in the user role.
        max_tokens=800,  # Limit the response to 800 tokens to control output length.
        temperature=0.7,  # Set the temperature to balance creativity and focus.
    )
    return response.choices[0].message.content

def is_code_file(filepath):
    """Check if a file is likely a code file by its extension."""
    file_extensions = ['.tsx', '.jsx']
    
    return any(filepath.endswith(ext) for ext in file_extensions)

def get_repo_files():
    """
    Get all files tracked by Git, filtering out non-code files.
    Uses the `git ls-files` command to list tracked files and applies a filter
    based on file extensions defined in `is_code_file`.
    """
    # Run the `git ls-files` command to get all tracked files in the repository
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    
    # If the Git command fails, raise an error
    if result.returncode != 0:
        raise RuntimeError("Failed to list repository files.")
    
    # Extract the list of tracked files from the command output
    tracked_files = result.stdout.strip().split("\n")
    
    text_files = [file for file in tracked_files if is_code_file(file)]
    
    # Print the filtered list of code files
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
    Dynamically calculates the chunk size for splitting content.
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
    Splits text into smaller chunks that fit within the calculated token limit.
    Args:
        text (str): The input text to split.
        max_token_limit (int): The maximum token limit allowed (default: 120000).
        num_chunks (int): Approximate number of chunks to split into (default: 40).
    Returns:
        list: A list of text chunks.
    """
    chunk_size = calculate_chunk_size(len(text), max_token_limit, num_chunks)
    
    # Split the text into lines for easier chunking
    lines = text.splitlines()
    chunks = []  # List to store chunks
    current_chunk = []  # Buffer for the current chunk

    # Iterate through lines and add them to the current chunk until the size limit is reached
    for line in lines:
        if len(" ".join(current_chunk) + " " + line) <= chunk_size:
            current_chunk.append(line)
        else:
            # Save the completed chunk and start a new one
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks

def generate_embedding(text):
    """
    Generates embeddings for the given text. Splits text into manageable chunks if needed.
    Args:
        text (str): The input text.
    Returns:
        list: A single averaged embedding vector for the input text.
    """
    # Split text into chunks to stay within token limits
    chunks = split_into_chunks(text)
    chunk_embeddings = []  # Store embeddings for each chunk

    # Generate embeddings for each chunk
    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,  # Input text for the embedding model
            model=embedding_model,  # The embedding model to use
        )
        # Extract the embedding vector from the response
        chunk_embeddings.append(response.data[0].embedding)
    
    # Average all chunk embeddings to create a single embedding vector
    avg_embedding = [sum(x) / len(chunks) for x in zip(*chunk_embeddings)]
    return avg_embedding

def detect_redundancy(latest_changes, repo_files):
    """
    Detect redundancy by comparing the latest changes with repository files.
    Args:
        latest_changes (str): The content of the latest changes.
        repo_files (list): A list of repository files to compare against.
    Returns:
        list: A redundancy report containing files with high similarity to the latest changes.
    """
    # Generate an embedding for the latest changes
    latest_embedding = generate_embedding(latest_changes)
    redundancy_report = []  # Store details of redundant files

    # Compare the latest embedding with embeddings of all repository files
    for file in repo_files:
        try:
            # Read the content of the current file
            file_content = read_file_content(file)

            # Generate an embedding for the file content
            file_embedding = generate_embedding(file_content)

            # Calculate similarity between the latest changes and the file
            similarity = 1 - cosine(latest_embedding, file_embedding)

            # If the similarity exceeds the threshold, record the redundancy
            if similarity > 0.9:  # Threshold for redundancy detection
                redundancy_report.append({
                    "file": file,            # File name
                    "similarity": similarity, # Similarity score
                    "content": file_content,  # File content
                    "changes": latest_changes,  # Latest changes
                })
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    return redundancy_report

def group_similar_files(redundancy_report, threshold=0.02):
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

def generate_combined_refactor_suggestions(grouped_files, redundancy_report):
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
    """Generate a combined report in text format."""

    # Save Text Report
    with open("commit_feedback_report.txt", "w", encoding="utf-8") as text_file:
        text_file.write("### Repository Analysis Report ###\n\n")
        text_file.write(f"### Feedback on Latest Commit Changes ###\n\n{suggestions}\n\n")

    with open("commit_redundancy_refactor_report.txt", "w", encoding="utf-8") as text_file:   
        text_file.write("### Redundancy Report ###\n")
        for item in redundancy_report:
            text_file.write(f"File: {item['file']}\nSimilarity: {item['similarity']:.2f}\n\n")
        text_file.write("### Refactoring Suggestions ###\n")
        for suggestion in refactor_suggestions:
            try:
                # Check if 'file' and 'refactor_suggestion' keys exist in the suggestion
                group = suggestion.get('group', 'No group specified')  # Default to 'No group specified' if missing
                refactor_suggestion = suggestion.get('refactor_suggestion', 'No suggestion provided')  # Default message

                # Write the suggestion to the report
                text_file.write(f"Group: {group}\nSuggestion: {refactor_suggestion}\n\n")
            except KeyError as e:
                # Log and handle the error
                print(f"KeyError: {e} in refactor suggestion: {suggestion}")
                text_file.write("Error: Missing key in refactor suggestion.\n\n")

    print("Reports generated: commit_feedback_report.txt and commit_redundancy_refactor_report.txt")

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
    grouped_files = group_similar_files(redundancy_report, threshold=0.02)

    print("Generating refactor suggestions...")
    refactor_suggestions = generate_combined_refactor_suggestions(grouped_files, redundancy_report)

    print("Generating final report...")
    generate_report(changes, suggestions, redundancy_report, refactor_suggestions)

if __name__ == "__main__":
    main()
