import os
import subprocess
from openai import AzureOpenAI

# Azure OpenAI Configuration
endpoint = os.getenv("ENDPOINT_URL", "https://aep-portal-hackathon-openai.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "TestDeployment0621")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "API_key")

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


def analyze_changes(changes):
    """Analyzes changes and generates suggestions in a structured format."""
    chat_prompt = [
        {
            "role": "system",
             "content": (
                "You are an AI assistant that reviews code changes and provides concise, structured suggestions. "
                "Please format the response like this:\n\n"
                "### Commit Analysis Report ###\n\n"
                "For each issue, follow this format:\n"
                "File: <File Name>\n"
                "Issue: <Brief description of the issue>\n"
                "What Should Be Done: <Action to resolve the issue>\n"
                "Why It Should Be Done: <Brief reasoning>\n\n"
                "---\n\n"
                "No numbering, only one heading at the top."
            ),
        },
        {
            "role": "user",
            "content": f"Here are the changes in the latest commit:\n\n{changes}",
        },
    ]

    # Generate completion
    completion = client.chat.completions.create(
        model=deployment,
        messages=chat_prompt,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
    )
    choices = completion.choices
    if choices:
        return choices[0].message.content
    return "No suggestions generated."


def format_suggestions(raw_suggestions, changes):
    """Formats the raw suggestions into a structured and presentable format."""
    formatted_suggestions = []
    lines = changes.split("\n")
    current_file = None

    for line in lines:
        if line.startswith("diff --git"):
            # Extract file name
            parts = line.split(" ")
            if len(parts) > 2:
                current_file = parts[2].replace("b/", "")
        
        if line.startswith("+") and not line.startswith("+++") :
            # Capture added lines for context
            formatted_suggestions.append({
                "file": current_file,
                "line": line.strip()
            })

    # Combine suggestions with context (without numbering)
    structured_suggestions = []
    for suggestion in raw_suggestions.split("\n"):
        structured_suggestions.append(suggestion.strip())

    return "\n".join(structured_suggestions)


def generate_report(suggestions, changes):
    """Generates a JSON report and a formatted text report."""
    formatted_suggestions = format_suggestions(suggestions, changes)

    # Text Report
    with open("commit_analysis_report.txt", "w", encoding="utf-8") as report_file:
        report_file.write(formatted_suggestions)

    print("Reports generated: commit_analysis_report.txt")


def main():
    # Get the git diff
    changes = get_git_diff()
    if not changes:
        print("No changes found in the latest commit.")
        return

    # Analyze the changes
    print("Analyzing changes...")
    suggestions = analyze_changes(changes)

    # Generate the reports
    generate_report(suggestions, changes)


if __name__ == "__main__":
    main()