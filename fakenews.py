import gradio as gr
import json
import requests
import os
import time
import csv
import io

# Set up the API endpoint and key.
# IMPORTANT: The API key is not being automatically provided by the Canvas environment.
# Please get your API key from Google AI Studio and paste it in the line below.
# https://aistudio.google.com/app/apikey
API_KEY = "AIzaSyDEC5x3942xy-aNQLKY_Iw9_1kvdyG1KHc"
# FIX: Updated the model name in the URL from the outdated preview name to the current alias.
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key="

# Define the system instruction for the LLM to act as a fact-checker.
SYSTEM_INSTRUCTION = {
    "parts": [
        {
            "text": """
            You are a world-class fact-checker. Your task is to analyze a given news claim,
            verify its accuracy using Google Search, and provide a concise, factual summary.
            Always include citations for your claims, referencing the sources found by Google Search.
            If a claim is false or misleading, explain why based on the evidence you find.
            If you cannot definitively verify or refute the claim, state that clearly.
            Provide your response in a well-formatted markdown.
            """
        }
    ]
}

def log_to_csv(status, response_text):
    """Logs the fact-checking result to a CSV file."""
    HISTORY_FILE = "fact_check_history.csv"

    # Check if file exists to write header
    file_exists = os.path.isfile(HISTORY_FILE)

    with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Status", "Response"])
        writer.writerow([status, response_text])

    return HISTORY_FILE

def check_news(news_claim):
    """
    Checks the factual accuracy of a news claim using the Gemini API with Google Search grounding.
    """
    # NOTE: The API key check has been removed as the key is now provided.
    if not news_claim:
        return "", "Please enter a news claim to check.", None

    # Construct the payload for the Gemini API call.
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": news_claim
                    }
                ]
            }
        ],
        "systemInstruction": SYSTEM_INSTRUCTION,
        "tools": [
            {
                "google_search": {}
            }
        ]
    }

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        # A simple retry mechanism with exponential backoff for API calls.
        for i in range(3):
            try:
                # Make the API call to the Gemini API.
                response = requests.post(
                    f"{API_URL}{API_KEY}",
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=60
                )
                response.raise_for_status() # Raise an exception for bad status codes
                break # Exit the loop if the request was successful
            except requests.exceptions.HTTPError as e:
                # Handle 429 Too Many Requests error with backoff
                if response.status_code == 429:
                    print(f"Rate limit exceeded. Retrying in {2**i} seconds...")
                    time.sleep(2**i)
                else:
                    raise e
        else:
            error_message = "Failed to get a response from the API after multiple retries due to rate limiting."
            return "", error_message, None

        # Parse the JSON response.
        result = response.json()
        candidate = result.get("candidates", [])[0]
        text_response = candidate.get("content", {}).get("parts", [])[0].get("text", "No response text found.")

        # Determine status based on response content
        status = "No"  # Default to No if not explicitly found as Yes
        if "true" in text_response.lower() or "factually accurate" in text_response.lower() or "verified" in text_response.lower():
            status = "Yes"

        # Extract grounding metadata for citations.
        sources = []
        grounding_metadata = candidate.get("groundingMetadata")
        if grounding_metadata and grounding_metadata.get("groundingAttributions"):
            sources = [
                {
                    "uri": attr.get("web", {}).get("uri"),
                    "title": attr.get("web", {}).get("title"),
                    "domain": attr.get("web", {}).get("domain")
                }
                for attr in grounding_metadata["groundingAttributions"]
            ]

        citations_markdown = ""
        if sources:
            citations_markdown = "\n\n### Citations\n"
            for i, source in enumerate(sources):
                citations_markdown += f"- [{i+1}]({source['uri']}) {source['title']}\n"

        full_response = text_response + citations_markdown

        # Log the result to a CSV file
        csv_file_path = log_to_csv(status, full_response)

        output_markdown = f"**{status}.** " + full_response

        return "", output_markdown, csv_file_path

    except Exception as e:
        print(f"An error occurred: {e}")
        error_message = f"An error occurred: {e}"
        return "", error_message, None

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Fake News Detection in Gradio") as demo:
    gr.Markdown("# Fake News Detection in Gradio")
    gr.Markdown(
        """
        Enter a news claim or a statement, and this tool will use the latest information
        from Google Search to verify its accuracy and provide a grounded response with citations.
        This approach can handle new, "unseen" data that a pre-trained model would not have.
        """
    )

    gr.Markdown("### How This App Improves Accuracy for New Information")
    gr.Markdown(
        """
        Unlike traditional fake news detectors that use a static, pre-trained model, this application
        uses a Large Language Model (LLM) with **real-time Google Search grounding**. This means:

        -   **Up-to-date Verification**: The model performs a live web search for every claim, ensuring it uses the most current data available.
        -   **Reduced Hallucinations**: Responses are anchored to verifiable web sources, significantly lowering the chance of making things up.
        -   **Increased Transparency**: The app provides direct citations so you can easily review the original sources yourself.

        This methodology is a significant improvement for detecting fake news that is too recent for older models.
        """
    )
    with gr.Row():
        text_input = gr.Textbox(
            placeholder="Enter a news claim, e.g., 'A new species of glowing fish was discovered in the Amazon river last week.'",
            label="News Claim"
        )

    with gr.Row():
        check_button = gr.Button("Fact Check", variant="primary")

    output_text = gr.Markdown(label="Fact-Check Result")
    download_btn = gr.File(label="Download History", interactive=False)

    # Link the button click event to the check_news function and update all necessary components.
    check_button.click(
        fn=check_news,
        inputs=[text_input],
        outputs=[text_input, output_text, download_btn]
    )

# Launch the Gradio app
demo.launch(share=True)
