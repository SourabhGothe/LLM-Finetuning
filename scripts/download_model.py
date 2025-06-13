# download_model.py
# A simple script to download a model from the Hugging Face Hub using a command-line argument.

# ====================================================================
#  USER ACTION: Add your SSL bypassing imports and code here.
#  For example:
#
# import ssl
# import os
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# ====================================================================


from huggingface_hub import snapshot_download
import argparse # Import the argument parsing library
import os

# --- Command-Line Argument Parsing ---
# Create a parser object
parser = argparse.ArgumentParser(
    description="Download a model from the Hugging Face Hub.",
    epilog="Example: python download_model.py unsloth/deepseek-coder-6.7b-instruct-bnb-4bit"
)

# Add a required positional argument for the model name
parser.add_argument(
    "model_name",
    type=str,
    help="The full name of the model on the Hugging Face Hub (e.g., 'unsloth/llama-3-8b-Instruct-bnb-4bit')."
)

# Parse the command-line arguments
args = parser.parse_args()
model_name = args.model_name
# --- End of Argument Parsing ---


# Set a local directory for the download if needed, otherwise it uses the default cache
# cache_dir = "./model_cache" # Optional: uncomment to download to a local folder

print(f"--- Starting download for model: {model_name} ---")

try:
    # snapshot_download will download all files from the repository
    # and place them in the Hugging Face cache, or a specified local directory.
    snapshot_download(
        repo_id=model_name,
        repo_type="model",
        # cache_dir=cache_dir, # Optional: uncomment to use a local folder
        local_dir_use_symlinks=False # Set to False for Windows compatibility
    )
    print(f"\n--- Download Complete! ---")
    print(f"Model '{model_name}' is now available in your local Hugging Face cache.")
    print("You can now run the training script again.")

except Exception as e:
    print(f"\n--- An error occurred during download ---")
    print(f"Error: {e}")
    print("Please check the model name, your network connection, and SSL configuration.")

