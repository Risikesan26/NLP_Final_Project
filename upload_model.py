import os
from huggingface_hub import HfApi, login

def upload():
    repo_id = "Risikesan/cardiffnlp_roberta"
    folder_path = "app_roberta_model"
    
    if not os.path.exists(folder_path):
        print(f"Error: Local model folder '{folder_path}' not found!")
        return

    print("=========================================")
    print("HUGGING FACE HUB UPLOADER")
    print("=========================================")
    # Prompt for Hugging Face token
    token = input("Enter your Hugging Face Write Token: ").strip()
    if not token:
        print("Error: Hugging Face token is required to upload.")
        return

    try:
        print("Logging in to Hugging Face...")
        login(token=token)
        
        print(f"Uploading '{folder_path}' folder to repository '{repo_id}' on Hugging Face Hub...")
        api = HfApi()
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print("\nSUCCESS! All model files have been uploaded to Hugging Face Hub.")
        print(f"Verify files at: https://huggingface.co/{repo_id}/tree/main")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    upload()
