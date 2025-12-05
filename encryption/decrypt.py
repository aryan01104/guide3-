from PIL import Image
import io
from cryptography.fernet import Fernet
import os

# --- Re-use the secure key logic ---
KEY_FILE = "secret.key"
def generate_or_load_key():
    if not os.path.exists(KEY_FILE):
        raise FileNotFoundError("Encryption key not found. Run the encryption script first.")
    with open(KEY_FILE, "rb") as key_file:
        key = key_file.read()
    return key
key = generate_or_load_key()
fernet = Fernet(key)
# --- End re-used logic ---


def decrypt_and_view_screenshot(input_filename):
    """Reads and decrypts a file, then shows the image."""
    print("Reading encrypted screenshot...")
    try:
        with open(input_filename, 'rb') as f:
            encrypted_data = f.read()
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found.")
        return

    # Decrypt the image data
    print("Decrypting screenshot...")
    decrypted_data = fernet.decrypt(encrypted_data)
    
    # Open the image from the decrypted byte stream
    decrypted_image = Image.open(io.BytesIO(decrypted_data))

    # Show the image
    print("Displaying decrypted screenshot...")
    decrypted_image.show()

if __name__ == "__main__":
    decrypt_and_view_screenshot("encrypted_screenshot.enc")
