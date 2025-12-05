import io
from PIL import ImageGrab
from cryptography.fernet import Fernet
import os

# --- Re-use the secure key logic ---
# For a full application, the key generation would be part of an initial setup
KEY_FILE = "secret.key"
def generate_or_load_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as key_file:
            key_file.write(key)
    else:
        with open(KEY_FILE, "rb") as key_file:
            key = key_file.read()
    return key
key = generate_or_load_key()
fernet = Fernet(key)
# --- End re-used logic ---


def take_and_encrypt_screenshot(output_filename):
    """Takes a screenshot, encrypts it, and saves it to a file."""
    print("Taking screenshot...")
    screenshot = ImageGrab.grab()

    # Convert the Pillow Image to a byte stream
    byte_stream = io.BytesIO()
    screenshot.save(byte_stream, format='PNG')
    byte_stream.seek(0)
    image_data = byte_stream.getvalue()

    # Encrypt the image data
    print("Encrypting screenshot...")
    encrypted_data = fernet.encrypt(image_data)

    # Save the encrypted data to a file
    with open(output_filename, 'wb') as f:
        f.write(encrypted_data)
    
    print(f"Screenshot saved to {output_filename} and encrypted.")

if __name__ == "__main__":
    take_and_encrypt_screenshot("encrypted_screenshot.enc")
