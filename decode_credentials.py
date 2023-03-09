import base64
import json
import os


def get_decoded_env(env_var):
    encoded_token = os.environ.get(env_var)
    decoded_token = base64.b64decode(encoded_token).decode('utf-8')
    return decoded_token
    

def decode_base64_str(base64_str: str, filename: str):
    bytes_str = base64.b64decode(base64_str)
    with open(filename, "w") as f:
        f.write(bytes_str.decode('utf-8'))

if __name__ == "__main__":
    env_file_encoded = os.environ.get("ENV_FILE")
    decode_base64_str(env_file_encoded, ".env")