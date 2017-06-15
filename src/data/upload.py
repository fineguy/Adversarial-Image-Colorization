import argparse
import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload dataset to floyd from Google Drive.')
    parser.add_argument('-i', '--id', required=True, help="Google Drive file id.")
    parser.add_argument('-n', '--name', default='baikal', help="Save file under the name.")
    args = parser.parse_args()
    download_file_from_google_drive(args.id, '/output/{}.h5'.format(args.name))
