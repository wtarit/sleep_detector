import requests
import os
from dotenv import load_dotenv

load_dotenv()


def lineNotify(message):
    payload = {'message': message}
    return _lineNotify(payload)


def notifyFile(msg, filename):
    file = {'imageFile': open(filename, 'rb')}
    payload = {'message': msg}
    return _lineNotify(payload, file)


def notifyPicture(url):
    payload = {'message': " ", 'imageThumbnail': url, 'imageFullsize': url}
    return _lineNotify(payload)


def notifySticker(stickerID, stickerPackageID):
    payload = {'message': " ", 'stickerPackageId': stickerPackageID,
               'stickerId': stickerID}
    return _lineNotify(payload)


def _lineNotify(payload, file=None):
    url = 'https://notify-api.line.me/api/notify'
    token = os.environ["LINE_TOKEN"]  # set in .env file
    headers = {'Authorization': 'Bearer '+token}
    return requests.post(url, headers=headers, data=payload, files=file)


if __name__ == "__main__":
    #lineNotify('ทดสอบภาษาไทย hello')
    notifyFile("elderly sleeping", "sleeping.jpg")
