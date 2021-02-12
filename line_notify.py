import requests

def lineNotify(message):
    payload = {'message':message}
    return _lineNotify(payload)

def notifyFile(msg, filename):
    file = {'imageFile':open(filename,'rb')}
    payload = {'message': msg}
    return _lineNotify(payload,file)

def notifyPicture(url):
    payload = {'message':" ",'imageThumbnail':url,'imageFullsize':url}
    return _lineNotify(payload)

def notifySticker(stickerID,stickerPackageID):
    payload = {'message':" ",'stickerPackageId':stickerPackageID,'stickerId':stickerID}
    return _lineNotify(payload)

def _lineNotify(payload,file=None):
    url = 'https://notify-api.line.me/api/notify'
    token = 'vVwVFagE6c7DKGrO6qhZyW5tESbmAes2jnI6hbCkgqw'	#EDIT
    headers = {'Authorization':'Bearer '+token}
    return requests.post(url, headers=headers , data = payload, files=file)

if __name__ == "__main__":
    #lineNotify('ทดสอบภาษาไทย hello')
    notifyFile("elderly sleeping","20201227_200014.jpg")
