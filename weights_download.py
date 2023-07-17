import urllib.request

url = 'https://pjreddie.com/media/files/yolov3.weights'
filename = 'yolov3.weights'

urllib.request.urlretrieve(url, filename)
