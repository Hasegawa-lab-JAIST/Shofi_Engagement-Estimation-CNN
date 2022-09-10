from flask import Flask, render_template, Response
from camera import VideoCamera
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


app = Flask(__name__)

@app.route('/')
def index(): #rendering webpage
    return render_template('index.html')

def gen(camera): ##activate VideoCamera feed
    while True: #get camera webpage
        frame = camera.get_frame() #get the feed frame by frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed') #get the prediction from the VideoCamera class
def video_feed():
    return Response(gen(VideoCamera()), #the prediction back to the web interface
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__': #defining server ip address and port
    app.run(host='0.0.0.0', port='5000', debug=True) #the app is running at localhost. the default port is 5000
    #app.run(host='http://127.0.0.1', port='5000', debug=True) #the app is running at localhost. the default port is 5000
