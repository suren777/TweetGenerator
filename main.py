from flask import Flask, render_template
from flask_socketio import SocketIO
from CODE.ANN.modelInference import reply_model
from CODE.ANN.model import DialogueModel

filename = 'FILES/SavedModels/model-train.hdf5'
tfmodel = DialogueModel(location=filename)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

@app.route('/')
def sessions():
    return render_template('session.html')

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

def messageSent(methods=['GET', 'POST']):
    print('message sent!!!')

@socketio.on('user connected')
def handle_question(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
   #socketio.emit('my question', json, callback=messageReceived)#

@socketio.on('my event')
def handle_question(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
    socketio.emit('my question', json, callback=messageReceived)

@socketio.on('bot listens')
def handle_bot_reply(json, methods=['GET', 'POST']):
    print('received bot listens: ' + str(json))
    json['msg_username'] = 'Bot'
    newMessage = reply_model(json['message'], tfmodel)
    json['message'] = newMessage
    socketio.emit('bot replies', json, callback=messageSent())

if __name__ == '__main__':
    socketio.run(app, debug=True)
