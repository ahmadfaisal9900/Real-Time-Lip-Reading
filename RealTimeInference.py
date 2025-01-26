import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token='')
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token='', invert=True)

model = Sequential()
model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))
#its going to be 75, 41 output since 75 frames and 41 chars
model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))


#from the ASR paper 
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


class ProduceExample(tf.keras.callbacks.Callback): 
    def __init__(self, dataset) -> None: 
        self.dataset = dataset.as_numpy_iterator()
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):           
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss=CTCLoss)

model.load_weights(r'E:\Projects\Unfinished\Lip Reading\models - checkpoint 96\checkpoint')

# Function to preprocess a single frame and isolate the mouth region
def preprocess_frame(frame):
    # Convert to grayscale
    frame = tf.image.rgb_to_grayscale(frame)
    # Isolate mouth region (adjust the coordinates as needed)
    mouth_region = frame[200:300, 150:290, :]  # Adjusted coordinates for a lower and right-shifted region
    # Resize to match the input shape of the model
    mouth_region = cv2.resize(mouth_region.numpy(), (140, 46))
    # Normalize the frame
    mouth_region = mouth_region / 255.0
    # Expand dimensions to match the input shape (46, 140, 1)
    mouth_region = np.expand_dims(mouth_region, axis=-1)
    return mouth_region

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Buffer to store the sequence of frames
frame_buffer = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    start_point = (150, 200)  # Top-left corner
    end_point = (300, 290)    # Bottom-right corner
    color = (0, 0, 255)       # Red color in BGR
    thickness = 2             # Thickness of the rectangle border
    cv2.rectangle(frame, start_point, end_point, color, thickness)



    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    frame_buffer.append(preprocessed_frame)

    # Ensure the buffer has exactly 75 frames
    if len(frame_buffer) > 75:
        frame_buffer.pop(0)

    # Perform inference if we have enough frames
    if len(frame_buffer) == 75:
        input_data = np.expand_dims(np.array(frame_buffer), axis=0)
        input_data = np.repeat(input_data, 2, axis=0)  # Repeat to create a batch of 2

        # Perform inference
        try:
            yhat = model.predict(input_data)
            decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75, 75], greedy=True)[0][0].numpy()
            decoded_sentences = [tf.strings.reduce_join([num_to_char(word) for word in sentence], axis=-1).numpy().decode('utf-8') for sentence in decoded]
        except Exception as e:
            print(f"Error during prediction: {e}")
            break

        # Process the prediction as needed
        # For example, display the prediction on the frame
        cv2.putText(frame, f'Prediction: {decoded_sentences[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Inference', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()