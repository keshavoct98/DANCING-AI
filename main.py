import os
import pandas as pd
from processVideo import video
from train import train
from display import displayResults

from absl import app, flags
from absl.flags import FLAGS
flags.DEFINE_string('video', 'data/0.mp4', 'Input video path')
flags.DEFINE_string('audio', 'data/0.wav', 'Input video path')
flags.DEFINE_string('background', 'inputs/background0.jpg', 'path to background image')

def main(_argv):
    
    head, tail = os.path.split(FLAGS.video)
    csv_xpath = 'data/'+tail.split('.')[0]+'X.csv'
    csv_ypath = 'data/'+tail.split('.')[0]+'Y.csv' 
    if os.path.isfile(csv_xpath) and os.path.isfile(csv_ypath):
        ''' If csv file is found, pose estimation part is skipped and 
            lstm is directly trained on the inputs from csv file.'''
        X, Y = pd.read_csv(csv_xpath, header = None), pd.read_csv(csv_ypath, header = None)
        X, Y = X.values, Y.values
    else:
        X, Y = video(FLAGS.video, FLAGS.audio) # Returns pose estimation coordinates of video and tempogram of input audio.
        pd.DataFrame(X).to_csv(csv_xpath, header = None, index = None)
        pd.DataFrame(Y).to_csv(csv_ypath, header = None, index = None)
        
    predictions = train(X, Y) # Split data, train lstm and return predictions.
    displayResults(predictions, FLAGS.background) # display and save output video
    print('video saved at "output/output.avi"')
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass