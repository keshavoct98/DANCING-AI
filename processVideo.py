import cv2
import time
import numpy as np
import pandas as pd
import librosa

protoFile = "models/pose_deploy_linevec.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
inWidth, inHeight, threshold = 256, 256, 0.3

POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13]]

def XY(df, audio_input):
    ''' Missing values in pose coordinates are replaced using forward and 
    backward filling method. Rows left with missing values after applying mentioned methods
    are deleted. Last few rows are dropped to match lengths of X and Y.
    '''
    min_length = min(audio_input.shape[0], df.shape[0])
    X = audio_input[:min_length, :]
    Y = df.iloc[:min_length, :]
    
    Y = Y.astype('float64')
    Y.replace(to_replace=-1, value=np.nan, inplace = True)
    Y.fillna(method='ffill',axis = 1, inplace = True)
    Y.fillna(method='bfill',axis = 1, inplace = True)
    Y = Y.dropna()
    
    X = X[Y.index]
    Y = Y.values
    
    return X, Y

def video(vid_path, aud_path):
    ''' Displays video with estimated pose. Returns pose coordinates
    and audio tempogram as numpy array.'''
    cap = cv2.VideoCapture(vid_path)
    hasFrame, frame = cap.read()
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]

    df = pd.DataFrame(columns = range(28))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    print('Video Processed(In sec):-', end = ' ')
    while(1):
        frame_count = frame_count + 1
        if frame_count % int(fps) == 0:
            print(int(frame_count / fps), end=', ', flush=True)
        t = time.time()  
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        H, W = output.shape[2], output.shape[3]

        points = []
        list_coordinates = []
        for i in range(14):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x, y = (frameWidth * point[0]) / W, (frameHeight * point[1]) / H

            if prob > threshold :
                points.append((int(x), int(y)))
                list_coordinates.extend([int(x), int(y)])                  
            else :
                points.append(None)
                list_coordinates.extend([-1, -1])
        df.loc[len(df)] = list_coordinates
        
        for pair in POSE_PAIRS:
            partA, partB = pair[0], pair[1]
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (255, 0, 0), 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), 
                    cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.imshow('Output-Skeleton', frame)
        if cv2.waitKey(1) == 27: break
    
    cv2.destroyAllWindows()
    print()
    y, sr = librosa.load(aud_path)
    audio_input = np.transpose(librosa.feature.tempogram(y, sr, hop_length = int(sr/fps), win_length = 36))

    return XY(df, audio_input)