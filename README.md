# DANCING-AI
1. Extraction of pose coordinates from dance videos using openpose human pose estimation.
2. Training LSTM network on extracted coordinates using audio as input and coordinates as output.
3. Trained lstm is used to predict dance coordinates using the remaining song as input( 95% of the audio is used for training and 5% is used for predictions ).
4. Display output video by joining predicted coordinates to form a dancing human stick figure.

### Requirements
&nbsp;&nbsp; keras==2.3.1 </br>
&nbsp;&nbsp; librosa==0.7.2 </br>
&nbsp;&nbsp; moviepy==1.0.1 </br>
&nbsp;&nbsp; opencv-python==4.2.0.34 </br>
&nbsp;&nbsp; pytube3==9.6.4 </br>
&nbsp;&nbsp; tensorflow==2.2.0 </br>

### Training/Demo
1. Run **get_data.py** to download videos and audios to data folder. You can add youtube videos links to "video_links.txt" file for downloading. Alternatively you can copy videos(In '.mp4' format) and audios(In '.wav' format) directly to the data folder.
2. Download pretrained weights for pose estimation from [here](https://www.kaggle.com/changethetuneman/openpose-model). Download **pose_iter_440000.caffemodel** and save it in "models" folder.
2. Run **main.py** to train lstm and display results.
<pre><code> python main.py --video "path to input video" --audio "path to input audio" --background "path to background image"
 Example - python main.py --video data/0.mp4 --audio data/0.wav --background inputs/bg0.jpg </code></pre>
 &nbsp;&nbsp; #Note - If the gpu-ram is 3 GB or less, Reduce memory-limit in this [line](https://github.com/keshavoct98/DANCING-AI/blob/23d12312bb8f9c03fcd3e28ba4217cd0e7c38d52/train.py#L9) to a value less than your gpu-ram.
 
 ### Pose estimation using openpose
 <p> <img src="https://github.com/keshavoct98/DANCING-AI/blob/master/outputs/pose1.gif" width="98.4%" height="100%"/>
 <img src="https://github.com/keshavoct98/DANCING-AI/blob/master/outputs/pose0.gif" width="49%" height="50%"/> 
 <img src="https://github.com/keshavoct98/DANCING-AI/blob/master/outputs/pose2.gif" width="49%" height="50%"/> </p>
 
 ### Predictions
 <p> <img src="https://github.com/keshavoct98/DANCING-AI/blob/master/outputs/output1.gif" width="98.4%" height="100%"/>
 <img src="https://github.com/keshavoct98/DANCING-AI/blob/master/outputs/output0.gif" width="49%" height="50%"/> 
 <img src="https://github.com/keshavoct98/DANCING-AI/blob/master/outputs/output2.gif" width="49%" height="50%"/> </p>
 
 ### References
 1. https://github.com/CMU-Perceptual-Computing-Lab/openpose
 2. https://python-pytube.readthedocs.io/en/latest/
 3. https://zulko.github.io/moviepy/
 4. https://librosa.org/librosa/
 5. https://www.youtube.com/channel/UCX9y7I0jT4Q5pwYvNrcHI_Q 
