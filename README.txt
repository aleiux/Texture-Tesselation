Requires the following python modules:
cv2, numpy, scipy

I've only tested on python 2.7; I'm not sure if open cv works with python3

arguments:
-path ... (string) the file path to find the image we want to tessellate
-gui ... (flag) whether or not we want to open the gui to define control points (must be specified if no.vtx file is present)
-L ... (flag) whether or not to perform lighting adjustments 
-aspect ... (float) aspect ratio. defaults to 1.0

example that generates output:

python main.py -path path/to/folder/image.jpg -L
