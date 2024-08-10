pip install ifnude
Example
Note: Importing ifnude for the first time will download a 139MB module to "/your/home/dir/.ifnude/", just once.

from ifnude import detect

# use mode="fast" for x3 speed with slightly lower accuracy
print(detect('/path/to/nsfw.png'))
Instead of a path, you can use a variable that contains an image loaded through cv2 (opencv) or PIL (pillow).

Output
[
  {'box': [164, 188, 246, 271], 'score': 0.8253238201141357, 'label': 'EXPOSED_BREAST_F'},
  {'box': [252, 190, 335, 270], 'score': 0.8235630989074707, 'label': 'EXPOSED_BREAST_F'}
]