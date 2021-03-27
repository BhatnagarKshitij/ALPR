import cv2 #For Image Processing Techniques
import numpy as np #For dealing with multi dimension arrays
import matplotlib.pyplot as plt #To display and Visualize data
from local_utils import detect_lp #LOCAL PYTHON FILE TO INTERACT WITH "Wpod-NET" model
from os.path import splitext, basename #Accessing File System
from keras.models import model_from_json #Load Model in JSON Format
import glob #Searching files through patterns

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)
        
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)


def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path) #Read Images
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Convert BGR TO RGB
    img = img / 255 #Normalize image
    if resize:
        img = cv2.resize(img, (224,224))
    return img


image_paths = glob.glob("Plate_examples\*.jpg")
print("Found %i images..."%(len(image_paths)))
# Visualize data in subplot 
fig = plt.figure(figsize=(12,8))
cols = 5
rows = 5
fig_list = []
for i in range(len(image_paths)):
    fig_list.append(fig.add_subplot(rows,cols,i+1))
    title = splitext(basename(image_paths[i]))[0]
    fig_list[-1].set_title(title)
    img = preprocess_image(image_paths[i],True)
    plt.axis(False)
    plt.imshow(img)

plt.tight_layout(True)
plt.show()