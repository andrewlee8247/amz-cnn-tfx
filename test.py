import requests
import json
from tensorflow.python.keras.preprocessing.image import img_to_array as img_to_array
from tensorflow.python.keras.preprocessing.image import load_img as load_img
image = img_to_array(load_img('./test_4.jpg', target_size=(128, 128))) / 255.
payload = {
  "instances": [{'input_image': image.tolist()}]
}
r = requests.post('http://localhost:8501/v1/models/PlanetModel:predict', json=payload)
print(json.loads(r.content))
