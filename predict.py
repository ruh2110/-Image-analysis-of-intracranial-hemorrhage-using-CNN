import tensorflow as tf
import numpy as np


model =tf.keras.models.load_model('D:\\Head-CT-hemorrhage-detection-master\\Code\\NAKSH.model.keras')


#img_path = "D:\\Head-CT-hemorrhage-detection-master\\basal_ganglia_hemorrhage4567_f.png"
img_path = "D:\\Head-CT-hemorrhage-detection-master\\Normal-CT-head-2Age-30-40.png"

img =tf.keras.utils.load_img(
    img_path, target_size=(140,140),color_mode="grayscale")

img_array =tf.keras.utils.img_to_array(img)

img_array =tf.expand_dims(img_array, 0)


prediction =model.predict(img_array)

print(prediction)

if (prediction[0][0] == 1):
    print("Yes")
else:
    print("No")