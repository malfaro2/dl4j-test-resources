from keras.models import load_model
import numpy as np

np.random.seed(42)

model = load_model("cnn_batch_norm_medium.h5")
input = np.random.random((5, 48, 48, 1))

output = model.predict(input)

assert abs(0.49185085  - output[0][0]) < 0.000001
assert abs(0.4922847   - output[1][0]) < 0.000001
assert abs(0.49167     - output[2][0]) < 0.000001
assert abs(0.49162313  - output[3][0]) < 0.000001
assert abs(0.491363    - output[4][0]) < 0.000001

np.save(arr=input, file="input.npy")
np.save(arr=output, file="predictions.npy")
