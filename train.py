from qcnn import QCNN
from qnn import QNN
from preprocessing import get_mnist, X_FLIP, Y_ROT, Y_POW

# visualization tools
import matplotlib.pyplot as plt

# Use less examples to prototype
NUM_EXAMPLES = 500
BATCH_SIZE = 8
EPOCHS = 3
IMG_DIM = 4
NUM_FILTERS = 8
PREPROCESSOR = X_FLIP

(x_train, y_train), (x_test, y_test) = get_mnist(IMG_DIM, PREPROCESSOR, NUM_EXAMPLES)

nn = QNN()
history = nn.model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                       verbose=1, validation_data=(x_test, y_test))

label = '{}x{}_{}.png'.format(IMG_DIM, IMG_DIM, PREPROCESSOR)

plt.plot(history.history['hinge_accuracy'], label='Training')
plt.plot(history.history['val_hinge_accuracy'], label='Testing')
plt.title('Hybrid CNN performance')
plt.xlabel('Epochs')
plt.legend()
plt.ylabel('Accuracy')
plt.savefig(label)
