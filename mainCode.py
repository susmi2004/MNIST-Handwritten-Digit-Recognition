import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for visualization

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train /= 255
X_test /= 255

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Data visualization
plt.bar(np.arange(10), np.bincount(y_train.argmax(axis=1)))
plt.xlabel('Digits')
plt.ylabel('Frequency')
plt.title('Distribution of Digits in Training Data')
plt.show()

# Define the model architecture
def cnn_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the model
model = cnn_model()
print("✅ Starting training...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=1)

# Function to predict digit from an image
def predict_digit(img_path):
    img = image.load_img(img_path, color_mode="grayscale", target_size=(28, 28))
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    digit = np.argmax(result)
    return digit

# Test the model with a sample image
img_path = r'ExampleImages/testSample/img_3.jpg'  # Replace 'path_to_your_image.jpg' with the actual path to your image
predicted_digit = predict_digit(img_path)
print("Predicted Digit:", predicted_digit)

# Show the image
plt.imshow(image.load_img(img_path, color_mode="grayscale", target_size=(28, 28)), cmap='gray')
plt.show()

# 4.2 Heat Map
plt.figure(figsize=(10, 8))
sns.heatmap(X_train[0].reshape(28, 28), cmap='gray')
plt.title('Heat Map of First Image in Training Data')
plt.show()

# 7. Evaluating the model
# Evaluate the model
_, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
if test_accuracy >= 0.75:
    print("The model has sufficient accuracy to proceed.")
else:
    print("The model needs improvement or change in algorithm.")
