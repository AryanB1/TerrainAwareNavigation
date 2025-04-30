#!/usr/bin/env python3
# train_terrain_model.py
# Script to train a TensorFlow model for terrain classification
# and convert it to TensorFlow Lite format

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# Define the terrain types matching the C++ enum
TERRAIN_TYPES = {
    0: "UNKNOWN",
    1: "FLAT",
    2: "ROUGH",
    3: "STEEP",
    4: "SLIPPERY",
    5: "SOFT",
    6: "OBSTACLE"
}

def load_dataset(dataset_path):
    """Load terrain dataset from CSV file.
    
    Expected format:
    roughness,slope,hardness,friction,visual_feature1,...,visual_featureN,terrain_class
    """
    print(f"Loading dataset from {dataset_path}")
    
    try:
        # Load data from CSV file
        data = pd.read_csv(dataset_path)
        
        # Extract features and labels
        X = data.iloc[:, :-1].values  # All columns except the last one
        y = data.iloc[:, -1].values   # Last column contains terrain class
        
        print(f"Dataset loaded: {len(X)} samples with {X.shape[1]} features per sample")
        
        return X, y
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None

def create_model(input_shape, num_classes):
    """Create a neural network model for terrain classification."""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X, y, epochs=50, batch_size=32, validation_split=0.2):
    """Train the terrain classification model."""
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, 
                                                      random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Save the scaler for later use
    np.save('feature_scaler.npy', [scaler.mean_, scaler.scale_])
    
    # Create model
    model = create_model(X_train.shape[1], len(np.unique(y)))
    model.summary()
    
    # Create callbacks
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tensorboard_callback, early_stopping]
    )
    
    # Evaluate model on validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")
    
    # Generate predictions for validation set
    y_pred = np.argmax(model.predict(X_val), axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=[TERRAIN_TYPES[i] for i in range(len(TERRAIN_TYPES))]))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(TERRAIN_TYPES))
    plt.xticks(tick_marks, [TERRAIN_TYPES[i] for i in range(len(TERRAIN_TYPES))], rotation=45)
    plt.yticks(tick_marks, [TERRAIN_TYPES[i] for i in range(len(TERRAIN_TYPES))])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return model, scaler

def convert_to_tflite(model, output_path):
    """Convert Keras model to TensorFlow Lite format."""
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimize for size and latency
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model to disk
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved to {output_path}")
    
    # Create a quantized model for better performance on embedded devices
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model_quant = converter.convert()
    
    # Save the quantized model
    quant_output_path = output_path.replace('.tflite', '_quantized.tflite')
    with open(quant_output_path, 'wb') as f:
        f.write(tflite_model_quant)
    
    print(f"Quantized TensorFlow Lite model saved to {quant_output_path}")

def generate_sample_data(output_file, num_samples=1000):
    """Generate synthetic data for testing the model training pipeline."""
    print(f"Generating {num_samples} synthetic data samples...")
    
    # Features: roughness, slope, hardness, friction, 15 visual features
    num_features = 19
    X = np.zeros((num_samples, num_features))
    y = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        # Assign random terrain class (1-6, excluding UNKNOWN)
        terrain_class = np.random.randint(1, 7)
        y[i] = terrain_class
        
        # Generate features based on terrain class
        if terrain_class == 1:  # FLAT
            X[i, 0] = np.random.uniform(0, 0.3)  # roughness
            X[i, 1] = np.random.uniform(0, 0.2)  # slope
            X[i, 2] = np.random.uniform(0.5, 1.0)  # hardness
            X[i, 3] = np.random.uniform(0.5, 0.9)  # friction
        elif terrain_class == 2:  # ROUGH
            X[i, 0] = np.random.uniform(0.5, 1.0)  # roughness
            X[i, 1] = np.random.uniform(0.2, 0.6)  # slope
            X[i, 2] = np.random.uniform(0.5, 1.0)  # hardness
            X[i, 3] = np.random.uniform(0.6, 0.9)  # friction
        elif terrain_class == 3:  # STEEP
            X[i, 0] = np.random.uniform(0.2, 0.7)  # roughness
            X[i, 1] = np.random.uniform(0.7, 1.0)  # slope
            X[i, 2] = np.random.uniform(0.5, 1.0)  # hardness
            X[i, 3] = np.random.uniform(0.3, 0.7)  # friction
        elif terrain_class == 4:  # SLIPPERY
            X[i, 0] = np.random.uniform(0, 0.4)  # roughness
            X[i, 1] = np.random.uniform(0, 0.6)  # slope
            X[i, 2] = np.random.uniform(0.2, 0.8)  # hardness
            X[i, 3] = np.random.uniform(0, 0.4)  # friction
        elif terrain_class == 5:  # SOFT
            X[i, 0] = np.random.uniform(0.1, 0.5)  # roughness
            X[i, 1] = np.random.uniform(0, 0.4)  # slope
            X[i, 2] = np.random.uniform(0, 0.5)  # hardness
            X[i, 3] = np.random.uniform(0.4, 0.8)  # friction
        elif terrain_class == 6:  # OBSTACLE
            X[i, 0] = np.random.uniform(0.8, 1.0)  # roughness
            X[i, 1] = np.random.uniform(0.5, 1.0)  # slope
            X[i, 2] = np.random.uniform(0.7, 1.0)  # hardness
            X[i, 3] = np.random.uniform(0.5, 0.9)  # friction
        
        # Generate random visual features
        X[i, 4:] = np.random.uniform(0, 1, 15)
        
        # Add some class-specific visual features (e.g., color ratios)
        if terrain_class == 5:  # SOFT (vegetation)
            X[i, 16] = np.random.uniform(0.5, 1.0)  # Green ratio (vegetation)
        elif terrain_class == 4:  # SLIPPERY
            X[i, 17] = np.random.uniform(0.5, 1.0)  # Blue ratio (water)
        elif terrain_class == 2:  # ROUGH
            X[i, 18] = np.random.uniform(0.5, 1.0)  # Yellow/brown ratio (dirt)
    
    # Create DataFrame with headers
    headers = ['roughness', 'slope', 'hardness', 'friction'] + [f'visual_feature_{i}' for i in range(1, 16)] + ['terrain_class']
    df = pd.DataFrame(np.column_stack((X, y)), columns=headers)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Synthetic data saved to {output_file}")
    
    return X, y

def main():
    parser = argparse.ArgumentParser(description='Train a terrain classification model.')
    parser.add_argument('--dataset', type=str, default=None, 
                        help='Path to the dataset CSV file')
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate synthetic data for testing')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--output', type=str, default='../models/terrain_classifier.tflite',
                        help='Output path for the TensorFlow Lite model')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Generate synthetic data if requested
    if args.generate_data:
        synthetic_data_path = 'synthetic_terrain_data.csv'
        X, y = generate_sample_data(synthetic_data_path)
        if args.dataset is None:
            args.dataset = synthetic_data_path
    
    # Load dataset
    if args.dataset:
        X, y = load_dataset(args.dataset)
        if X is None or y is None:
            print("Failed to load dataset. Exiting.")
            return
            
        # Train model
        model, scaler = train_model(X, y, epochs=args.epochs, batch_size=args.batch_size)
        
        # Save Keras model
        keras_model_path = args.output.replace('.tflite', '.keras')
        model.save(keras_model_path)
        print(f"Keras model saved to {keras_model_path}")
        
        # Convert to TensorFlow Lite
        convert_to_tflite(model, args.output)
    else:
        print("No dataset provided. Use --dataset to specify a CSV file or --generate-data to create synthetic data.")

if __name__ == "__main__":
    main()