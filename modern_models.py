#!/usr/bin/env python3
"""
Modern EMG Classification Models
CNN-LSTM, Transformer, and Domain Adaptation implementations
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, LSTM, GRU, Dense, Dropout, BatchNormalization,
    MaxPooling1D, GlobalAveragePooling1D, Input, Attention,
    LayerNormalization, MultiHeadAttention, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModernEMGModels:
    def __init__(self, input_shape, n_classes, fs=1000):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.fs = fs
        self.models = {}
        
    def create_cnn_lstm_model(self, config=None):
        """Create CNN-LSTM hybrid model"""
        if config is None:
            config = {
                'cnn_filters': [64, 128, 256],
                'cnn_kernel_size': 3,
                'lstm_units': [128, 64],
                'dropout_rate': 0.5,
                'dense_units': 64
            }
        
        model = Sequential([
            # CNN layers for spatial feature extraction
            Conv1D(config['cnn_filters'][0], config['cnn_kernel_size'], 
                   activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv1D(config['cnn_filters'][0], config['cnn_kernel_size'], 
                   activation='relu'),
            MaxPooling1D(2),
            Dropout(config['dropout_rate']),
            
            Conv1D(config['cnn_filters'][1], config['cnn_kernel_size'], 
                   activation='relu'),
            BatchNormalization(),
            Conv1D(config['cnn_filters'][1], config['cnn_kernel_size'], 
                   activation='relu'),
            MaxPooling1D(2),
            Dropout(config['dropout_rate']),
            
            Conv1D(config['cnn_filters'][2], config['cnn_kernel_size'], 
                   activation='relu'),
            BatchNormalization(),
            Conv1D(config['cnn_filters'][2], config['cnn_kernel_size'], 
                   activation='relu'),
            MaxPooling1D(2),
            Dropout(config['dropout_rate']),
            
            # LSTM layers for temporal modeling
            LSTM(config['lstm_units'][0], return_sequences=True),
            Dropout(config['dropout_rate']),
            LSTM(config['lstm_units'][1], return_sequences=False),
            Dropout(config['dropout_rate']),
            
            # Classification head
            Dense(config['dense_units'], activation='relu'),
            Dropout(config['dropout_rate']),
            Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['cnn_lstm'] = model
        return model
    
    def create_transformer_model(self, config=None):
        """Create Transformer-based model"""
        if config is None:
            config = {
                'd_model': 128,
                'n_heads': 8,
                'n_layers': 4,
                'dff': 512,
                'dropout_rate': 0.1
            }
        
        inputs = Input(shape=self.input_shape)
        
        # Input embedding
        x = Dense(config['d_model'])(inputs)
        x = LayerNormalization()(x)
        
        # Positional encoding
        pos_encoding = self._positional_encoding(self.input_shape[0], config['d_model'])
        x = x + pos_encoding
        
        # Transformer blocks
        for _ in range(config['n_layers']):
            x = self._transformer_block(x, config)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Classification head
        x = Dense(64, activation='relu')(x)
        x = Dropout(config['dropout_rate'])(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['transformer'] = model
        return model
    
    def create_domain_adaptive_model(self, config=None):
        """Create domain adaptive model for cross-subject generalization"""
        if config is None:
            config = {
                'feature_dim': 128,
                'n_domains': 2,  # source and target
                'lambda_domain': 0.1
            }
        
        # Feature extractor
        feature_extractor = self._create_feature_extractor(config)
        
        # Classifier
        classifier = self._create_classifier(config)
        
        # Domain discriminator
        domain_discriminator = self._create_domain_discriminator(config)
        
        # Combined model
        inputs = Input(shape=self.input_shape)
        features = feature_extractor(inputs)
        class_output = classifier(features)
        domain_output = domain_discriminator(features)
        
        model = Model(inputs, [class_output, domain_output])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=['categorical_crossentropy', 'binary_crossentropy'],
            loss_weights=[1.0, config['lambda_domain']],
            metrics=['accuracy']
        )
        
        self.models['domain_adaptive'] = model
        return model
    
    def create_ensemble_model(self, models_list):
        """Create ensemble of multiple models"""
        inputs = Input(shape=self.input_shape)
        
        # Get predictions from each model
        predictions = []
        for model in models_list:
            pred = model(inputs)
            predictions.append(pred)
        
        # Average predictions
        ensemble_output = tf.keras.layers.Average()(predictions)
        
        model = Model(inputs, ensemble_output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['ensemble'] = model
        return model
    
    def _positional_encoding(self, max_len, d_model):
        """Create positional encoding for transformer"""
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def _transformer_block(self, x, config):
        """Create transformer block"""
        # Multi-head attention
        attn_output = MultiHeadAttention(
            num_heads=config['n_heads'],
            key_dim=config['d_model']
        )(x, x)
        attn_output = Dropout(config['dropout_rate'])(attn_output)
        x = LayerNormalization()(x + attn_output)
        
        # Feed forward network
        ffn = Dense(config['dff'], activation='relu')(x)
        ffn = Dense(config['d_model'])(ffn)
        ffn = Dropout(config['dropout_rate'])(ffn)
        x = LayerNormalization()(x + ffn)
        
        return x
    
    def _create_feature_extractor(self, config):
        """Create feature extractor for domain adaptation"""
        inputs = Input(shape=self.input_shape)
        
        x = Conv1D(64, 3, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.25)(x)
        
        x = Conv1D(128, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.25)(x)
        
        x = LSTM(128, return_sequences=False)(x)
        x = Dropout(0.5)(x)
        
        x = Dense(config['feature_dim'], activation='relu')(x)
        
        return Model(inputs, x)
    
    def _create_classifier(self, config):
        """Create classifier for domain adaptation"""
        inputs = Input(shape=(config['feature_dim'],))
        
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.5)(x)
        x = Dense(self.n_classes, activation='softmax')(x)
        
        return Model(inputs, x)
    
    def _create_domain_discriminator(self, config):
        """Create domain discriminator for domain adaptation"""
        inputs = Input(shape=(config['feature_dim'],))
        
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, x)
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val, 
                   epochs=100, batch_size=32, verbose=1):
        """Train specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate model performance"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred_classes == y_true_classes)
        
        # Classification report
        report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm, model_name, class_names=None):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.n_classes)]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

class ModelComparison:
    def __init__(self, models_dict):
        self.models = models_dict
        self.results = {}
    
    def compare_models(self, X_test, y_test):
        """Compare all models"""
        print("🔍 Comparing Model Performance...")
        
        for model_name, model in self.models.items():
            print(f"\n📊 Evaluating {model_name}...")
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            # Calculate metrics
            accuracy = np.mean(y_pred_classes == y_true_classes)
            f1_macro = tf.keras.metrics.F1Score(average='macro')(y_test, y_pred).numpy()
            f1_weighted = tf.keras.metrics.F1Score(average='weighted')(y_test, y_pred).numpy()
            
            self.results[model_name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'predictions': y_pred_classes
            }
            
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   ✅ F1 Macro: {f1_macro:.4f}")
            print(f"   ✅ F1 Weighted: {f1_weighted:.4f}")
        
        return self.results
    
    def plot_comparison(self):
        """Plot model comparison"""
        if not self.results:
            print("No results to plot. Run compare_models first.")
            return
        
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        f1_macros = [self.results[model]['f1_macro'] for model in models]
        f1_weighted = [self.results[model]['f1_weighted'] for model in models]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Accuracy comparison
        ax1.bar(models, accuracies, color='skyblue')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # F1 Macro comparison
        ax2.bar(models, f1_macros, color='lightgreen')
        ax2.set_title('Model F1 Macro Comparison')
        ax2.set_ylabel('F1 Macro')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # F1 Weighted comparison
        ax3.bar(models, f1_weighted, color='lightcoral')
        ax3.set_title('Model F1 Weighted Comparison')
        ax3.set_ylabel('F1 Weighted')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def main():
    """Demo modern models"""
    print("🚀 Modern EMG Classification Models Demo")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 16
    n_classes = 6
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    y_categorical = tf.keras.utils.to_categorical(y, n_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"📊 Data Shape: {X.shape}")
    print(f"📊 Classes: {n_classes}")
    
    # Initialize models
    model_builder = ModernEMGModels(
        input_shape=(n_features,),
        n_classes=n_classes
    )
    
    # Create models
    print("\n🔧 Creating Models...")
    cnn_lstm = model_builder.create_cnn_lstm_model()
    transformer = model_builder.create_transformer_model()
    
    print("   ✅ CNN-LSTM Model Created")
    print("   ✅ Transformer Model Created")
    
    # Train models
    print("\n🏋️ Training Models...")
    
    # Train CNN-LSTM
    print("   Training CNN-LSTM...")
    history_cnn_lstm = model_builder.train_model(
        'cnn_lstm', X_train, y_train, X_val, y_val, epochs=10, verbose=0
    )
    
    # Train Transformer
    print("   Training Transformer...")
    history_transformer = model_builder.train_model(
        'transformer', X_train, y_train, X_val, y_val, epochs=10, verbose=0
    )
    
    # Evaluate models
    print("\n📊 Evaluating Models...")
    
    # CNN-LSTM evaluation
    cnn_lstm_results = model_builder.evaluate_model('cnn_lstm', X_test, y_test)
    print(f"   CNN-LSTM Accuracy: {cnn_lstm_results['accuracy']:.4f}")
    
    # Transformer evaluation
    transformer_results = model_builder.evaluate_model('transformer', X_test, y_test)
    print(f"   Transformer Accuracy: {transformer_results['accuracy']:.4f}")
    
    # Compare models
    print("\n🔍 Comparing Models...")
    comparison = ModelComparison({
        'CNN-LSTM': model_builder.models['cnn_lstm'],
        'Transformer': model_builder.models['transformer']
    })
    
    comparison_results = comparison.compare_models(X_test, y_test)
    
    print("\n✅ Model comparison complete!")
    print("✅ Ready for real EMG data analysis")

if __name__ == "__main__":
    main()
