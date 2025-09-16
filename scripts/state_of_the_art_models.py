#!/usr/bin/env python3
"""
State-of-the-Art EMG Classification Models (2024-2025)
Based on latest research: Vision Transformers, Attention Mechanisms, Domain Adaptation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, LSTM, GRU, MultiHeadAttention, 
    LayerNormalization, Dropout, GlobalAveragePooling1D,
    Add, Embedding, TimeDistributed, BatchNormalization,
    Attention, Concatenate, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class StateOfTheArtEMGModels:
    def __init__(self, input_shape, n_classes, fs=1000):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.fs = fs
        self.models = {}
        
    def create_vision_transformer_emg(self, config=None):
        """Vision Transformer adapted for EMG signals (2024 SOTA)"""
        if config is None:
            config = {
                'patch_size': 16,
                'embed_dim': 128,
                'num_heads': 8,
                'num_layers': 6,
                'mlp_dim': 512,
                'dropout_rate': 0.1
            }
        
        # Input
        inputs = Input(shape=self.input_shape)
        
        # Patch embedding
        patches = self._create_patches(inputs, config['patch_size'])
        patches = Dense(config['embed_dim'])(patches)
        
        # Add positional encoding
        positions = tf.range(start=0, limit=tf.shape(patches)[1], delta=1)
        positions = tf.expand_dims(positions, 0)
        pos_encoding = self._positional_encoding(positions, config['embed_dim'])
        patches = patches + pos_encoding
        
        # Add class token
        class_token = tf.Variable(tf.random.normal([1, 1, config['embed_dim']]))
        class_token = tf.tile(class_token, [tf.shape(patches)[0], 1, 1])
        x = tf.concat([class_token, patches], axis=1)
        
        # Transformer blocks
        for _ in range(config['num_layers']):
            x = self._transformer_block(x, config)
        
        # Classification head
        x = LayerNormalization()(x)
        x = x[:, 0]  # Class token
        x = Dense(config['mlp_dim'], activation='gelu')(x)
        x = Dropout(config['dropout_rate'])(x)
        x = Dense(config['mlp_dim']//2, activation='gelu')(x)
        x = Dropout(config['dropout_rate'])(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['vision_transformer'] = model
        return model
    
    def create_attention_cnn_lstm(self, config=None):
        """CNN-LSTM with Multi-Head Attention (2024 SOTA)"""
        if config is None:
            config = {
                'cnn_filters': [64, 128, 256, 512],
                'lstm_units': [256, 128],
                'attention_heads': 8,
                'dropout_rate': 0.3
            }
        
        inputs = Input(shape=self.input_shape)
        
        # CNN feature extraction
        x = inputs
        for filters in config['cnn_filters']:
            x = Conv1D(filters, 3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = Dropout(config['dropout_rate'])(x)
            x = Conv1D(filters, 3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling1D(2)(x)
        
        # LSTM layers
        x = LSTM(config['lstm_units'][0], return_sequences=True)(x)
        x = Dropout(config['dropout_rate'])(x)
        x = LSTM(config['lstm_units'][1], return_sequences=True)(x)
        x = Dropout(config['dropout_rate'])(x)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=config['attention_heads'],
            key_dim=x.shape[-1]//config['attention_heads']
        )(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        
        # Global attention pooling
        attention_weights = Dense(1, activation='tanh')(x)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        x = tf.reduce_sum(x * attention_weights, axis=1)
        
        # Classification head
        x = Dense(128, activation='relu')(x)
        x = Dropout(config['dropout_rate'])(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['attention_cnn_lstm'] = model
        return model
    
    def create_domain_adaptive_transformer(self, config=None):
        """Domain Adaptive Transformer for Cross-Subject Generalization (2024 SOTA)"""
        if config is None:
            config = {
                'embed_dim': 128,
                'num_heads': 8,
                'num_layers': 4,
                'domain_lambda': 0.1
            }
        
        # Shared feature extractor
        feature_extractor = self._create_domain_feature_extractor(config)
        
        # Classifier
        classifier = self._create_domain_classifier(config)
        
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
            loss_weights=[1.0, config['domain_lambda']],
            metrics=['accuracy']
        )
        
        self.models['domain_adaptive_transformer'] = model
        return model
    
    def create_hybrid_transformer_cnn(self, config=None):
        """Hybrid Transformer-CNN Architecture (2024 SOTA)"""
        if config is None:
            config = {
                'cnn_filters': [32, 64, 128],
                'transformer_layers': 4,
                'num_heads': 8,
                'embed_dim': 128
            }
        
        inputs = Input(shape=self.input_shape)
        
        # CNN backbone
        x = inputs
        for filters in config['cnn_filters']:
            x = Conv1D(filters, 3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling1D(2)(x)
        
        # Reshape for transformer
        x = Dense(config['embed_dim'])(x)
        
        # Transformer layers
        for _ in range(config['transformer_layers']):
            x = self._transformer_block(x, {
                'embed_dim': config['embed_dim'],
                'num_heads': config['num_heads'],
                'mlp_dim': config['embed_dim'] * 4,
                'dropout_rate': 0.1
            })
        
        # Global pooling and classification
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['hybrid_transformer_cnn'] = model
        return model
    
    def create_ensemble_sota(self, models_list):
        """Ensemble of SOTA models with attention weighting"""
        inputs = Input(shape=self.input_shape)
        
        # Get predictions from each model
        predictions = []
        for model in models_list:
            pred = model(inputs)
            predictions.append(pred)
        
        # Attention-weighted ensemble
        attention_weights = Dense(len(models_list), activation='softmax')(inputs)
        attention_weights = tf.expand_dims(attention_weights, -1)
        
        weighted_predictions = []
        for i, pred in enumerate(predictions):
            weight = attention_weights[:, i:i+1, :]
            weighted_predictions.append(pred * weight)
        
        ensemble_output = tf.keras.layers.Add()(weighted_predictions)
        
        model = Model(inputs, ensemble_output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['ensemble_sota'] = model
        return model
    
    def _create_patches(self, x, patch_size):
        """Create patches from EMG signal"""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        channels = x.shape[2]
        
        # Reshape to patches
        patches = tf.reshape(x, [batch_size, seq_len // patch_size, patch_size * channels])
        return patches
    
    def _positional_encoding(self, positions, d_model):
        """Create positional encoding"""
        angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        return tf.cast(angle_rads, dtype=tf.float32)
    
    def _transformer_block(self, x, config):
        """Transformer block with self-attention"""
        # Multi-head attention
        attn_output = MultiHeadAttention(
            num_heads=config['num_heads'],
            key_dim=config['embed_dim']//config['num_heads']
        )(x, x)
        attn_output = Dropout(config['dropout_rate'])(attn_output)
        x = LayerNormalization()(x + attn_output)
        
        # Feed forward
        ffn = Dense(config['mlp_dim'], activation='gelu')(x)
        ffn = Dropout(config['dropout_rate'])(ffn)
        ffn = Dense(config['embed_dim'])(ffn)
        ffn = Dropout(config['dropout_rate'])(ffn)
        x = LayerNormalization()(x + ffn)
        
        return x
    
    def _create_domain_feature_extractor(self, config):
        """Domain adaptive feature extractor"""
        inputs = Input(shape=self.input_shape)
        
        x = Conv1D(64, 3, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        
        x = Conv1D(128, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        
        x = Dense(config['embed_dim'])(x)
        
        return Model(inputs, x)
    
    def _create_domain_classifier(self, config):
        """Domain adaptive classifier"""
        inputs = Input(shape=(None, config['embed_dim']))
        
        x = GlobalAveragePooling1D()(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.n_classes, activation='softmax')(x)
        
        return Model(inputs, x)
    
    def _create_domain_discriminator(self, config):
        """Domain discriminator for adversarial training"""
        inputs = Input(shape=(None, config['embed_dim']))
        
        x = GlobalAveragePooling1D()(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, x)

def main():
    """Demo SOTA models"""
    print("🚀 State-of-the-Art EMG Models Demo (2024-2025)")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 200  # Time series length
    n_classes = 6
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    y_categorical = tf.keras.utils.to_categorical(y, n_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )
    
    print(f"📊 Data Shape: {X.shape}")
    print(f"📊 Classes: {n_classes}")
    
    # Initialize SOTA models
    sota_models = StateOfTheArtEMGModels(
        input_shape=(n_features,),
        n_classes=n_classes
    )
    
    # Create models
    print("\n🔧 Creating State-of-the-Art Models...")
    
    # Vision Transformer
    print("   Creating Vision Transformer...")
    vit_model = sota_models.create_vision_transformer_emg()
    
    # Attention CNN-LSTM
    print("   Creating Attention CNN-LSTM...")
    att_cnn_lstm = sota_models.create_attention_cnn_lstm()
    
    # Domain Adaptive Transformer
    print("   Creating Domain Adaptive Transformer...")
    domain_transformer = sota_models.create_domain_adaptive_transformer()
    
    # Hybrid Transformer-CNN
    print("   Creating Hybrid Transformer-CNN...")
    hybrid_model = sota_models.create_hybrid_transformer_cnn()
    
    print("   ✅ All SOTA models created!")
    
    # Train and evaluate
    print("\n🏋️ Training Models...")
    
    for model_name, model in sota_models.models.items():
        print(f"   Training {model_name}...")
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"     ✅ {model_name}: Accuracy = {test_acc:.4f}")
    
    print("\n🎉 State-of-the-Art models ready!")
    print("✅ Ready for real EMG data analysis")

if __name__ == "__main__":
    main()
