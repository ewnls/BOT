"""
Modèles ML pour le trading
LSTM, Transformer, XGBoost avec optimisations avancées
"""
import numpy as np
import os


class LSTMModel:
    """Modèle LSTM avec optimisations avancées"""
    
    def __init__(self, lookback=60, features=30):
        self.lookback = lookback
        self.features = features
        self.model = None
    
    def prepare_sequences(self, X, y):
        """Convertit les données en séquences pour LSTM"""
        n_samples = len(X) - self.lookback + 1
        X_seq = np.zeros((n_samples, self.lookback, X.shape[1]))
        y_seq = np.zeros((n_samples, y.shape[1]))
        
        for i in range(n_samples):
            X_seq[i] = X[i:i + self.lookback]
            y_seq[i] = y[i + self.lookback - 1]
        
        return X_seq, y_seq
    
    def build_model(self, output_dim=4):
        """Construit l'architecture LSTM optimisée"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2
        
        from tensorflow.keras.layers import Input

        inputs = Input(shape=(self.lookback, self.features))
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.2)(x)
        outputs = Dense(output_dim)(x)

        from tensorflow.keras import Model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Optimizer Adam avec bons paramètres
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print(f"✅ Modèle LSTM construit: {self.model.count_params():,} paramètres")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
        """Entraîne avec callbacks avancés (early stopping, LR adaptatif)"""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1 if verbose > 0 else 0
        )
        
        # Learning rate adaptatif
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,           # Divise LR par 2
            patience=7,           # Attend 7 epochs sans amélioration
            min_lr=1e-7,          # LR minimum
            verbose=1 if verbose > 0 else 0
        )
        
        # Checkpoint (sauvegarde temporaire du meilleur modèle)
        os.makedirs('models/temp', exist_ok=True)
        checkpoint = ModelCheckpoint(
            'models/temp/best_checkpoint.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        # Entraînement
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=verbose
        )
        
        print(f"✅ Entraînement terminé après {len(history.history['loss'])} epochs")
        print(f"   Train loss: {history.history['loss'][-1]:.4f}")
        print(f"   Val loss: {history.history['val_loss'][-1]:.4f}")
        
        return history
    
    def predict(self, X):
        """Prédictions"""
        return self.model.predict(X, verbose=0)
    
    def save(self, filepath):
        """Sauvegarde le modèle"""
        self.model.save(filepath + '.keras')
    
    def load(self, filepath):
        """Charge le modèle"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath + '.keras')


class TransformerModel:
    """Modèle Transformer pour séries temporelles"""
    
    def __init__(self, lookback=60, features=30):
        self.lookback = lookback
        self.features = features
        self.model = None
    
    def prepare_sequences(self, X, y):
        """Convertit les données en séquences"""
        n_samples = len(X) - self.lookback + 1
        X_seq = np.zeros((n_samples, self.lookback, X.shape[1]))
        y_seq = np.zeros((n_samples, y.shape[1]))
        
        for i in range(n_samples):
            X_seq[i] = X[i:i + self.lookback]
            y_seq[i] = y[i + self.lookback - 1]
        
        return X_seq, y_seq
    
    def build_model(self, output_dim=4):
        """Construit un Transformer simplifié"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Dense, Dropout, LayerNormalization, 
            MultiHeadAttention, GlobalAveragePooling1D, Input
        )
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras import Model
        
        # Input
        inputs = Input(shape=(self.lookback, self.features))
        
        # Multi-Head Attention
        attention_output = MultiHeadAttention(
            num_heads=4,
            key_dim=32
        )(inputs, inputs)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(attention_output + inputs)
        
        # Feed Forward
        ff = Dense(128, activation='relu')(x)
        ff = Dropout(0.2)(ff)
        ff = Dense(self.features)(ff)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(x + ff)
        
        # Pooling et output
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(output_dim)(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Optimizer
        optimizer = Adam(
            learning_rate=0.0005,  # LR plus faible pour Transformer
            beta_1=0.9,
            beta_2=0.999
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print(f"✅ Modèle Transformer construit: {self.model.count_params():,} paramètres")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
        """Entraîne avec callbacks avancés"""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1 if verbose > 0 else 0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1 if verbose > 0 else 0
        )
        
        os.makedirs('models/temp', exist_ok=True)
        checkpoint = ModelCheckpoint(
            'models/temp/best_checkpoint.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=verbose
        )
        
        print(f"✅ Entraînement terminé après {len(history.history['loss'])} epochs")
        print(f"   Train loss: {history.history['loss'][-1]:.4f}")
        print(f"   Val loss: {history.history['val_loss'][-1]:.4f}")
        
        return history
    
    def predict(self, X):
        """Prédictions"""
        return self.model.predict(X, verbose=0)
    
    def save(self, filepath):
        """Sauvegarde le modèle"""
        self.model.save(filepath + '.keras')
    
    def load(self, filepath):
        """Charge le modèle"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath + '.keras')


class XGBoostModel:
    """Modèle XGBoost pour trading"""
    
    def __init__(self, lookback=60, features=30):
        self.lookback = lookback
        self.features = features
        self.models = []  # Liste de modèles (un par output)
    
    def build_model(self):
        """XGBoost n'a pas besoin de build explicite"""
        pass
    
    def train(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None, extra_params=None):
        """Entraîne 4 modèles XGBoost (un par target)"""
        import xgboost as xgb
        
        print(f"🚀 Entraînement XGBoost...")
        
        # Paramètres par défaut
        base_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 200,
            'early_stopping_rounds': 20,
            'random_state': 42
        }

        # Si des hyperparamètres Optuna sont fournis, on les injecte
        if extra_params is not None:
            print(f"🔧 Application des hyperparamètres personnalisés: {extra_params}")
            base_params.update({
                'max_depth': extra_params.get('max_depth', base_params['max_depth']),
                'learning_rate': extra_params.get('learning_rate', base_params['learning_rate']),
                'n_estimators': extra_params.get('n_estimators', base_params['n_estimators']),
                'subsample': extra_params.get('subsample', base_params['subsample']),
                'colsample_bytree': extra_params.get('colsample_bytree', base_params['colsample_bytree']),
                'min_child_weight': extra_params.get('min_child_weight', base_params['min_child_weight']),
                'gamma': extra_params.get('gamma', base_params['gamma']),
            })
        
        self.models = []
        
        # Entraîne un modèle par output
        for i in range(y_train.shape[1]):
            print(f"   Training output {i+1}/4...")
            
            model = xgb.XGBRegressor(**base_params)
            
            model.fit(
                X_train, y_train[:, i],
                eval_set=[(X_val, y_val[:, i])],
                verbose=False
            )
            
            self.models.append(model)
        
        print(f"✅ XGBoost entraîné")
    
    def predict(self, X):
        """Prédictions pour les 4 outputs"""
        predictions = np.zeros((len(X), len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        return predictions
    
    def save(self, filepath):
        """Sauvegarde les modèles"""
        import pickle
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath + '_xgb.pkl', 'wb') as f:
            pickle.dump(self.models, f)
    
    def load(self, filepath):
        """Charge les modèles"""
        import pickle
        
        with open(filepath + '_xgb.pkl', 'rb') as f:
            self.models = pickle.load(f)


if __name__ == '__main__':
    print("ML Models prêts!")
    print("  - LSTMModel (avec ReduceLROnPlateau)")
    print("  - TransformerModel (avec attention)")
    print("  - XGBoostModel (gradient boosting)")
