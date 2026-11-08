"""Inferensi KNN dengan checkpoint tensor PyTorch (.pth) hasil ekspor Colab.

Ini KNN non-parametrik, bukan neural network. Berkas .pth berisi vektor
training terstandardisasi, label training, parameter StandardScaler, dan k.
"""
from pathlib import Path
from typing import Sequence

import torch


class IrisKNN:
    REQUIRED = {
        'train_features_scaled', 'train_labels', 'scaler_mean', 'scaler_scale',
        'k', 'p', 'class_names', 'feature_names'
    }

    def __init__(self, checkpoint_path: str | Path):
        checkpoint_path = Path(checkpoint_path)
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        absent = self.REQUIRED.difference(state)
        if absent:
            raise ValueError(f'Checkpoint .pth tidak lengkap: {sorted(absent)}')
        self.X = state['train_features_scaled'].to(torch.float64)
        self.y = state['train_labels'].to(torch.long)
        self.mean = state['scaler_mean'].to(torch.float64)
        self.scale = state['scaler_scale'].to(torch.float64)
        self.k = int(state['k'])
        self.p = int(state['p'])
        self.class_names = list(state['class_names'])
        self.feature_names = list(state['feature_names'])
        if (self.X.ndim != 2 or self.X.size(1) != 4 or self.y.numel() != self.X.size(0)
                or self.mean.numel() != 4 or self.scale.numel() != 4
                or not 1 <= self.k <= self.X.size(0) or self.p not in (1, 2)
                or len(self.class_names) != 3):
            raise ValueError('Isi checkpoint tidak sesuai spesifikasi Iris KNN.')

    def predict_one(self, measurements: Sequence[float]) -> tuple[str, dict[str, float]]:
        """Keluarkan prediksi dan proporsi suara 3 kelas (bukan probabilitas terkalibrasi)."""
        values = torch.as_tensor(measurements, dtype=torch.float64)
        if values.shape != (4,) or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError('Masukkan empat angka ukuran bunga yang valid dan tidak negatif.')
        query = (values - self.mean) / self.scale
        distances = torch.linalg.vector_norm(self.X - query, ord=self.p, dim=1)
        # KNN voting uniform; saat jumlah suara seri, pilih kelas dengan indeks terkecil.
        nearest = torch.argsort(distances, stable=True)[:self.k]
        counts = torch.bincount(self.y[nearest], minlength=len(self.class_names))
        result = int(torch.argmax(counts))
        votes = {name: float(counts[i].item()) / self.k for i, name in enumerate(self.class_names)}
        return self.class_names[result], votes
