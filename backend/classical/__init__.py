"""
backend/classical/__init__.py
=============================
Classical machine learning baseline and benchmark models for QML Benchmark Studio.
"""

from backend.classical.dummy_baseline import DummyBaseline
from backend.classical.mlp import MLPClassifier
from backend.classical.random_forest import RandomForestClassifier
from backend.classical.svm import SVMClassifier

__all__ = [
    "DummyBaseline",
    "MLPClassifier",
    "RandomForestClassifier",
    "SVMClassifier",
]
