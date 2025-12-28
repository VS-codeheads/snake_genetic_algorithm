# src/agents/__init__.py
"""Agents module."""

from src.agents.dqn_feature import DQNFeatureAgent
from src.agents.random_agent import RandomAgent

__all__ = ['DQNFeatureAgent', 'RandomAgent']