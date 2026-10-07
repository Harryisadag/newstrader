"""The local machine-learning engine: free, runs on your own computer, no API key needed.

Two layers:
  1. sentiment.py - FinBERT, a language model pre-trained on financial news, scores the wording about each
     company as positive / negative / neutral.
  2. model.py     - a model trained on YOUR Alpaca history (dataset.py + trainer.py): it learns how stocks
     actually moved (vs the S&P 500) in the hour after past headlines, and turns that into a probability.
engine.py ties both together and answers in the same format as the Claude engine, so every safety check,
risk rule and the validator still apply.
"""
