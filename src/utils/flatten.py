def flatten_metrics(metrics: dict, parent_key='', sep='.') -> dict:
    """
    Recursively flatten nested dictionaries for MLflow logging.
    {'eval': {'accuracy': 0.8}} → {'eval.accuracy': 0.8}
    Ignores values that are not int or float.
    """
    items = {}
    for k, v in metrics.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_metrics(v, new_key, sep=sep))
        elif isinstance(v, (int, float)):
            items[new_key] = v
    return items
