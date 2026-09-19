import json, torch


def run(metrics):
    """
    Calculate and save the perplexity from the evaluation metrics.
    The perplexity is calculated as the exponential of the evaluation loss.
    The result is saved in a JSON file named 'metrics.json'.
    """
    ppl = torch.exp(torch.tensor(metrics["eval_loss"])).item()
    json.dump({"perplexity": ppl}, open("metrics.json", "w"))
    return ppl
