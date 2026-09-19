def run(in_path, out_path, cfg=None):
    """
    Clean the raw Hawaiian text based on the provided configuration.
    """
    if cfg is None:
        # Raise an error if no configuration is provided
        raise ValueError("Configuration (cfg) must be provided for cleaning.")

    # Grab corpus_data_clean
    corpus_data_clean = cfg.get("corpus_data_clean");
    if corpus_data_clean is None:
        raise ValueError("Configuration must specify 'corpus_data_clean'.")

    if "remove_numbers" in corpus_data_clean:
        import re
        # Read the input file
        text = open(in_path, "r", encoding="utf-8").read()
        # Remove a that start with a number and nothing else
        text = re.sub(r'^\d.*\n?', '', text, flags=re.MULTILINE)
        # Write the cleaned text back to the output file
        open(out_path, "w", encoding="utf-8").write(text)
        return

    open(out_path, "w").write(open(in_path).read())
