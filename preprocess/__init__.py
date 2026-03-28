def main():
    "Entry point for CLI"
    from .preprocess import main as _main
    return _main()