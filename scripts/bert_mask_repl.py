"""
Interactive REPL for testing Hawaiian Masked Language Model (MLM).
Loads a locally fine-tuned model (`quick-haw-mlm`) and allows users to input sentences
with [MASK] tokens, returning top predictions.
"""

from transformers import pipeline

def run():
    try:
        fm = pipeline(
            "fill-mask",
            model="quick-haw-mlm",
            tokenizer="quick-haw-mlm",
            top_k=5,
            device_map="auto",
        )
    except Exception as e:
        print(f"🚨 Failed to load model from quick-haw-mlm: {e}")
        return

    print("=" * 72)
    print("🧠 HAWAIIAN MASKED LANGUAGE MODEL — INTERACTIVE")
    print("💡 Type a sentence containing [MASK] to view top predictions.")
    print("🔚 Type /exit or press Ctrl+C to quit.")
    print("=" * 72)

    while True:
        try:
            sent = input("\n» ")
            if sent.strip().lower() == "/exit":
                print("👋 Exiting...")
                break
            results = fm(sent)
            for out in results:
                print(f" {out['token_str']:15} (p={out['score']:.3f})")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except EOFError:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    run()
