import json
from pathlib import Path

results = json.loads(
    Path("models/training_results.json").read_text(encoding="utf-8")
)

print("Classes:", results["classes"])
print("Total images:", results["images"])
print("Training images:", results["train_images"])
print("Validation images:", results["validation_images"])
print("Best validation accuracy:",
      f'{results["best_validation_accuracy"]:.4%}')
print("Best validation loss:",
      f'{results["best_validation_loss"]:.4f}')

print("\nTraining script settings:")
script = Path("src/train_character_model.py").read_text(encoding="utf-8")

for line in script.splitlines():
    if any(term in line.lower() for term in [
        "epochs=", "batch_size=", "random_state",
        "train_test_split", "rescale", "rotation",
        "translation", "zoom", "model.fit"
    ]):
        print(line.strip())
