import os

folders = [
    "src",
    "data",
    "data/raw",
    "data/processed",
    "models",
    "webapp",
    "webapp/templates",
    "webapp/static"
]

files = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "src/__init__.py",
    "src/train_model.py",
    "src/predict.py",
    "src/utils.py",
    "webapp/app.py",
    "webapp/templates/index.html",
    "webapp/static/style.css"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file in files:
    open(file, "w").close()

print("🎉 Project structure created successfully!")
