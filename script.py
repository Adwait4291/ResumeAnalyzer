import os

# Define the base directory (optional, defaults to current directory)
# base_dir = "my_streamlit_app"
# os.makedirs(base_dir, exist_ok=True) # Create base directory if needed
# os.chdir(base_dir) # Change into the base directory

# List of directories to create
directories = [
    "assets",
    "src",
    "src/nlp_processor",
    "src/utils",
    "data",
    "tests"
]

# List of files to create (relative paths)
files = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "app.py",
    "src/__init__.py",
    "src/nlp_processor/__init__.py",
    "src/nlp_processor/models.py",
    "src/nlp_processor/similarity.py",
    "src/utils/__init__.py",
    "src/utils/helpers.py",
    "tests/__init__.py",
    "tests/test_similarity.py"
]

# Create directories
print("Creating directories...")
for directory in directories:
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}/")
    except OSError as e:
        print(f"Error creating directory {directory}: {e}")

# Create files
print("\nCreating files...")
for file_path in files:
    try:
        # Check if the directory for the file exists, create if not
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
             os.makedirs(dir_name, exist_ok=True)
             print(f"  Created intermediate directory: {dir_name}/")

        # Create the file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                pass  # Create an empty file
            print(f"  Created: {file_path}")
        else:
            print(f"  Skipped (already exists): {file_path}")
    except OSError as e:
        print(f"Error creating file {file_path}: {e}")

print("\nDirectory structure created successfully!")
