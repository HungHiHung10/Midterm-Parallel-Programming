# Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Execute and save every notebook output.
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root = Path(__file__).resolve().parents[1]
path = root/'Group22_Midterm.ipynb'
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(root)}})
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print('Executed notebook:', path)
