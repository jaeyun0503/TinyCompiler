"""
Module for handling output
"""

class OutStream:
    def __init__(self, *destinations):
        self.destinations = destinations
    
    def write(self, content: str) -> None:
        for destination in self.destinations:
            destination.write(content)
            destination.flush()
    
    def flush(self) -> None:
        for destination in self.destinations:
            destination.flush()

def open_files(file_paths: list, mode="w", encoding="utf-8") -> list:
    files = []
    for file_path in file_paths:
        files.append(open(file_path, mode=mode, encoding=encoding))
    return files

def close_files(*files) -> None:
    for file in files:
        file.close()
