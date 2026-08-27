import csv
from pathlib import Path


class Loader:
    def __init__(self, path: str, has_header: bool = True):
        self.path = Path(path)
        self.has_header = has_header
        self.headers: list[str] = []
        self.rows: list[list[str]] = []
        self._load()

    def _load(self):
        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            if self.has_header:
                self.headers = next(reader)
            else:
                # Infer header from first row
                first_row = next(reader)
                self.headers = [f"feat_{i}" for i in range(len(first_row) - 1)] + [
                    "target"
                ]
                self.rows.append(first_row)
            for row in reader:
                if row and any(cell.strip() for cell in row):
                    # Skip rows with missing values ('?')
                    if "?" not in row:
                        self.rows.append(row)

    @property
    def feature_count(self) -> int:
        return len(self.headers) - 1 if len(self.headers) > 1 else len(self.headers)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return f"Loader(features={self.feature_count}, rows={self.row_count})"
