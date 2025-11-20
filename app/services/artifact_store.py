# artifact_store.py
import os, uuid
from pathlib import Path

ARTIFACT_DIR = Path("./artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

class ArtifactStore:
    def save(self, data: bytes, mime: str, filename: str) -> dict:
        artifact_id = str(uuid.uuid4())
        safe_name = f"{artifact_id}_{filename}"
        path = ARTIFACT_DIR / safe_name
        with open(path, "wb") as f:
            f.write(data)
        return {"artifact_id": artifact_id, "url": f"/artifacts/{safe_name}", "path": str(path)}

artifact_store = ArtifactStore()
