# Legacy files

This folder keeps old files for reference. No current script uses them.

| File | Origin | Status |
|---|---|---|
| `main.cpp` | Redis version of the engine | Not built. It does not examine a `NULL` reply. Its loop has no delay. |
| `train_v1.py` | v1 synchronous training script | Cannot run. It imports `RLAgent`, which `agent.py` does not contain. |

The v1 results were not kept. These files cannot reproduce them.
