# Tests

From the repo root, with the virtualenv activated:

```bash
pytest tests/ -q                        # full suite
pytest tests/test_unit.py -v            # fast unit tests (~5s)
pytest tests/test_grid_regression.py    # full-model regression vs pinned
                                        # snapshots (~3 min, re-runs the
                                        # model at ~160 parameter points)
```

The regression snapshots in `tests/reference/grid/` are regenerated
with `tools/build_grid_python.py`; regenerate them in the same commit
as any intentional physics change.
