# Pinned upstream references

These checkouts are reference-only and are excluded from the parent Git repository.

| Repository | Pinned commit | Purpose | Software license status checked 2026-07-31 |
|---|---|---|---|
| `longstnguyen/ViRE` | `dfd092cf0750f304f67264455aa91b4723618f51` | Dataset/evaluation reproduction | No `LICENSE` file or GitHub-detected license |
| `ivpb/qiepsm` | `abf9779dc7de439293cd5072bf6d07a1433bd3c7` | QI formulation audit only | No `LICENSE` file or GitHub-detected license |

No source file from either repository may be copied into `src/`. CSConDa remains in the ignored ViRE checkout and is not redistributed by QIR-Route. ViRE documents CSConDa as CC BY-NC 4.0; third-party datasets retain their original licenses.

Verification:

```bash
git -C upstream/ViRE rev-parse HEAD
git -C upstream/qiepsm rev-parse HEAD
```
