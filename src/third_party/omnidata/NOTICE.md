This directory vendors the minimal Omnidata MiDaS implementation required by
`src/model/encoder/mono_estimator/mono_normal.py`.

Source:
- Repository: https://github.com/HanzhiChang/MeshSplat
- Commit: `256027dbd735dddc06f0ddbcf5c9438b32d38410`
- Imported files:
  - `src/omnidata/modules/midas/__init__.py`
  - `src/omnidata/modules/midas/base_model.py`
  - `src/omnidata/modules/midas/blocks.py`
  - `src/omnidata/modules/midas/dpt_depth.py`
  - `src/omnidata/modules/midas/vit.py`

Local modifications:
- `dpt_depth.py` defaults to `pretrained=False` during backbone construction so
  loading the mono-normal checkpoint does not trigger an additional download of
  ImageNet pretrained weights.

License:
- See `LICENSE.MeshSplat` in this directory.
