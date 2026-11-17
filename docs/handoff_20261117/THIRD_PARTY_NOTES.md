# Third-Party Source Notes

This private research handoff includes the existing AdaRES working-tree source
and missing runtime source read from its original server-2 deployment.
It does not introduce a new blanket project license or claim authorship of
third-party code.

- `AdaRES/setup.py` retains the original IPDN metadata and author attribution.
- `AdaRES/pointnet2/` retains its original copyright and MIT-license references
  to Facebook/Pointnet2_PyTorch. Verify the complete upstream license before
  redistribution beyond the authorized private archive.
- `AdaRES/ipdn/torch/` contains existing vendored/customized PyTorch code,
  including the project's `nn_new` attention path. Preserve source notices and
  review upstream licensing if redistributing.
- PointGroup, segmentator, scene-graph parser and other existing modules retain
  their original in-source notices. This handoff does not certify that every
  upstream license file was present in the historical deployment.
- Pretrained RoBERTa/SPFormer weights, ScanNet/ScanRefer data and CLIP features
  are not included in Git and retain their own access and license requirements.

`publication_manifest.json` records byte hashes and origins.
The remote source collection does not dereference the `backbones` symlink or
copy model weights, datasets, compiled native binaries or login credentials.
