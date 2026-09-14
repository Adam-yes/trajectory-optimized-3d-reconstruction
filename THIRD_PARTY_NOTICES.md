# Third-party notices and licensing status

## Licensing status of this repository

No blanket license has been assigned to the code, configuration or scene assets in this
repository. Nothing here should be read as a grant of MIT, Apache, BSD or any other
terms. The rights holders must select and approve a release license before public
distribution.

## External systems

This package integrates with external projects that it does **not** bundle and does not
relicense:

| System | Used for | Obtained separately |
|---|---|---|
| VGGT | Feed-forward sparse-view reconstruction | Upstream checkout and local checkpoint |
| BUFFER-X | Learned point-cloud registration | Upstream checkout and local checkpoint |
| TEASER++ | Optional robust transform estimation | Upstream build |
| ROS 2 and MoveIt 2 | Planning, execution and image capture | System installation |
| NVIDIA Isaac Sim | Simulated workcell and synthetic imagery | Vendor installation |
| Open3D, trimesh, PyTorch, NumPy, SciPy, Pillow, matplotlib | Geometry, inference and figures | Python package index |

Each retains its own license and citation requirements. Review them before
redistribution or publication.

## What is not included

No trained model weights, no font files, no captured imagery and no CAD assets are
bundled. The adapters never download weights; they load a local checkpoint whose
SHA-256 you supply explicitly.
