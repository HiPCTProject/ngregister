# ngregister

Start a local Neuroglancer instance with pre-configured shortcuts for custom registration actions:

- **Translate** the active layer at the cursor 🖱️ to the center 🎯 of the view with **\<t\>**.
- **Place** a translation landmark 📍 at the cursor 🖱️ (in the fixed volume) with **\<y\>**,<br />and **translate** the active layer at the position of the cursor 🖱️ to the landmark 📍 with **\<shift+t\>**.<br />You can delete manually the "\_\_LANDMARK\_\_" layer when done.
- **Rotate** from the center 🎯 of the view around the <u>absolute</u> axis $z$ (resp. $x$ and $y$) clockwise 🔃 with **\<i\>** (resp. **\<j\>** and **\<k\>**). <br/>**Rotate** counter-clockwise 🔄 with **\<shift+i\>**, **\<shift+j\>**, **\<shift+k\>**.

And an automated **refinement** step that fine-tunes the current alignment around a landmark using an affine [SimpleITK](https://simpleitk.org/) registration on subvolumes fetched lazily with [TensorStore](https://google.github.io/tensorstore/) (supports precomputed, zarr, sharded zarr3 and n5):

- **Tag** the active layer as the **moving** layer (the one being refined) with **\<alt+m\>**, and as the **reference** (fixed) layer with **\<alt+r\>**. This prepends a `mov::` / `ref::` prefix to the layer name. Tagging is only needed when the moving/reference pair is ambiguous (more than two image layers); with exactly two image layers the selected one is the moving layer and the other is the reference.
- **Place** a translation landmark 📍 with **\<y\>** to mark the region to refine, then in the interactive Python session call:

  ```python
  refine_registration()              # default: 200-voxel cube around the landmark
  refine_registration(size_voxels=128)
  ```

  It fetches a small subvolume around the landmark from both layers, runs the registration, and composes the resulting affine correction onto the moving layer. The change is recorded in `__HISTORY__` like the manual gestures.

## How to use

- Install dependencies using a virtual environment and the [./requirements.txt](./requirements.txt) file.

> [!TIP]
> To support the most recent Neuroglancer states which load resources using, e.g. `gs://my-bucket/data.ome.zarr|zarr:`, you will need the development version of Neuroglancer, that can be installed using:
>
> ```bash
> pip install git+https://github.com/google/neuroglancer.git
> ```
> 
> You will require a build environment, including `node` and `Python.h`.
> To install `node` on your system, use [`nvm`](https://www.nvmnode.com/).
> If you use [`uv`](https://docs.astral.sh/uv/) (`uv pip ...`), you should pick and install a `uv`-managed version of Python, as explained [here](https://docs.astral.sh/uv/guides/install-python/).

- Start an interactive Python session with (blank state):

```bash
python -i ngregister.py
```

- Or start with a previous state from either a JSON file or a Neuroglancer URL:

```bash
python -i ngregister.py --url "https://neuroglancer-demo.appspot.com/#!..."
```

- Either save manually the JSON file from the web browser or get the Neuroglancer URL once you quit the interactive Python session.

## Other related projects

As of https://github.com/neuroscales/ngtools/pull/42, these features have been implemented in [neuroscales/ngtools](https://github.com/neuroscales/ngtools), which handles more complex use cases. [HiPCTProject/ngregister](https://github.com/HiPCTProject/ngregister) can still be used as a standalone.

> [!WARNING]
> Keybindings are not yet uniformized between the two tools.
