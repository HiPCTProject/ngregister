# ngregister

Start a local Neuroglancer instance with pre-configured shortcuts for custom registration actions:

- **Translate** the active layer at the cursor 🖱️ to the center 🎯 of the view with **\<t\>**.
- **Place** a translation landmark 📍 at the cursor 🖱️ (in the fixed volume) with **\<y\>**,<br />and **translate** the active layer at the position of the cursor 🖱️ to the landmark 📍 with **\<shift+t\>**.<br />You can delete manually the "\_\_LANDMARK\_\_" layer when done.
- **Rotate** from the center 🎯 of the view around the <u>absolute</u> axis $z$ (resp. $x$ and $y$) clockwise 🔃 with **\<i\>** (resp. **\<j\>** and **\<k\>**). <br/>**Rotate** counter-clockwise 🔄 with **\<shift+i\>**, **\<shift+j\>**, **\<shift+k\>**.

## How to use

- Install dependencies using a virtual environment and the [./requirements.txt](./requirements.txt) file.
- Start an interactive Python session with (blank state):

```bash
python -i ngregister.py
```

- Or start with a previous state from either a JSON file or a Neuroglancer URL:

```bash
python -i ngregister.py --url "https://neuroglancer-demo.appspot.com/#!..."
```

- Either save manually the JSON file from the web browser or get the Neuroglancer URL once you quit the interactive Python session.
