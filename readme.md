# Income Prediction

Predicting average yearly income based on satellite imagery using CNNs.

Built for the Stanford ACMLab Fall 2020 project
by Clare Chua, Ryan Chi, Tejas Narayanan, Nathan Kim, and Zander Lack
(Team CRouToNZ).

## Setup

### Data
Add the following files/folders to the `data` folder:
```
imagery/
test_imagery/
16zpallnoagi.csv
worldelev.npy
ziplatlon.csv
```
These files are not included in the Git repository because they are too large.

### Libraries

Next, ensure the most recent versions of the following libraries are installed:
```
numpy==1.19.2
torch==1.6.0
Pillow==8.0.0
pandas==1.1.3
```

To set up a virtual environment for this project:

1. Navigate to the project directory in the terminal.
2. `python3 -m venv venv/`. This will create a virtual environment folder.
3. `source venv/bin/activate`. This will activate the virtual environment.
`(venv)` should appear at the beginning of the terminal prompt.
4. `pip install numpy torch pillow pandas`. This will install the most recent versions
of Numpy, PyTorch, and Pillow (Python Image Library).

After following these steps the first time, the only required step is to
`source venv/bin/activate` to activate the environment.