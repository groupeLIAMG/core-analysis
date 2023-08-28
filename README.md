# Core analysis by deep learning
## Exploring the capabilities of *Transformer*-based neural network

Script port of the core analysis project developed by Victor Silva dos Santos and Eric: https://drive.google.com/drive/folders/1SSuE4GVQ5xjePujH0AGdXBVWXauanteG

### Installation

Run `pip install -e .` or `conda env create -f environment.yml; conda activate core-analysis`.

Create a symbolic link called `data` to the `PROJ_ERIC` Google Drive folder (`mklink /D data <drive_path>` on Windows or `ln -s <drive_path> data` on Linux) and a symbolic link called `images` to the `SELECTION FORAGES SUNRISE` folder. The links should be at the root folder of the Python module (`core-analysis/data` and `core-analysis/images`).

### Usage

Run `python core_analysis <--train> <--test> <-p --plot> <-w --weights-filename> <-a --do-augment> <-e --eager-execution>` to run the main script, or use the (equivalent) `Core analysis.ipynb` Jupyter notebook.

Arguments:
- `--train`: train the model
- `--test`: test the model
- `-p` or `--plot`: show the results onscreen (they are saved to disk anyway at `core-analysis/plots`)
- `-w` or `--weights-filename`: filename of the checkpoint file to load
- `-a` or `--do-augment`: use data augmentation (up/downscaling, left-right flip, up-down flip)
- `-e` or `--eager-execution`: execute TensorFlow/Keras in eager mode (slower, but allows debugging)

### Contents

- `Core analysis.ipynb`: Jupyter notebook version of the main script
- `core_analysis`:
    - `__main__.py`: entry point of `python core_analysis <args>`
    - `architecture.py`: define model and hyperparameters
    - `dataset.py`: load and manage data
    - `postprocess.py`: split data into tiles and merge them back
    - `preprocess.py`: preprocess the data prior to training (e.g. detecting background)
    - `utils`:
        - `constants.py`: constants
        - `format_figures.py`: default figure format and general Pyplot utilities
        - `processing.py`: ease data management
        - `transform.py`: geometrical and data augmentation utilities
        - `visualize.py`: plot results in an unified way