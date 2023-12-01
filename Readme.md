# Industrial Science project solution

## Instalation

In order to get all the packages installed, create an environment using any package manager (conda, pip, poetry, ...). All the dependencies can be extracted from ```requirements.txt``` file. Below is an example of installing the packages using pip:

```pip install requirements.txt```

## Running the program

The program can be run using command line. It takes such arguments:
-   ```--datapath``` (required) - path to the signal file
-   ```--savepath``` (required) - path to save to/read from the analysis data
-   ```--analyze``` (flag, optional) - if provided the analysis will be performed on the signal file
-   ```--save``` (flag, optional) - if provided the result of the analysis will be saved to the file provided by ```--savepath```, has no effect if ```--analyze``` is not provided
-   ```visualize``` (flag, optionla) - whether to make a visualization of the analysis results and save it to a folder provided by ```--vis_path```, can be used both with and visout ```--analyze```, in the later case the data will be read from the file provided by ```--savepath```
-   ```--vis_path``` (optional, default=```./images```) - path to a folder for storing visualization images, will be created if doesn't exist

### Example of an execution command
```python final_code.py --datapath="../2Gb signal.csv" --savepath="../tmp.csv" --analyze --save --visualize --vis_path="./images"```