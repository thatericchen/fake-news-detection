# Usage Document
## 1. Clone the repo
```
git clone https://github.gatech.edu/ML4641Team19/CS4641-Project.git
cd <repo-folder>
```
## 2. Create and activate Conda env
```
conda env create -f environment.yml
conda activate MLP
```
## 3. Run the pipeline script
```
python data_pipeline.py
```
Optional flags (--force-download, --force-combine, --force-preprocess)
- Ex:  ``` python data_pipeline.py --force-download --force-combine ```

## 4. Choose which model to run
```
logisticregression / naivebayes / randomforest
```

## Important Notes
- When you input your choice of model, the program will automatically apply lower() and remove all spaces.
- When you run the pipeline, all of the .csv files and extra files will be found in the data folder.
- Data is local so do not try to commit it or you will not be able to push due to its size (if you accidentally do, run 'git log --oneline' and remove that specific commit).
- If you installed new packages using ```pip install```, you have to use ``` conda env export > environment.yml ``` to update it.
