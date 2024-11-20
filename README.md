## Sheet 02 submission data links
#### Task 01 synthetic data
- [k2_synthetic_data](https://www.b-tu.de/owncloud/s/SHK8BZwKqnPKmZ2)
- [k5_synthetic_data](https://www.b-tu.de/owncloud/s/BzWxAb7wCacfzYx)
#### Task 02 plots
- [plots](https://www.b-tu.de/owncloud/s/Wn4ipAY9ewx367w)
#### Other required data
- [gan_synthetic_data](https://www.b-tu.de/owncloud/s/kK6fyDrtAereLzg)
- [original_data_for_gan_training](https://www.b-tu.de/owncloud/s/SWR7WD628GC4dK3)

> Task 03 is implemented in `app.py`

## Project Structure
```
project_root/
├── data/
│   ├── gan/ ...
│   ├── original/...
│   └── oversampling/
│       ├── k2_synthetic_data.csv
│       ├── k5_synthetic_data.csv
│       ├── train_4_task_02.csv
│       └── synthetic_data-HU.csv
├── app.py
└── src/
    ├── sheet_01/ ...
    └── sheet_02/
        ├── __init__.py
        ├── task_02.py
        └── task_01.py
```

>The solution for **Sheet 2** which is submitted for assessment is limited to the files within `sheet_02` directory. However here is complete [[source code]](https://github.com/darth-assan/stuProj.git)

## Requirements
Install the required packages using pip:
```bash
pip install -r requirements.txt
```
The datasets within the `data/oversampling` is required for the demonstration and should be populated as is. If files are saved at different locations, they should be configured respectively. **These datasets are submitted with the source code.**


## Usage
Lunch the toolbox in a specific mode
```
python app.py -s {mode}
```

Read the help manual for detailed guidelines
```
python app.py -h
```