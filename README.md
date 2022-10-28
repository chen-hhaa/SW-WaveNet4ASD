# SW-WaveNet4ASD
This is a pytorch implementaion of paper "SW-WaveNet: Learning Representation from Spectrogram and Wavegram Using WaveNet for Anomalous Sound Detection"
![Fig1_overview](./Fig1_overview.png)


### dataset
We manually mixed the development and additional training dataset of DCASE 2020 Task2
+ development dataset: https://zenodo.org/record/3678171
+ additional training dataset: https://zenodo.org/record/3727685

If you want test the result in evaluation dataset (https://zenodo.org/record/3841772#.YoCDkOhBxaQ), you can use official evaluator: https://github.com/y-kawagu/dcase2020_task2_evaluator

data directory tree:
```text
data
├── dataset
│   ├── fan
│   │   ├── test
│   │   └── train
│   ├── pump
│   │   ├── test
│   │   └── train
│   ├── slider
│   │   ├── test
│   │   └── train
│   ├── ToyCar
│   │   ├── test
│   │   └── train
│   ├── ToyConveyor
│   │   ├── test
│   │   └── train
│   └── valve
│       ├── test
│       └── train
└── pre_data
    └── dataset_train_path_list.db  # Automatically generated file
```

 ### Cite
