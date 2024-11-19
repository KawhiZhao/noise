
### create the environment

```
conda env create -f environment.yaml
```

### train

```
python main_stft.py
```
The multi-channel speech is generated online based on librispeech (test-other-500 and dev-other), which is needed to downloaded in advance. Need to replace the data path under speech_data folder.
For the noise file, you may want to change the esc50 dataset path in data/simulated_speech_stft.py

For the input parameter --snr in main_stft.py, -1 means no noise, you can also choose other db level such as -10, 0, 10 etc

In the training process, if you do not want to use the wandb, you can disable the following line

```
    wandb_logger = WandbLogger(
        project='noise adaptation', 
        name='stft resnet conformer simualted', 
    )
    args.logger = wandb_logger
```

### test

```
python main_stft_test.py
```
Need to change the ckpt path --load_path

### model performance

| Model    | no noise | 20db      | 10db      | 0db      | -10db      | -20db      |
|----------|----------|-----------|-----------|----------|------------|------------|
|no noise  | < 1      | 4.8       | 11        | 31       | 48         | 61         |
|20db noise| < 1      | < 1       | < 1       | 2        | 9.8        | 25         |


