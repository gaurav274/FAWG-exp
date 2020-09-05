## Enviornment Installation
- Install Anaconda3
- For setting up the enviornment run : 
   ```
        bash scripts/create_experiment_env_linux.sh srtml-exp
        conda activate srtml-exp
        python -c "import srtml; srtml.init()"
    ```


## Model Repository

### Commands
- `cleanmr`: Cleans the model repository
- `lsmr`: lists the model repisotory
- `plsmr`: specify s3 uri and dive deeper into model repository tree

## Experiment1 - Prepoc
One end-to-end running example of image preproc

### Commands
For any command run `<cmd> --help` to get inputs

- `prepoc_profile`: profile the vertices given from a config file. Config files look like : 

    ```
        [
           {
               "Model Name": "resnet50",
               "Accuracy": 75.8
           },
           {
               "Model Name": "resnet34",
               "Accuracy": 75.8
           }
        ]
    ```
    
- `prepoc_populate`: puts the profiled models into model repository
- `prepoc_configure`: configures the virtual abstract image classification model based on arrival curve config. Config looks like

    ```
        [
           {
               "mu (qps)": 100.0,
               "cv": 0,
               "# requests": 2000,
               "Latency Constraint (ms)": 100.0,
               "Planner": "SimulatedAnnealing"
           }
        ]
    ```
- `prepoc_provision`: provisions the configured models to get latency, throughput information

### Demo
```
prepoc_profile
prepoc_populate
prepoc_configure
prepoc_provision

ls image_preprocessing/two_vertex/accuracy_degradation/virtual/virtual_image_classification.xlsx
ls image_preprocessing/two_vertex/accuracy_degradation/physical/image_classification.xlsx
```


