# bitpred
Open source bitcoin price movement prediction engine.

V0.01 - Features
* Is it "going down" - tomorrow ( within 24hours) api backend classifier
* Get data from public sources (yahoo)
* Combine data with relevant influencers data ( Twitter for ) now
* Create a complete e2e backend classifier and ML pipeline
* Deployable artefact

# data 
Currently using live streams from : Yahoo 

## How to run training

### Install MLFlow locally
`$ pip install mlflow`


### From mlflow 

Locally:
`$ mlflow run .`

From github:
`$ mlflow run https://github.com/nlauchande/bitpred/ `



## How to run a listening prediction api
Using the id of the model and the model name you can run the following commnad :

`$  mlflow models serve -m ./mlruns/0/b9ee36e80a934cef9cac3a0513db515c/artifacts/model_random_forest/ `


## How to run a prediction

`curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{"data":[[1,1,1,1,0,1,1,1,0,1,1,1,0,0]]}'                                                                                                            
[1]%`

## How to monitor experiments with MLFlow

### Run the server



## How to add a new algorithm
You can modify the train.py file .

=======
Currently using live streams from Yahoo.
 
