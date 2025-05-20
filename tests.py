# I just have to go through the subsample dict and note the results of the training.
#does this warrant having a new function for testing. 
#This is for a 4 fidelity model no scan
#Ngllbsc performance (diferent amounts of ngllbsc) vs npbe
#nscan perfromance
#nexp vs npbe
#nhse vs npbe
from multi_main import MultiTrainer

model_params = {
    "num_elements": 118,
    "num_fidelities": 5,
    "embedding_dim": 114,
    "fidelity_dim": 16,
    "num_blocks": 5,
    "num_heads": 10,
    "hidden_dim": 250,
    "dropout": 0.1,
    "pooling_type": "gated"
}
training_params = {
    "multi_train_split": 0.8,
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
}
subsample_dict = {"pbe": 0.6, "scan": 1, "gllb-sc": 1, "hse": 1, "expt": 1}
trainer = MultiTrainer(model_params=model_params, subsample_dict=subsample_dict, training_params=training_params)
results, plot_paths = trainer.run_multifidelity_experiments()
print(results)
