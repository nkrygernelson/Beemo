import optuna

from multi_main import MultiTrainer


#subsample_dict = {"pbe": 0.1, "scan": 1, "gllb-sc": 1, "hse": 1, "expt": 1}



def objective(trial):
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
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
    }
    
    mp = True
    if mp:
        fidelity_map = {"pbe":0, "scan":1, "gllbsc":2, "hse":3, "expt":4}
        fidelities_dir = "train"
        subsample_dict = {"pbe": 1, "scan": 1, "gllb-sc": 1, "hse": 1, "expt": 1}
    else:
        fidelities_dir = "train_homemade"
        fidelity_map = {
            "GGA": 0,
            "SCAN": 1,
            "GLLBSC": 2,
            "HSE": 3,
            "EXPT": 4
            }
        subsample_dict = {"GGA": 1, "SCAN": 1, "GLLBSC": 1, "HSE": 1, "EXPT": 1}
    #fidelities_dir = "train"
    trainer = MultiTrainer(model_params=model_params, subsample_dict=subsample_dict, training_params=training_params, fidelity_map=fidelity_map, property_name='BG', fidelities_dir=fidelities_dir)
    combined_train_df, test_datasets, train_stats = trainer.prepare_datasets_for_multifidelity()
    model, preprocess = trainer.train_multifidelity_model(combined_train_df)
    nmaes = {}
    total_nmaes = 0
    for dataset_name, dataset in test_datasets.items():
        train_mean, train_std, num_samples = train_stats[dataset_name]["mean"], train_stats[dataset_name]["std"], train_stats[dataset_name]["num_samples"]
        test_mean, test_std = dataset[trainer.property_name].mean(), dataset[trainer.property_name].std()
        test_loader = trainer.create_test_dataloader(dataset, preprocess, train_mean, train_std)
        metrics, predictions, targets = trainer.evaluate_on_fidelity( model,test_loader,  train_mean, train_std)
        nmae = metrics["mae"]/test_std
        nmaes[dataset_name] = {'nmae': nmae, "num_samples": num_samples}
        print(f"Dataset: {dataset_name}, NMAE: {nmae}, Num samples: {num_samples}")
        total_nmaes += nmae
    return total_nmaes



    
