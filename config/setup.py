initial_model = {
    "train_dataset": "train",
    "ims_per_batch": 5,
    "base_lr": 0.001,
    "warmup_iters": 200,
    "model_weights": None,
    "num_decays": 4,
    "steps": (400, 500, 600, 700),
    "gamma": 0.2,
    "max_iter": 1000,
    "eval_period": 2000,
}

initial_score = {
    "train_dataset": "train",
    "ims_per_batch": 5,
    "base_lr": 0.0001,
    "warmup_iters": 100,
    "model_weights": None,
    "num_decays": 4,
    "steps": (200, 300, 400, 500),
    "gamma": 0.2,
    "max_iter": 1000,
    "eval_period": 2000,
}

cycle_model = {
    "ims_per_batch": 5,
    "base_lr": 0.00005,
    "warmup_iters": 100,
    "model_weights": None,
    "num_decays": 0,
    "steps": (),
    "gamma": 0.2,
    "max_iter": 400,
    "eval_period": 2000,
}

cycle_score = {
    "ims_per_batch": 5,
    "base_lr": 0.00005,
    "warmup_iters": 100,
    "model_weights": None,
    "num_decays": 0,
    "steps": (),
    "gamma": 0.2,
    "max_iter": 400,
    "eval_period": 2000,
}
