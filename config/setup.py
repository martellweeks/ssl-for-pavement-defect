initial_model = {
    "train_dataset": "train",
    "ims_per_batch": 5,
    "base_lr": 0.001,
    "warmup_iters": 200,
    "model_weights": None,
    "num_decays": 4,
    "steps": (600, 700, 800, 900),
    "gamma": 0.2,
    "max_iter": 2000,
    "eval_period": 2000,
}

initial_score = {
    "train_dataset": "train",
    "ims_per_batch": 1,
    "base_lr": 0.00001,
    "warmup_iters": 100,
    "model_weights": None,
    "num_decays": 0,
    "steps": (),
    "gamma": 0.2,
    "max_iter": 5000,
    "eval_period": 200000,
}

initial_score_mask_test = {
    "train_dataset": "train",
    "ims_per_batch": 1,
    "base_lr": 0.000001,
    "warmup_iters": 100,
    "model_weights": None,
    "num_decays": 0,
    "steps": (),
    "gamma": 0.2,
    "max_iter": 10000,
    "eval_period": 200000,
}

cycle_model = {
    "ims_per_batch": 5,
    "base_lr": 0.0001,
    "warmup_iters": 100,
    "num_decays": 0,
    "steps": (),
    "gamma": 0.2,
    "max_iter": 1500,
    "eval_period": 2000,
}

cycle_model_short = {
    "ims_per_batch": 5,
    "base_lr": 0.0001,
    "warmup_iters": 100,
    "num_decays": 0,
    "steps": (),
    "gamma": 0.2,
    "max_iter": 500,
    "eval_period": 2000,
}

cycle_score = {
    "ims_per_batch": 1,
    "base_lr": 0.00001,
    "warmup_iters": 100,
    "num_decays": 0,
    "steps": (),
    "gamma": 0.2,
    "max_iter": 2000,
    "eval_period": 3000,
}

cns_test = {
    "base_lr": 0.001,
    "warmup_iters": 200,
    "model_weights": None,
    "num_decays": 4,
    "steps": (600, 700, 800, 900),
    "gamma": 0.2,
    "max_iter": 2000,
    "eval_period": 10000,
    "checkpoint_period": 100,
    "cns_beta": 2,
    "cns_w_t0": 200,
    "cns_w_t1": 400,
    "cns_w_t2": 1600,
    "cns_w_t": 1800,
}

cns_control = {
    "base_lr": 0.001,
    "warmup_iters": 200,
    "model_weights": None,
    "num_decays": 4,
    "steps": (600, 700, 800, 900),
    "gamma": 0.2,
    "max_iter": 2000,
    "eval_period": 10000,
    "checkpoint_period": 100,
    "cns_w_t0": 10000,
    "cns_w_t1": 15000,
    "cns_w_t2": 20000,
    "cns_w_t": 2000,
}

al_control = {
    "ims_per_batch": 1,
    "base_lr": 0.001,
    "warmup_iters": 200,
    "model_weights": None,
    "num_decays": 4,
    "steps": (600, 700, 800, 900),
    "gamma": 0.2,
    "max_iter": 2000,
    "eval_period": 10000,
    "checkpoint_period": 100,
}
