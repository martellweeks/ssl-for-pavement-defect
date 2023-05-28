from config import config, paths, setup
from src.controller import d2_mask
from src.data import al_label_transfer
from src.scores import al_scoring


def cns_control_0519():
    cfg = config.get_cfg_for_cns(**setup.cns_control)
    d2_mask.train_model_only_cns(
        output_folder="0519_cns_control",
        regist_instances=True,
        logfile="0519_cns_control",
        cfg=cfg,
    )


def cns_exp_0519():
    cfg = config.get_cfg_for_cns(**setup.cns_exp)
    d2_mask.train_model_only_cns(
        output_folder="0519_cns_exp",
        regist_instances=True,
        logfile="0519_cns_exp",
        cfg=cfg,
    )


def cns_exp_0519_2():
    cfg = config.get_cfg_for_cns(**setup.cns_exp2)
    d2_mask.train_model_only_cns(
        output_folder="0519_cns_exp2",
        regist_instances=True,
        logfile="0519_cns_exp2",
        cfg=cfg,
    )


def cns_control_0511():
    cfg = config.get_cfg_for_cns(**setup.cns_control)
    d2_mask.train_model_only_cns(
        output_folder="0511_control_cns",
        regist_instances=True,
        cfg=cfg,
    )


def al_control_0511():
    cfg = config.get_cfg_for_al(**setup.al_control)
    d2_mask.train_model_only(
        output_folder="0511_control_al",
        regist_instances=True,
        cfg=cfg,
    )


def vanilla_control_0519():
    cfg = config.get_cfg_for_al(**setup.al_control)
    d2_mask.train_vanilla_mrcnn(
        output_folder="0519_control_vanilla",
        regist_instances=True,
        logfile="0519_control_vanilla",
        cfg=cfg,
    )


def regular_mask_rcnn_training_0511():
    logger, _ = d2_mask.startup(
        regist_instances=True, cfg=None, logfile="0511_A12_patch_vanilla"
    )

    cfg = config.get_cfg_for_al(
        warmup_iters=400,
        steps=(700, 1000, 1300, 1600),
        max_iter=2000,
        eval_period=250,
    )

    d2_mask.train_vanilla_mrcnn(
        output_folder="0512_A12_patch_vanilla", cfg=cfg, logger=logger
    )


def exp_0509(mode: str = "AL"):
    """
    Experiment on May 09
    Checking simultaneous training of
    box loss predictor and mask loss predictor can be done
    """

    d2_mask.startup(regist_instances=True, cfg=None)

    d2_mask.train_scores_only(
        output_folder="0509_init_score",
        model_weights=paths.final_model_full_path,
        regist_instances=False,
        cfg=config.get_cfg_for_al(**setup.initial_score),
    )

    d2_mask.predict_scores(
        model_weights=paths.final_model_full_path,
        output_path="./output/0509_init_score",
        regist_instances=False,
        test_anns_file="./data/annotations/A14_L2/test.json",
    )

    al_scoring.calculate_al_score(
        file_path="./output/0509_init_score/loss_predictions.csv",
        output_path="./output/0509_init_score",
    )

    if mode == "AL":
        update_image_list = al_scoring.get_top_n_images(
            score_file_path="./output/0509_init_score/al_score.csv", no_img=40
        )
    else:
        update_image_list = al_label_transfer.get_random_n_images(
            test_anns_file="./data/annotations/A14_L2/test.json", no_img=40
        )

    al_label_transfer.move_labels_to_train(
        train_anns_file="./data/annotations/A14_L2/train.json",
        image_list=update_image_list,
        output_path="./output/0509_labels",
        output_file_tag="0",
    )


def exp_0324(mode: str = "AL"):
    """
    Experiment on Mar 24
    Based on first experiment on Mar 22
    Testing active learning pipeline for A14 Lane 2 Pavement data
    Against random sampling control

    Same pipeline repeated on Apr 14
    With mask score predictor finally working!
    """
    # d2_mask.train_model_only(
    #     output_folder="0324_init_model",
    #     regist_instances=True,
    #     cfg=config.get_cfg_for_al(**setup.initial_model),
    # )

    d2_mask.get_coco_eval_results(
        model_weights="./output/A14_L2/0414_init_model/model_final.pth",
        regist_instances=True,
        output_path="./output/0414_init_model",
    )

    # d2_mask.train_box_score_only(
    #     output_folder="0324_init_score_box",
    #     model_weights=paths.final_model_full_path,
    #     regist_instances=True,
    #     cfg=config.get_cfg_for_al(**setup.initial_score),
    # )

    # d2_mask.train_mask_score_only(
    #     output_folder="0324_init_score",
    #     model_weights=paths.final_model_full_path,
    #     regist_instances=False,
    #     cfg=config.get_cfg_for_al(**setup.initial_score),
    # )

    d2_mask.predict_scores(
        model_weights=paths.final_model_full_path,
        output_path="./output/0414_init_score",
        regist_instances=False,
        test_anns_file="./data/annotations/A14_L2/test.json",
    )

    al_scoring.calculate_al_score(
        file_path="./output/0414_init_score/loss_predictions.csv",
        output_path="./output/0414_init_score",
    )

    if mode == "AL":
        update_image_list = al_scoring.get_top_n_images(
            score_file_path="./output/0414_init_score/al_score.csv", no_img=40
        )
    else:
        update_image_list = al_label_transfer.get_random_n_images(
            test_anns_file="./data/annotations/A14_L2/test.json", no_img=40
        )

    al_label_transfer.move_labels_to_train(
        train_anns_file="./data/annotations/A14_L2/train.json",
        image_list=update_image_list,
        output_path="./output/0414_labels",
        output_file_tag="0",
    )

    # d2_mask.predict_scores(
    #     model_weights="output/A14_L2/0414_init_score/model_final.pth",
    #     output_path="./tests",
    #     regist_instances=False,
    #     test_anns_file="./data/annotations/A14_L2/quick_test.json",
    # )

    for it in range(10):
        d2_mask.register_new_coco_instance(
            annotation_path=f"./output/0414_labels/train_{it}.json",
            data_path=paths.raw_data_path,
            tag=f"train_{it}",
        )

        d2_mask.register_new_coco_instance(
            annotation_path=f"./output/0414_labels/test_{it}.json",
            data_path=paths.raw_data_path,
            tag=f"test_{it}",
        )

        if it >= 3:
            d2_mask.train_model_only(
                output_folder=f"0414_model_{it}",
                regist_instances=False,
                model_weights=paths.final_model_full_path,
                cfg=config.get_cfg_for_al(
                    train_dataset=f"train_{it}", **setup.cycle_model_short
                ),
            )
        else:
            d2_mask.train_model_only(
                output_folder=f"0414_model_{it}",
                regist_instances=False,
                model_weights=paths.final_model_full_path,
                cfg=config.get_cfg_for_al(
                    train_dataset=f"train_{it}", **setup.cycle_model
                ),
            )

        # Always run eval of model on original test set (original unlabeled pool)
        # Do not pass test_data_tag parameter
        d2_mask.get_coco_eval_results(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path=f"./output/0414_model_{it}",
        )

        d2_mask.train_box_score_only(
            output_folder=f"0414_score_{it}_box",
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            cfg=config.get_cfg_for_al(train_dataset=f"train_{it}", **setup.cycle_score),
        )

        d2_mask.train_mask_score_only(
            output_folder=f"0414_score_{it}",
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            cfg=config.get_cfg_for_al(train_dataset=f"train_{it}", **setup.cycle_score),
        )

        d2_mask.predict_scores(
            model_weights=paths.final_model_full_path,
            output_path=f"./output/0414_score_{it}",
            regist_instances=False,
            test_anns_file=f"./output/0414_labels/test_{it}.json",
        )

        if mode == "AL":
            al_scoring.calculate_al_score(
                file_path=f"./output/0414_score_{it}/loss_predictions.csv",
                output_path=f"./output/0414_score_{it}",
            )

            update_image_list = al_scoring.get_top_n_images(
                score_file_path=f"./output/0414_score_{it}/al_score.csv", no_img=40
            )
        else:
            update_image_list = al_label_transfer.get_random_n_images(
                test_anns_file=f"./output/0414_labels/test_{it}.json", no_img=40
            )

        al_label_transfer.move_labels_to_train(
            train_anns_file=f"./output/0414_labels/train_{it}.json",
            test_anns_file=f"./output/0414_labels/test_{it}.json",
            image_list=update_image_list,
            output_path="./output/0414_labels",
            output_file_tag=f"{it+1}",
        )


# Experiments for 0519 CNS final verification


def exp_0519_long_cns_base():
    cfg = config.get_cfg_for_cns(**setup.exp_0519_long_cns_base)
    d2_mask.train_model_only_cns(
        output_folder="exp_0519/exp_0519_long_cns_base",
        regist_instances=True,
        logfile="exp_0519_long_cns_base",
        cfg=cfg,
    )


def exp_0519_long_cns_bigbatch():
    cfg = config.get_cfg_for_cns(**setup.exp_0519_long_cns_bigbatch)
    d2_mask.train_model_only_cns(
        output_folder="exp_0519/exp_0519_long_cns_bigbatch",
        regist_instances=True,
        logfile="exp_0519_long_cns_bigbatch",
        cfg=cfg,
    )


def exp_0519_long_vanilla():
    cfg = config.get_cfg_for_al(**setup.exp_0519_long_vanilla)
    d2_mask.train_vanilla_mrcnn(
        output_folder="exp_0519/exp_0519_long_vanilla",
        regist_instances=True,
        logfile="exp_0519_long_vanilla",
        cfg=cfg,
    )


def exp_0519_long_vanilla_bigbatch():
    cfg = config.get_cfg_for_al(**setup.exp_0519_long_vanilla_bigbatch)
    d2_mask.train_vanilla_mrcnn(
        output_folder="exp_0519/exp_0519_long_vanilla_bigbatch",
        regist_instances=True,
        logfile="exp_0519_long_vanilla_bigbatch",
        cfg=cfg,
    )


def exp_0519_long_cns_bigbatch_wronglabels():
    cfg = config.get_cfg_for_cns(**setup.exp_0519_long_cns_bigbatch)
    d2_mask.train_model_only_cns(
        output_folder="exp_0519/exp_0519_long_cns_bigbatch_wronglabels",
        regist_instances=True,
        logfile="exp_0519_long_cns_bigbatch_wronglabels",
        cfg=cfg,
    )
