from config import config, paths, setup
from src.data import al_label_transfer
from src.model import d2_mask
from src.scores import al_scoring


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
    #     cfg=config.get_cfg_with_exp_setup(**setup.initial_model),
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
    #     cfg=config.get_cfg_with_exp_setup(**setup.initial_score),
    # )

    # d2_mask.train_mask_score_only(
    #     output_folder="0324_init_score",
    #     model_weights=paths.final_model_full_path,
    #     regist_instances=False,
    #     cfg=config.get_cfg_with_exp_setup(**setup.initial_score),
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
                cfg=config.get_cfg_with_exp_setup(
                    train_dataset=f"train_{it}", **setup.cycle_model_short
                ),
            )
        else:
            d2_mask.train_model_only(
                output_folder=f"0414_model_{it}",
                regist_instances=False,
                model_weights=paths.final_model_full_path,
                cfg=config.get_cfg_with_exp_setup(
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
            cfg=config.get_cfg_with_exp_setup(
                train_dataset=f"train_{it}", **setup.cycle_score
            ),
        )

        d2_mask.train_mask_score_only(
            output_folder=f"0414_score_{it}",
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            cfg=config.get_cfg_with_exp_setup(
                train_dataset=f"train_{it}", **setup.cycle_score
            ),
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


if __name__ == "__main__":
    exp_0324(mode="AL")

    # d2_mask.train_model_only(
    #     output_folder="0414_init_model",
    #     regist_instances=True,
    #     cfg=config.get_cfg_with_exp_setup(**setup.initial_model),
    # )

    # d2_mask.train_box_score_only(
    #     output_folder="0414_init_score_box",
    #     model_weights=paths.final_model_full_path,
    #     regist_instances=False,
    #     cfg=config.get_cfg_with_exp_setup(**setup.initial_score),
    # )

    # d2_mask.train_mask_score_only(
    #     output_folder="0414_init_score",
    #     model_weights=paths.final_model_full_path,
    #     regist_instances=False,
    #     cfg=config.get_cfg_with_exp_setup(**setup.initial_score),
    # )

    # d2_mask.predict_scores(
    #     model_weights='output/A14_L2/0324_init_score_box/model_0002999.pth',
    #     output_path=f"./tests",
    #     regist_instances=True,
    #     test_anns_file='./data/annotations/A14_L2/quick_test.json'
    # )

    # update_image_list = al_label_transfer.get_random_n_images(
    #     test_anns_file=f"./output/0324_labels/test_0.json", no_img=200
    # )

    # al_label_transfer.move_labels_to_train(
    #     train_anns_file=f"./data/annotations/A14_L2/metadata.json",
    #     test_anns_file=f"./output/0324_labels/test_0.json",
    #     image_list=update_image_list,
    #     output_path="./output/0324_labels",
    #     output_file_tag="for_test",
    # )
    # d2_mask.get_coco_eval_results(
    #     model_weights=paths.final_model_full_path,
    #     regist_instances=True,
    #     output_path=f"./tests",
    # )
    # imlist = al_label_transfer.get_random_n_images(
    #     test_anns_file="./data/annotations/A14_L2/raw.json", no_img=200
    # )

    # al_label_transfer.move_labels_to_train(
    #     image_list=imlist,
    #     output_path="./data/annotations/A14_L2",
    #     output_file_tag=None,
    #     train_anns_file="./data/annotations/A14_L2/metadata.json",
    #     test_anns_file="./data/annotations/A14_L2/raw.json",
    # )

    # d2_mask.train_model_only(output_folder="0313_model_2")

    # d2_mask.train_model_only(
    #     output_folder="0320_exp_rand",
    #     model_weights="./output/0317_score_2/model_0001999.pth",
    #     regist_instances=True,
    # )

    # d2_mask.train_scores_only(
    #     output_folder="0320_exp_rand",
    #     model_weights="./output/0317_score_2/model_0001999.pth",
    #     regist_instances=False,
    # )

    # d2_mask.get_coco_eval_results(
    #     model_weights="./output/0320_exp_al/model_final.pth",
    #     regist_instances=True,
    #     output_path="./output/0320_exp_al",
    # )
    # d2_mask.get_coco_eval_results(
    #     model_weights="./output/0320_exp_rand/model_final.pth",
    #     regist_instances=False,
    #     output_path="./output/0320_exp_rand",
    # )

    # d2_mask.get_coco_eval_results(
    #     model_weights="./output/0317_score_2/model_0001999.pth",
    #     regist_instances=True,
    #     output_path="./output/0317_evals",
    # )

    # d2_mask.predict_scores(
    #     model_weights="./output/0317_score_2/model_0001999.pth",
    #     output_path="./output/0320_exp",
    #     regist_instances=True,
    # )

    # al_scoring.calculate_al_score(
    #     file_path="./output/0320_exp/loss_predictions.csv",
    #     output_path="./output/0320_exp",
    # )

    # al_image_list = al_scoring.get_top_n_images(
    #     score_file_path="./output/0324_score_1/al_score.csv", no_img=40
    # )

    # d2_mask.label_predictions_on_images(
    #     image_list=al_image_list,
    #     regist_instances=True,
    #     output_path="./output/0324_score_1/imgs/al",
    # )

    # al_label_transfer.move_labels_to_train(train_anns_file="./data/annotations/empty.json", image_list=al_image_list, output_path="./output/0320_exp", output_file_tag="al")

    # print("AL: ", al_image_list)

    # rand_image_list = al_label_transfer.get_random_n_images(no_img=40)

    # d2_mask.label_predictions_on_images(
    #     image_list=rand_image_list,
    #     regist_instances=False,
    #     output_path="./output/0320_exp/imgs/rand",
    # )

    # al_label_transfer.move_labels_to_train(train_anns_file="./data/annotations/empty.json", image_list=rand_image_list, output_path="./output/0320_exp", output_file_tag="rand")

    # print("RAND: ", rand_image_list)
