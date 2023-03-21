from config import config, paths, setup
from src.data import al_label_transfer
from src.model import d2_mask
from src.scores import al_scoring


def exp_0322(mode: str = "AL"):
    """
    Experiment on Mar 22
    Testing active learning pipeline for A14 Lane 2 Pavement data
    Against random sampling control
    """
    d2_mask.train_model_only(
        output_folder="0322_init_model",
        regist_instances=True,
        cfg=config.get_cfg_with_exp_setup(**setup.initial_model),
    )

    d2_mask.get_coco_eval_results(
        model_weights=paths.final_model_full_path,
        regist_instances=False,
        output_path="./output/0322_init_model",
    )

    d2_mask.train_scores_only(
        output_folder="0322_init_score",
        model_weights=paths.final_model_full_path,
        regist_instances=False,
        cfg=config.get_cfg_with_exp_setup(**setup.initial_score),
    )

    d2_mask.predict_scores(
        model_weights=paths.final_model_full_path,
        output_path="./output/0322_init_score",
        regist_instances=False,
    )

    al_scoring.calculate_al_score(
        file_path="./output/0322_init_score/loss_predictions.csv",
        output_path="./output/0320_init_score",
    )

    if mode == "AL":
        update_image_list = al_scoring.get_top_n_images(
            score_file_path="./output/0322_init_score/al_score.csv", no_img=40
        )
    else:
        update_image_list = al_label_transfer.get_random_n_images(
            test_anns_file="./data/annotations/A14_L2/test.json", no_img=40
        )

    al_label_transfer.move_labels_to_train(
        train_anns_file="./data/annotations/A14_L2/train.json",
        image_list=update_image_list,
        output_path="./output/0322_labels",
        output_file_tag="0",
    )

    for it in range(10):
        d2_mask.register_new_coco_instance(
            annotation_path=f"./output/0322_labels/train_{it}.json",
            data_path=paths.raw_data_path,
            tag=f"train_{it}",
        )

        d2_mask.register_new_coco_instance(
            annotation_path=f"./output/0322_labels/test_{it}.json",
            data_path=paths.raw_data_path,
            tag=f"test_{it}",
        )

        d2_mask.train_model_only(
            output_folder=f"0322_model_{it}",
            regist_instances=False,
            cfg=config.get_cfg_with_exp_setup(
                train_dataset=f"train_{it}", **setup.cycle_model
            ),
        )

        # Evaluation of model always with the initial unlabeled set
        d2_mask.get_coco_eval_results(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path=f"./output/0322_model_{it}",
        )

        d2_mask.train_scores_only(
            output_folder=f"0322_score_{it}",
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            cfg=config.get_cfg_with_exp_setup(
                train_dataset=f"train_{it}", **setup.cycle_score
            ),
        )

        d2_mask.predict_scores(
            model_weights=paths.final_model_full_path,
            output_path=f"./output/0322_score_{it}",
            regist_instances=False,
        )

        if mode == "AL":
            update_image_list = al_scoring.get_top_n_images(
                score_file_path=f"./output/0322_score_{it}/al_score.csv", no_img=40
            )
        else:
            update_image_list = al_label_transfer.get_random_n_images(
                test_anns_file=f"./output/0322_labels/test_{it}", no_img=40
            )

        update_image_list = al_scoring.get_top_n_images(
            score_file_path=f"./output/0322_score{it}/al_score.csv", no_img=40
        )

        al_label_transfer.move_labels_to_train(
            train_anns_file=f"./output/0322_labels/train_{it}.json",
            image_list=update_image_list,
            output_path="./output/0322_labels",
            output_file_tag=f"{it+1}",
        )


if __name__ == "__main__":
    # exp_0322(mode="AL")

    imlist = al_label_transfer.get_random_n_images(
        test_anns_file="./data/annotations/A14_L2/raw.json", no_img=200
    )

    al_label_transfer.move_labels_to_train(
        image_list=imlist,
        output_path="./data/annotations/A14_L2",
        output_file_tag=None,
        train_anns_file="./data/annotations/A14_L2/metadata.json",
        test_anns_file="./data/annotations/A14_L2/raw.json",
    )

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
    #     score_file_path="./output/0320_exp/al_score.csv", no_img=40
    # )

    # d2_mask.label_predictions_on_images(
    #     image_list=al_image_list,
    #     regist_instances=True,
    #     output_path="./output/0320_exp/imgs/al",
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
