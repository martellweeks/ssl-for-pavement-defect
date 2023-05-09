from config import config, paths, setup
from src.data import al_label_transfer
from src.model import d2_mask
from src.scores import al_scoring

if __name__ == "__main__":
    d2_mask.predict_scores(
        model_weights="output/A14_L2/0414_init_score/model_final.pth",
        output_path="./tests",
        regist_instances=True,
        test_anns_file="./data/annotations/A14_L2/quick_test.json",
    )

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
