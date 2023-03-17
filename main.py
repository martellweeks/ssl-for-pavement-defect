from src.model import d2_mask
from src.scores import al_scoring

if __name__ == "__main__":
    # d2_mask.train_model_only(output_folder="0313_model_2")

    # d2_mask.train_scores_only(
    #     output_folder="0317_score_2",
    #     model_weights="./output/0313_model/model_0000999.pth",
    #     regist_instances=True,
    # )

    d2_mask.predict_scores(
        model_weights="./output/0317_score_2/model_0001999.pth",
        output_path="./output/0317_preds",
        regist_instances=True,
    )

    al_scoring.al_score_calculation(
        file_path="./output/0317_preds/loss_predictions.csv",
        output_path="./output/0317_preds",
    )

    image_list = al_scoring.get_top_n_images(
        score_file_path="./output/0317_preds/al_score.csv", no_img=20
    )

    d2_mask.label_predictions_on_images(
        image_list=image_list,
        regist_instances=False,
        output_path="./output/0317_preds",
    )
