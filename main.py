from src.model import d2_mask
from src.scores import al_scoring

if __name__ == "__main__":
    # d2_mask.train_model_only(output_folder="0310_model_2")
    # d2_mask.train_scores_only(
    #     output_folder="0310_score_2",
    #     model_weights="./output/0310_model_2/model_0001999.pth",
    #     regist_instances=True,
    # )
    d2_mask.predict(
        model_weights="./output/0310_score_2/model_0001499.pth",
        output_path="./output/0310_preds_2",
    )
    al_scoring.al_score_calculation(
        file_path="./output/0310_preds_2/loss_predictions.csv", output_path="./output/"
    )
    image_list = al_scoring.get_top_n_images(
        score_file_path="./output/0310_preds_2/al_score.csv", no_img=20
    )
    d2_mask.label_predictions_on_images(
        image_list=image_list,
        regist_instances=False,
        output_path="./output/0310_preds_2",
    )
