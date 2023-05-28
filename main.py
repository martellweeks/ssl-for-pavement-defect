from config import config, paths, setup
from scripts import experiments
from src.controller import d2_mask

if __name__ == "__main__":
    # experiments.exp_0519_long_cns_bigbatch_wronglabels()

    d2_mask.get_coco_eval_results_cns(
        model_weights="output/exp_0519/exp_0519_long_cns_bigbatch_wronglabels/model_0001999.pth",
        regist_instances=True,
        output_path="./eval/exp_0519/exp_0519_long_cns_bigbatch_wronglabels/model_0001999",
        cfg=config.get_cfg_for_cns(**setup.exp_0519_long_cns_bigbatch_wronglabels),
    )

    d2_mask.get_coco_eval_results_cns(
        model_weights="output/exp_0519/exp_0519_long_cns_bigbatch_wronglabels/model_0003999.pth",
        regist_instances=False,
        output_path="./eval/exp_0519/exp_0519_long_cns_bigbatch_wronglabels/model_0003999",
        cfg=config.get_cfg_for_cns(**setup.exp_0519_long_cns_bigbatch_wronglabels),
    )

    d2_mask.get_coco_eval_results_cns(
        model_weights="output/exp_0519/exp_0519_long_cns_bigbatch_wronglabels/model_0005999.pth",
        regist_instances=False,
        output_path="./eval/exp_0519/exp_0519_long_cns_bigbatch_wronglabels/model_0005999",
        cfg=config.get_cfg_for_cns(**setup.exp_0519_long_cns_bigbatch_wronglabels),
    )

    d2_mask.get_coco_eval_results_cns(
        model_weights="output/exp_0519/exp_0519_long_cns_bigbatch_wronglabels/model_0007999.pth",
        regist_instances=False,
        output_path="./eval/exp_0519/exp_0519_long_cns_bigbatch_wronglabels/model_0007999",
        cfg=config.get_cfg_for_cns(**setup.exp_0519_long_cns_bigbatch_wronglabels),
    )

    d2_mask.get_coco_eval_results_cns(
        model_weights="output/exp_0519/exp_0519_long_cns_bigbatch_wronglabels/model_0009999.pth",
        regist_instances=False,
        output_path="./eval/exp_0519/exp_0519_long_cns_bigbatch_wronglabels/model_0009999",
        cfg=config.get_cfg_for_cns(**setup.exp_0519_long_cns_bigbatch_wronglabels),
    )
