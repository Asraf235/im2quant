"""
im2quant — image-based property prediction using a YOLO backbone.

Quick start
-----------
Training::

    from im2quant.config import Config
    from im2quant.pipeline import load_csv, build_metadata, stratified_split
    from im2quant.train import tune_hyperparameters, train

    cfg = Config(image_dir="path/to/images", csv_file="results.csv",
                 output_dir="./runs/my_material")
    df = load_csv(cfg)
    meta = build_metadata(cfg, df)
    splits = stratified_split(meta, cfg)
    best_params = tune_hyperparameters(cfg, splits)
    ckpt = train(cfg, splits, best_params)

Inference::

    from im2quant.inference import predict_single_image

    result = predict_single_image(
        "runs/my_material/best_yolo26n_tuned.pt",
        "batch3.1.jpg",
        {"printlayers": 46, "printmaxpower(%)": 14,
         "printspeed(mm/s)": 14, "polishlayers": 3,
         "polishmaxpower(%)": 20},
    )
    print(result)  # {'R_ohm': ..., 'log10R': ..., 'p_low_cov': ...}
"""

__version__ = "0.1.0"
