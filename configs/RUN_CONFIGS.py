from pathlib import Path

training_configs = dict(
    default=dict(
        loss="L2",
        optim="adam",
        learning_rate=0.0001,
        batch_size=16,
        sigma_jitter=0.05,
        max_epochs=300,
        gpus=1,
        nan_value_target=-1000,
        adapt_epochs=300,
    ),
)


datasplit_configs = dict(
    hotyear_temporal=dict(
        split_mode=str(
            Path(__file__).resolve().parent.parent / "splits/hotyear-temporal-split.json"
        ),
    ),
    highelevation_spatial=dict(
        split_mode=str(
            Path(__file__).resolve().parent.parent / "splits/highelevation-spatial-split.json"
        ),
    ),
    smoothelevation_spatial=dict(
        split_mode=str(
            Path(__file__).resolve().parent.parent / "splits/smoothelevation-spatial-split.json"
        ),
    ),
    uniformly_rdm=dict(
        split_mode=str(
            Path(__file__).resolve().parent.parent / "splits/uniformly-rdm-split.json"
        ),
    ),
    rdm_spatial=dict(
        split_mode=str(
            Path(__file__).resolve().parent.parent / "splits/rdm-spatial-split.json"
        ),
    ),
    rdm_temporal=dict(
        split_mode=str(
            Path(__file__).resolve().parent.parent / "splits/rdm-temporal-split.json"
        ),
    ),
    structured_temporal=dict(
        split_mode="structured", train_years_to=2002, val_years_to=2012
    ),
)
model_configs = dict(
    # baselines
    phenoformer_default=dict(
        pheno_model="PhenoFormer", 
        n_layers=2, 
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    # adaptations
    dann=dict(
        gan_loss_type="gan",
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp", # [mlp, transformer]
        pheno_model="PhenoFormer",
        n_layers=2,
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    adda=dict(
        gan_loss_type="gan",
        adapt=True,
        adapt_from_scratch=False,
        use_pretrained=True,
        adapter="adda",
        discriminator_type="mlp", # [mlp, transformer]
        pheno_model="PhenoFormer",
        n_layers=2,
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    adaBN=dict(
        adaBN=True,
        gan_loss_type="no_domain_loss",
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp", # [mlp, transformer]
        pheno_model="PhenoFormer",
        n_layers=2,
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    CORAL=dict(
        use_CORAL=True,
        gan_loss_type="no_domain_loss",
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp", # [mlp, transformer]
        pheno_model="PhenoFormer",
        n_layers=2,
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow=dict(
        gan_loss_type="gan",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_year=dict(
        gan_loss_type="year",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_cls_year_temp=dict(
        gan_loss_type="cls_year_temp",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_rank=dict(
        gan_loss_type="rank",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_rank_elevation=dict(
        rank_label="elevation_normalised",
        gan_loss_type="rank",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_rank_cos_elevation=dict(
        rank_label="elevation_normalised",
        rank_feature_sim="cos",
        gan_loss_type="rank",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_rank_cos=dict(
        rank_feature_sim="cos",
        gan_loss_type="rank",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_daln=dict( # original dann, with daln layers in both TE layers
        norm_type="danl",
        norm_first=True,
        gan_loss_type="gan",
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp", # [mlp, transformer]
        pheno_model="PhenoFormer",
        n_layers=2,
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_rank_cos_daln=dict(
        adapt_norm_type="danl",
        rank_feature_sim="cos",
        gan_loss_type="rank",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    MIRANDA=dict( # dann_shallow_rank_cos_daln_nf
        adapt_norm_type="danl",
        adapt_norm_first=True,
        rank_feature_sim="cos",
        gan_loss_type="rank",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_daln_nf=dict( # FINAL OUR METHOD without L_rank
        adapt_norm_type="danl",
        adapt_norm_first=True,
        gan_loss_type="gan",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_rank_cos_daln_nf_yeartemp=dict(
        rank_label="year_temp",
        adapt_norm_type="danl",
        adapt_norm_first=True,
        rank_feature_sim="cos",
        gan_loss_type="rank",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_rank_cos_daln_nf_elevation=dict(
        rank_label="elevation_normalised",
        adapt_norm_type="danl",
        adapt_norm_first=True,
        rank_feature_sim="cos",
        gan_loss_type="rank",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),

    dann_shallow_rank_cos_residual=dict(
        residual=True,
        rank_feature_sim="cos",
        gan_loss_type="rank",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_rank_cls=dict(
        rank_label="cls_year_temp",
        gan_loss_type="rank",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_elevation=dict(
        gan_loss_type="elevation",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
    dann_shallow_yeartemp=dict(
        gan_loss_type="year_temp",
        shallow=True,
        adapt=True,
        adapt_from_scratch=True,
        adapter="dann",
        discriminator_type="mlp",
        pheno_model="PhenoFormer",
        n_layers=1, # this is layer used by both phenoformer and dann
        nhead=8,
        d_model=64,
        dim_feedforward=256,
    ),
)
