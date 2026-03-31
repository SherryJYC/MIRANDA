<div align="center">




# 🍀🌿🌱 MIRANDA 🌱🌿🍀<br>
# MId-feature RANk-adversarial Domain Adaptation toward climate change-robust ecological forecasting with deep learning

CVPR EarthVision workshop 2026
<br>
<br>
[![arXiv](https://img.shields.io/badge/arxiv-xxxxxx-b31b1b.svg)](placeholder)
[![Project page](https://img.shields.io/badge/Project_page-8A2BE2)](https://sherryjyc.github.io/MIRANDA/)
<br>
<br>
</div>

<div style="background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007acc; font-style: italic; font-size: 14px; line-height: 1.5;">
Plant phenology modelling aims to predict the timing of seasonal phases, such as leaf-out or flowering, from meteorological time series. Reliable predictions are crucial for anticipating ecosystem responses to climate change. While phenology modelling has traditionally relied on mechanistic approaches, deep learning methods have recently been proposed as flexible, data-driven alternatives with often superior performance. However, mechanistic models tend to outperform deep networks when data distribution shifts are induced by climate change. Domain Adaptation (DA) techniques could help address this limitation. Yet, unlike standard DA settings, climate change induces a tem- poral continuum of domains and involves both a covariate and label shift, with warmer records and earlier start of spring. To tackle this challenge, we introduce Mid-feature Rank-adversarial Domain Adaptation (MIRANDA). Whereas conventional adversarial methods enforce domain invariance on final latent representations, an approach that does not explicitly address label shift, we apply adversarial regularization to intermediate features. Moreover, instead of a binary domain-classification objective, we employ a rank-based objective that enforces year-invariance in the learned meteorological representations. On a country-scale dataset spanning 70 years and comprising 67,800 phenological observations of 5 tree species, we demonstrate that, unlike conventional DA approaches, MIRANDA improves robustness to climatic distribution shifts and narrows the performance gap with mechanistic models.
</div>

## The proposed method: MIRANDA 🍀

<div style="display: flex; align-items: center;">
  <div style="flex: 1;">
    <!-- Left side text -->
    MIRANDA is an architecture designed for species-level phenology forecasting. It takes daily climatic time series as input, and predicts the date of occurence of a given phenophase. 
    The backbone architecture is PhenoFormer, which is a transformer model designed to predict the phenophases of multiple species at the same time. The proposed MIRANDA addresses domain shifts in phenology modelling via two key components: rank-based adversarial training on intermediate features (green parts) and hybrid layer normalization (blue parts).
  </div>
  <div style="flex: 1; text-align: center;">
    <!-- Right side image -->
    <img src="https://github.com/SherryJYC/MIRANDA/blob/main/docs/static/images/method.png" alt="method overview" style="width: 1000px" class="center"/>
  </div>
</div>

## Setting up 📦

#### ⬇️ Download the dataset 
The dataset is from [PhenoFormer work](https://github.com/VSainteuf/PhenoFormer/tree/v1.0).
You can retrieve our dataset from the [Zenodo archive](https://zenodo.org/records/15045780). This dataset contains two subfolders: one version of the dataset formatted for R scripts and the other one for python scripts. Please use the `learning-models-data` subfolder for all python scripts. 

#### 🧑‍💻 Clone the repository and install requirements
```
git clone git@github.com:SherryJYC/MIRANDA.git
cd MIRANDA
conda create --name miranda python==3.10
conda activate miranda
pip install -r requirements.txt
```
We recommend to create a new virtual environment with `python==3.10` :

> ⚠️ If you run into issues , make sure that your pip version is < 24.1 by running:

```setup
pip install pip==24.0
```

#### ⚙ Hardware

For the deep learning scripts, we recommend using a machine with GPU to have reasonable training times. Our models are still quite small (for deep learning standards) so a small GPU of even 4 or 8GB VRAM would do. 

## Run experiments :fire:

#### 👟 Training 

Main script to run the configurations of MIRANDA:
- `run-phenoformer-multispecies-spring.py` to train the multi-species variants for spring phenology.

To run the main script:
1. Complete the `data_folder` field with the path to the `learning-models-data` dataset folder on your machine. 
2. Complete the `save_dir` field with the path to the folder where to write the results. 
3. Activate the proper python environement and run the script.

#### :robot: Model variants
Different models used in the paper are defined in `model_configs` in `configs/RUN_CONFIGS.py`, you can choose the model name and put in `dict_model_to_do_list` in `run-phenoformer-multispecies-spring.py`.
- the proposed method: `MIRANDA`
- baselines: `dann`, `adda`, `adaBN`, `CORAL`
- ablations: `dann_shallow` (dann on mid-feature), `dann_shallow_rank_cos` (dann on mid-feature + rank loss), `dann_shallow_daln_nf` (dann on mid-feature + hybrid Norm)

#### :open_file_folder: Dataset variants
There are three datasets used in the paper, and you can choose which dataset to use inside `run-phenoformer-multispecies-spring.py`.
- `structured_temporal`: Chronological in the paper.
- `hotyear_temporal`: Annual temperature in the paper.
- `highelevation_spatial`: Elevation in the paper.

## 📯 Credits 

To cite this work please use:
```bibtex
@article{jiang2026miranda,
          title={MIRANDA: MId-feature RANk-adversarial Domain Adaptation toward climate change-robust ecological forecasting with deep learning},
          author={Jiang, Yuchang and Wegner, Jan Dirk and Garnot, Vivien Sainte Fare},
          journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
          year={2026},
        }
```

- Data source: Federal Office of Meteorology and Climatology (MeteoSwiss)  
- Meteorological data processing: Swiss Federal Institute for Forest, 
Snow and Landscape Research (WSL)
- Codes are largely adopted from [PhenoFormer work](https://github.com/VSainteuf/PhenoFormer/tree/v1.0).
