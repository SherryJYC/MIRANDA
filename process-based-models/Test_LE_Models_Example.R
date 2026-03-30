#-------------------------------------------------
# Preamble: This script calibrates plant phenology
# models in phenor using ground observations 
# from MeteoSwiss with a selection of training
# and testing site-years based on a json file 
# specifying train-test splits.

#-------------------------------------------------
# I. Clear environment.
#-------------------------------------------------
gc()
rm(list=ls(all=TRUE))
#-------------------------------------------------
# II. Install and load libraries as necessary.
#-------------------------------------------------
if(!require('phenor')) install.packages('phenor'); library(phenor)
if(!require('stringr')) install.packages('stringr'); library(stringr)
if(!require('plyr')) install.packages('plyr'); library(plyr)
if(!require('dplyr')) install.packages('dplyr'); library(dplyr)
if(!require('RJSONIO')) install.packages('RJSONIO'); library(RJSONIO)

#-------------------------------------------------
# III. Define constants and functions.
#-------------------------------------------------
# Define path to folder where:
# 1. training data are stored in the form of ".RDS" files,
# 2. parameter range files are stored in the form of .csv files, 
# 3. train-test splits are stored in the form of JSON files. 
path_to_data = "data/PhenoFormer-data/process-models-data"
path_to_split_files = "splits"

# Define species of interest: 
# European_beech, Horse_chestnut, European_larch, Common_spruce, Hazel
sp = "Common_spruce"

# Here specify the training and testing split of interest, 
# [structured-temporal, hotyear-temporal, highelevation-spatial]
splits = c("highelevation-spatial", "hotyear-temporal", "structured-temporal")

for(split in splits){
  print(">>>>>> species: ")
  print(sp)
  print(">>>>>> in split: ")
  print(split)

  if(split == "structured-temporal"){
    training_config = "structured_temporal_split"
  }
  if(split == "hotyear-temporal"){
    training_config = "hotyear_temporal_split"
    json_file = paste0(path_to_split_files,"/hotyear-temporal-split.json")
    json_data = fromJSON(json_file)
  }
  if(split == "highelevation-spatial"){
    training_config = "highelevation_spatial_split"
    json_file = paste0(path_to_split_files,"/highelevation-spatial-split.json")
    json_data = fromJSON(json_file)
  }

  # Define path to phenor parameter ranges, with Tbase for forcing > 0°C, and t0 for 
  # forcing accumulation greater than Jan. 1
  par_ranges = paste0(path_to_data,"/parameter_ranges_spring.csv")

  # Define leaf emergence models of interest.
  spring_models = c("M1") #c("LIN","TT","M1","PTT","PTTs","AT","SQ","PA","DP","NULL")

  # Use Hufkens et al. (2018) parameterization approach:
  control = list(temperature = 10000, max.call = 40000)

  #-------------------------------------------------
  # IV. Load species training data.
  #-------------------------------------------------
  # Collect names of training lists.
  training_lists <- list.files(path_to_data, pattern = ".RDS")
  # Collect species order of training lists.
  species <- unlist(lapply(training_lists, function(x) str_split(str_split(x,"list_")[[1]][2],".RDS")[[1]][1]))
  # Collect phenophase order of training lists.
  phenophases <- unlist(lapply(training_lists, function(x) str_split(str_split(x,"ground_")[[1]][2],"_nested")[[1]][1]))
  # Filter to phenophase and species of interest.
  index = which(phenophases %in% c("leaf_unfolding","needle_emergence")&(species == sp))
  # Load phenor list with all phenology observations and daily data.
  pooled_list = readRDS(paste0(path_to_data,"/",training_lists[index]))

  # Load results
  path_to_output = "output/processed_models"
  print(">> use model")
  file = paste0(path_to_output,"/",training_config,"_",sp,"_optimal_params_LE_models.RDS")
  print(file)
  optimal_params <- readRDS(file)

  # Final output
  path_to_pred = "output/processed_models/preds"

  data = pooled_list
  list_items = names(data)
  for(i in 1:length(list_items)){
    item = data[[i]]
    name = list_items[i]
    if(name == "doy"){
      next()
    }
    if(length(item)%in% c(0,1)){
      next()
    }
    
    dims = dim(item)
    if(is.null(dims)){
      data[[i]] = item
      next()
    }
    if(dims[1]==1){
      item = item
    } else {
      item = item[,]
    }
    data[[i]] = item
  }

  filtered_data <- list(
      site = data$site,
      year = data$year,
      transition_dates = data$transition_dates,
      species = data$species,
      elevation = data$elevation,
      phenophases = data$phenophase
    )

  # Designate training and testing site-years based on train-test splits
  print("Testing")
  for(k in 1:10){
    # Train and Test
    for(model in spring_models){
      if(model != "NULL"){
        optim_par = optimal_params[[model]][[sp]][[k]]
        print(k)
        # Test
        out <- pr_predict(optim_par, data = data, model = model)
        col_name <- paste0("predf", k)
        filtered_data[[col_name]] <- out
        
        write.csv(filtered_data, file = paste0(path_to_pred,"/",training_config,"_",sp, "_M1_preds.csv"))
        print("Finished")
        
      } else {
        
        print("NO MODEL")
        
      }
      
  }

  }

}
