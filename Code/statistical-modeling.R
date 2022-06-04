# This code is to construct statistical models and quantify associations

rm(list = ls())
setwd("C:/Users/xiang/Erdos/covexper/Data/")

needed_packages <- c("tidyverse", "INLA")
lapply(needed_packages, require, character.only = TRUE)

##########################
###### process data ######
##########################

data_select <- read_csv("Output/data_select.csv")
data_model <- drop_na(data_select, Town_Death) %>%
  filter(Population > 10, Death_Rate < 0.006)  # 352*24
data_predict <- filter(data_select, !OBJECT_ID %in% data_model$OBJECT_ID)  # 213*24
data_cor <- as.data.frame(cor(data_model[, c(1:10, 18:23)]))
model_name <- c("poisson", "nbinomial.bym")
xref <- c("Over65_Pct", "Minority_Pct", "BelowHS_Pct", "Med_Gross_Rent", "PD",
          "Job_Pct_HRisk", "PM25", "Ozone", "CrowdHouse_Pct", "Unemployed_Pct") 
xtest <- c("NO2", "Resp_Risk", "NPL_Prox", "Traffic_Prox", "Noise_Level", "SVI_Overall")
xscale <- c("Over65_Pct", "Minority_Pct", "BelowHS_Pct", "Med_Gross_Rent", "PD",
            "Job_Pct_HRisk", "CrowdHouse_Pct", "Unemployed_Pct", "Resp_Risk", 
            "NPL_Prox", "Traffic_Prox", "Noise_Level", "SVI_Overall")

#######################
###### functions ######
#######################

fmla.function <- function(variable, model_type) {  # formula
  if (model_type %in% c("poisson")) {
    as.formula(paste("Town_Death ~ ", paste(variable, collapse = "+")))
  } else if (model_type %in% c("nbinomial.bym")) {
    update(as.formula(paste("Town_Death ~ ", paste(variable, collapse = "+"))), . ~ . + 
             f(OBJECT_ID, model = "bym", graph = "municipality_adjacency.graph"))
  }
}

mod.function <- function(fmla, data, model_type) {  # fit model
  model_family <- str_split(model_type, "[.]", simplify = TRUE)[1]
  if (model_family == "poisson") { 
    return(inla(fmla, data = data, family = "poisson", offset = log(Population),
                control.compute = list(dic = FALSE, cpo = FALSE)))
  } else if (model_family == "nbinomial") {
    return(inla(fmla, data = data, family = "nbinomial", offset = log(Population),
                control.compute = list(dic = FALSE, cpo = FALSE)))
  }
}

predict.function <- function(fmla, data, model_type, link) {
  model_family <- str_split(model_type, "[.]", simplify = TRUE)[1]
  if (model_family == "poisson") { 
    return(inla(fmla, data = data, family = "poisson", offset = log(Population),
                control.compute = list(dic = FALSE, cpo = FALSE),
                control.predictor = list(link = link)))
  } else if (model_family == "nbinomial") {
    return(inla(fmla, data = data, family = "nbinomial", offset = log(Population),
                control.compute = list(dic = FALSE, cpo = FALSE, config = TRUE),
                control.predictor = list(link = link)))
  }
}

cor.remove <- function(xname_add, xname_ref) {  # remove highly-correlated variables
  cor_select <- abs(data_cor[xname_add, xname_ref, drop = FALSE])
  return(c(xname_add, names(cor_select )[cor_select <= 0.6]))
}

generate_predictor_mean <- function(model_type, feature, num = 50) {  # predictor mean matrix
  feature_seq <- seq(min(data_concat[, names(data_concat) == feature]),
                     max(data_concat[, names(data_concat) == feature]), length.out = num)  # range for pep
  predictor_mean <- select(data_concat, c("Town_Death", xname, "Population")) %>%
    apply(2, mean) %>%
    t() %>%
    as.data.frame() %>%
    .[rep(1, num), ]  # copy multiple rows
  predictor_mean$Town_Death <- NA
  predictor_mean[, feature] <- feature_seq
  if (model_type %in% c("poisson")) {
    data_fit <- data_model %>%
      select(c("Town_Death", xname, "Population")) %>%
      bind_rows(predictor_mean)
  } else if (model_type %in% c("nbinomial.bym")) {
    predictor_mean <- predictor_mean[rep(1:num, rep(nrow(data_model), num)), ] %>%
      mutate(OBJECT_ID = rep(unique(data_model$OBJECT_ID), num))
    data_fit <- data_model %>%
      select(c("Town_Death", xname, "Population", "OBJECT_ID")) %>%
      bind_rows(predictor_mean)
  }
  return(data_fit)  
}

calculate_effect <- function(model_type, feature, num = 50) {  # predictor effects
  if (model_type %in% c("poisson")) {
    effect_df <- mod[["summary.fitted.values"]] %>%
      slice((nrow(.)-num+1):nrow(.)) %>%
      mutate(Feature = data_fit[[feature]][(nrow(data_fit)-num+1):nrow(data_fit)],
             Mean = mean/population_mean*100000, Lower = `0.025quant`/population_mean*100000,
             Upper = `0.975quant`/population_mean*100000) %>%
      select(Feature, Mean, Lower, Upper)
  } else if (model_type %in% c("nbinomial.bym")) {
    effect_df <- mod[["summary.fitted.values"]] %>%
      slice((nrow(.)-num*nrow(data_model)+1):nrow(.)) %>%
      mutate(Feature = data_fit[[feature]][(nrow(data_fit)-num*nrow(data_model)+1):nrow(data_fit)],
             Mean = mean/population_mean*100000, Lower = `0.025quant`/population_mean*100000,
             Upper = `0.975quant`/population_mean*100000) %>%
      group_by(Feature) %>%
      summarise(Mean = quantile(Mean, 0.5), Lower = quantile(Lower, 0.5), Upper = quantile(Upper, 0.5))
  }
  return(effect_df)
}

############################################
###### leave-one-out cross validation ######
############################################

loocv_statistical <- data.frame(matrix(NA, nrow(data_model), 4))
names(loocv_statistical) <- c("OBJECT_ID", "Town_Death", "Poisson", "NB_BYM")
for (i in 1:nrow(data_model)) {
  data_validate <- data_model
  data_validate$Town_Death[i] <- NA
  link <- rep(NA, nrow(data_model))
  link[which(is.na(data_validate$Town_Death))] <- 1
  fmla1 <- fmla.function(xref, model_name[1])
  mod1 <- predict.function(fmla1, data_validate, model_name[1], link)
  fmla2 <- fmla.function(xref, model_name[2])
  mod2 <- predict.function(fmla2, data_validate, model_name[2], link)
  loocv_statistical$OBJECT_ID[i] <- data_model$OBJECT_ID[i]
  loocv_statistical$Town_Death[i] <- data_model$Town_Death[i]
  loocv_statistical$Poisson[i] <- mod1$summary.fitted.values$mean[i]
  loocv_statistical$NB_BYM[i] <- mod2$summary.fitted.values$mean[i]
  print(i)
}
# write_csv(loocv_statistical, file = "Output/statistical_modeling/loocv_statistical-model.csv")

########################
###### prediction ######
########################

data_concat <- rbind(data_model, data_predict)
link <- rep(NA, nrow(data_concat))
link[(nrow(data_model)+1):nrow(data_concat)] <- 1 
fmla1 <- fmla.function(xref, model_name[1])
mod1 <- predict.function(fmla1, data_concat, model_name[1], link)
fmla2 <- fmla.function(xref, model_name[2])
mod2 <- predict.function(fmla2, data_concat, model_name[2], link)
pred <- data.frame(Poisson = mod1$summary.fitted.values$mean, NB_BYM = mod2$summary.fitted.values$mean)
pred <- cbind(data_concat[, c(17, 15, 12, 11, 24)], pred) 
# write_csv(pred, file = "Output/statistical_modeling/predict_statistical-model.csv")

#####################################################
###### interpretation (predictor effects plot) ######
#####################################################

data_concat <- rbind(data_model, data_predict)
data_ref <- read_csv("Output/data_ref_unscaled.csv")
xref1 <- c("% Population (Age>64)", "% Minority", "% Below High School Edu.",
           "Median Gross Rent", "Population Density", "% Occupation (High Risk)",
           "PM25 Average Conc.", "Ozone Seasonal DM8HA", 
           "% High Occupancy Residence", "% Unemployed")

x1 <- c(4, 0, 0, 850, 0, 10, 7, 38, 0, 2)  # plot setting: xlim, ylim and breaks
x2 <- c(38, 70, 30, 2800, 4.1, 28, 9.4, 45, 26, 15)
x3 <- c(0, 0, 0, 1000, 0, 10, 7, 38, 0, 0)
xbreak <- c(10, 15, 10, 500, 1, 5, 0.5, 2, 5, 5)
y1 <- c(rep(0, 4), 0, 0, 25, 0, 25, 0)
y2 <- c(520, 305, 440, 480, 265, 220, 230, 300, 250, 250)
y3 <- rep(0, 10)
ybreak <- c(100, 50, 100, 100, rep(50, 6))

## reference covariates
feature_mean <- apply(data_ref[, xref], 2, mean)
feature_sd <- apply(data_ref[, xref], 2, sd)
xname <- xref
for (i in 1:length(model_name)) {
  for (j in 1:length(xref)) {
    data_fit <- generate_predictor_mean(model_type = model_name[i], feature = xref[j])  # include fitting/predicting data
    link <- rep(NA, nrow(data_fit))  # former rows for fitting, latter 50 rows for making effects plot
    link[which(is.na(data_fit$Town_Death))] <- 1 
    fmla <- fmla.function(xref, model_name[i])
    mod <- predict.function(fmla, data_fit, model_name[i], link)
    population_mean <- data_fit$Population[nrow(data_fit)]  # population average
    effect_df <- calculate_effect(model_type = model_name[i], feature = xref[j])
    
    if (xref[j] %in% xscale) {
      effect_df$Feature <- feature_sd[j] * effect_df$Feature + feature_mean[j]
    }  # transform to original scale
    
    if (xref[j] == "PD") {
      effect_category <- data.frame()
      category <- c("0-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%")
      for (k in 0:4) {  # five levels
        idx <- which.min(abs(effect_df$Feature-k))
        data_row <- effect_df[idx, ] 
        data_row$Feature <- category[k+1]
        effect_category <- rbind(effect_category, data_row)
      }
      # write_csv(effect_category, file = paste0("Output/statistical_modeling/effect-df_",
      #                                          model_name[i], "_", xref[j], ".csv", sep = ""))
      ggplot(data = effect_category, aes(x = Feature, y = Mean)) +
        geom_bar(stat = "identity", fill = "goldenrod", alpha = 0.2 , width = 0.4) + # goldenrod, mediumseagreen
        geom_errorbar(aes(ymin = Lower, ymax = Upper), col = "goldenrod", width = 0.08, size = 1.1) +
        scale_x_discrete(name = xref1[j]) +
        scale_y_continuous(name = "Death Rate (Per 100k)", limits = c(y1[j], y2[j]), 
                           breaks = seq(y3[j], y2[j], by = ybreak[j])) +
        theme_bw() +
        theme(panel.grid = element_blank())
      ggsave(paste0("../Figure/effect-plot/effect-plot_", model_name[i], "_", xref[j], ".png", sep = ""), 
             width = 4, height = 3.2, units = "in", dpi = 300)
      next
    }
    
    ggplot(data = effect_df, aes(x = Feature, y = Mean)) +
      geom_line(col = "mediumseagreen", size = 1.1) +
      geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "mediumseagreen", alpha = 0.2) +
      scale_x_continuous(name = xref1[j], limits = c(x1[j], x2[j]), breaks = seq(x3[j], x2[j], by = xbreak[j])) +
      scale_y_continuous(name = "Death Rate (Per 100k)", limits = c(y1[j], y2[j]),
                         breaks = seq(y3[j], y2[j], by = ybreak[j])) +
      theme_bw() +
      theme(panel.grid = element_blank())
    # write_csv(effect_df, file = paste0("Output/statistical_modeling/effect-df_",
    #                                    model_name[i], "_", xref[j], ".csv", sep = ""))
    ggsave(paste0("../Figure/effect-plot/effect-plot_", model_name[i], "_", xref[j], ".png", sep = ""), 
           width = 4, height = 3.2, units = "in", dpi = 300)
    print(j)
  }
}

## other covariates
feature_mean <- apply(data_ref[, xtest], 2, mean)
feature_sd <- apply(data_ref[, xtest], 2, sd)

for (i in 1:2) {
  for (j in 1:length(xtest)) {
    xname <- cor.remove(xname_add = xtest[j], xname_ref = xref)
    data_fit <- generate_predictor_mean(model_type = model_name[i], feature = xtest[j]) 
    link <- rep(NA, nrow(data_fit)) 
    link[which(is.na(data_fit$Town_Death))] <- 1 
    fmla <- fmla.function(xname, model_name[i])
    mod <- predict.function(fmla, data_fit, model_name[i], link)
    population_mean <- data_fit$Population[nrow(data_fit)]  # population average
    effect_df <- calculate_effect(model_type = model_name[i], feature = xtest[j])
    
    if (xtest[j] %in% xscale) {
      effect_df$Feature <- feature_sd[j] * effect_df$Feature + feature_mean[j]
    }
    
    if (xtest[j] %in% c('NPL_Prox', 'Traffic_Prox','PD')) {
      effect_category <- data.frame()
      category <- c("0-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%")
      for (k in 0:4) {  # five levels
        idx <- which.min(abs(effect_df$Feature-k))
        data_row <- effect_df[idx, ] 
        data_row$Feature <- category[k+1]
        effect_category <- rbind(effect_category, data_row)
      }
      # write_csv(effect_category, file = paste0("Output/statistical_modeling/effect-df_", 
      #                                          model_name[i], "_", xtest[j], ".csv", sep = ""))
      ggplot(data = effect_category, aes(x = Feature, y = Mean)) +
        geom_bar(stat = "identity", fill = "black", alpha = 0.2 , width = 0.4) + 
        geom_errorbar(aes(ymin = Lower, ymax = Upper), col = "black", width = 0.06, size = 1.1) +
        # scale_x_continuous(name = "NO2 Average Concentration (ppb)", limits = c(10, 30), breaks = seq(10, 30, by = 5)) +
        # scale_y_continuous(name = "Death Rate (Per 100k)", limits = c(0, 250), breaks = seq(0, 250, by = 50)) +
        theme_bw() +
        theme(panel.grid = element_blank())
      ggsave(paste0("../Figure/effect-plot/effec-plot_", model_name[i], "_", xtest[j], ".png", sep = ""),
             width = 4.5, height = 3.6, units = "in", dpi = 300)
      print(j)
      next
    }
    
    ggplot(data = effect_df, aes(x = Feature, y = Mean)) +
      geom_line(col = "goldenrod", size = 1.1) +  # dodgerblue, mediumseagreen
      geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "goldenrod", alpha = 0.2) +
      # scale_x_continuous(name = "SVI", limits = c(0, 0.8), breaks = seq(0, 0.8, by = 0.2)) +
      # scale_y_continuous(name = "SVI", limits = c(0, 400), breaks = seq(0, 400, by = 50)) +
      theme_bw() +
      theme(panel.grid = element_blank())
    # write_csv(effect_df, file = paste0("Output/statistical_modeling/effect-df_",
    #                                    model_name[i], "_", xtest[j], ".csv", sep = ""))
    ggsave(paste0("../Figure/effect-plot/effect-plot_", model_name[i], "_", xtest[j], ".png", sep = ""),
           width = 4, height = 3.2, units = "in", dpi = 300)
    print(j)
  }
}