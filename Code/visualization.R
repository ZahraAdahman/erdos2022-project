rm(list = ls())
setwd("C:/Users/xiang/Erdos/covexper/Data/")

needed_packages <- c("tidyverse", "sf", "Metrics", "ggcorrplot")
lapply(needed_packages, require, character.only = TRUE)

#################################
###### correlation heatmap ######
#################################

data_select <- read_csv("data_covexper.csv") %>%
  mutate(Death_Rate = Town_Death/Population) %>%
  select("# COVID19 Deaths" = "Town_Death", "COVID19 Death Rates" = "Death_Rate", 
         "% Population (Age>64)" = "Over65_Pct", "% Minority" = "Minority_Pct",
         "% Below High School Edu." = "BelowHS_Pct", "Median Gross Rent" = "Med_Gross_Rent", "Population Density" = "PD", 
         "% Occupation (High Risk)" = "Job_Pct_HRisk", "PM25 Average Conc." = "PM25", "Ozone Seasonal DM8HA" = "Ozone",
         "% High Occupancy Residence" = "CrowdHouse_Pct", "% Unemployed" = "Unemployed_Pct", 
         "NO2 Average Conc." = "NO2", "Respiratory Hazard Index" = "Resp_Risk", "NPL Site Proximity" = "NPL_Prox", 
         "Traffic Proximity" = "Traffic_Prox", "DOT Noise Level" = "Noise_Level", "SVI (Overall)" = "SVI_Overall")

corr_matrix <- round(cor(data_select, method = "spearman", use = "pairwise.complete.obs"), 2)
ggcorrplot(corr_matrix, lab = TRUE, colors = c("#6D9EC1", "white", "#E46726"))
# ggsave(filename = "../Figure/heatmap/heatmap.png", width = 12, height = 12, units = "in", dpi = 300)

idx <- sort(corr_matrix[1, ], decreasing = TRUE, index.return = TRUE)$ix
corr_matrix <- corr_matrix[, idx]
corr_matrix <- corr_matrix[idx, ]
ggcorrplot(corr_matrix, lab = TRUE, colors = c("#6D9EC1", "white", "#E46726"))
# ggsave(filename = "../Figure/heatmap/heatmap_ordered.png", width = 12, height = 12, units = "in", dpi = 300)

############################################
###### leave-one-out cross validation ######
############################################

data_select <- read_csv("Output/data_select.csv")
loocv <- drop_na(data_select, Town_Death) %>%
  filter(Population > 10, Death_Rate < 0.006) %>%
  select(c(12, 17)) %>%
  left_join(read_csv("Output/statistical_modeling/loocv_statistical-model.csv"), by = "OBJECT_ID") %>%
  left_join(read_csv("Output/loocv_rf.csv"), by = c("OBJECT_ID")) %>%
  left_join(read_csv("Output/loocv_xgboost.csv"), by = c("OBJECT_ID")) %>%
  mutate(Town_Death, Poisson, NB_BYM, RF = RF_Rate*Population, XGBOOST = XGBOOST_Rate*Population) %>%
  select(c(2:5, 10, 11))
print(rmse(loocv$Town_Death, loocv$XGBOOST))
print(cor(loocv$Town_Death, loocv$XGBOOST)^2)

#################################################
###### spatial distribution of death rates ######
#################################################

municipality_poly <- st_read("NJ_Municipality_Shapefile/NJ_Municipal_Boundaries_3424.shp") %>% 
  select(c(1:5, 24))
data_shp <- st_drop_geometry(municipality_poly) %>%
  mutate(OBJECTID = OBJECTID, County= str_to_title(COUNTY), Town = MUN_LABEL) %>%
  select(OBJECTID, County, Town)
data_prediction <- read_csv("Output/statistical_modeling/predict_statistical-model.csv") %>%
  left_join(read_csv("Output/predict_rf.csv"), by = c("OBJECT_ID", "Death_Rate")) %>%
  left_join(read_csv("Output/predict_xgboost.csv"), by = c("OBJECT_ID", "Death_Rate")) %>%
  mutate(Death_Rate = Death_Rate*10^5, Poisson_Rate = Poisson/Population*10^5, 
         NB_BYM_Rate = NB_BYM/Population*10^5, RF_Rate = RF_Rate*10^5, XGBOOST_Rate = XGBOOST_Rate*10^5) %>%
  arrange(OBJECT_ID) %>%
  bind_cols(data_shp) %>%
  select(c(1, 2, 5, 10, 11, 8, 9))

model_name <- paste0(c("Death", "Poisson", "NB_BYM", "RF", "XGBOOST"), "_Rate", sep = "")
col.br <- colorRampPalette(c("lightgoldenrodyellow", "lightgoldenrod1", "lightgoldenrod2",
                             "orange", "darkorange1", "orangered", "red", "red3", "darkred"))(10)  # color palette

for (i in 1:length(model_name)) {  
  data_prediction_sf <- bind_cols(municipality_poly, data_prediction) %>%
    mutate(variable_focus = get(model_name[i])) %>%
    mutate(category = case_when(is.na(variable_focus) ~ "a",
                                (variable_focus < 25) & (variable_focus >= 0) ~ 'b',
                                (variable_focus < 50) & (variable_focus >= 25) ~ 'c',
                                (variable_focus < 75) & (variable_focus >= 50) ~ 'd',
                                (variable_focus < 100) & (variable_focus >= 75) ~ 'e',
                                (variable_focus < 125) & (variable_focus >= 100) ~ 'f',
                                (variable_focus < 150) & (variable_focus >= 125) ~ 'g',
                                (variable_focus < 175) & (variable_focus >= 150) ~ 'h',
                                (variable_focus < 200) & (variable_focus >= 175) ~ 'i',
                                (variable_focus < 250) & (variable_focus >= 200) ~ 'j',
                                TRUE ~ 'k'))
  ggplot() +
    geom_sf(data = data_prediction_sf, aes(fill = category), size = 0.1, color = "black", inherit.aes = FALSE) +
    scale_fill_manual(values = c('a' = "lightgrey", 'b' = col.br[1], 'c' = col.br[2], 'd' = col.br[3],
                                 'e' = col.br[4], 'f' = col.br[5], 'g' = col.br[6], 'h' = col.br[7],
                                 'i' = col.br[8], 'j' = col.br[9], 'k' = col.br[10]),
                      labels = c('No Data','0-25', '25-50', '50-75', '75-100', '100-125',
                                 '125-150', '150-175', '175-200', '200-250', '>250')) +
    theme_void() +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          axis.ticks = element_blank(), axis.text = element_blank(),
          legend.title = element_text()) +
    labs(fill = "Death Rates\n(per 100k)")
}

##########################
###### Effects plot ######
##########################

## machine learning for reference covariates
xref <- c("Over65_Pct", "Minority_Pct", "BelowHS_Pct", "Med_Gross_Rent", "PD", 
          "Job_Pct_HRisk", "PM25", "Ozone", "CrowdHouse_Pct", "Unemployed_Pct") 
xref1 <- c("% Population (Age>64)", "% Minority", "% Below High School Edu.", 
           "Median Gross Rent", "Population Density", "% Occupation (High Risk)",
           "PM25 Average Conc.", "Ozone Seasonal DM8HA", "% High Occupancy Residence", 
           "% Unemployed")
model_name = c("rf", "xgboost")

x1 <- c(4, 0, 0, 850, 0, 10, 7, 38, 0, 2)  # plot setting: xlim, ylim and breaks
x2 <- c(38, 70, 30, 2800, 4.1, 28, 9.4, 45, 26, 15)
x3 <- c(0, 0, 0, 1000, 0, 10, 7, 38, 0, 0)
xbreak <- c(10, 15, 10, 500, 1, 5, 0.5, 2, 5, 5)
y1 <- c(rep(0, 4), 0, 0, 25, 0, 25, 0)
y2 <- c(520, 305, 440, 480, 265, 220, 230, 300, 250, 250)
y3 <- rep(0, 10)
ybreak <- c(100, 50, 100, 100, rep(50, 6))

for (i in 1:length(model_name)) {
  shap_df <- read_csv(paste0("Output/shap-effect_", model_name[i], "_basemodel.csv", sep = ""))
  shap_df[, 1:10] <- shap_df[, 1:10]*100000
  shap_df <- select(read_csv("Output/data_ref_unscaled.csv"), c("OBJECT_ID", "Town_Death", xref, "Population")) %>%
    left_join(shap_df, by = "OBJECT_ID")  
  
  for (j in 1:length(xref)) {
    data_plot <- shap_df[, which(names(shap_df) %in% c(xref[j], paste0(xref[j], "_ShapEffect", sep = "")))]
    names(data_plot) <- c("x", "y")
    if (xref[j] == "PD") {
      data_plot <- mutate(data_plot, x = case_when(x == 0 ~ "0-20%", x == 1 ~ "20%-40%",
                                                   x == 2 ~ "40%-60%", x == 3 ~ "60%-80%", TRUE ~ "80%-100%"))
      data_mean <- group_by(data_plot, x) %>%
        summarise(Mean = mean(y))
      ggplot(data = data_plot, aes(x = x, y = y)) +
        geom_jitter(position = position_jitter(0.15), col = "dodgerblue", alpha = 0.2) +  # mediumpurple
        geom_bar(data = data_mean, aes(x, Mean), stat = "identity", fill = "dodgerblue",
                 alpha = 0.2 , width = 0.4, inherit.aes = FALSE) +
        scale_x_discrete(name = xref1[j]) +
        scale_y_continuous(name = "Death Rate (Per 100k)", limits = c(y1[j], y2[j]),
                           breaks = seq(y3[j], y2[j], by = ybreak[j])) +
        theme_bw() +
        theme(panel.grid = element_blank())
      ggsave(paste0("../Figure/effect-plot/effect-plot_", model_name[i], "_", xref[j], ".png", sep = ""),
             width = 4, height = 3.2, units = "in", dpi = 300)
      next
    }
    
    ggplot(data = data_plot, aes(x = x, y = y)) +
      geom_point(col = "mediumpurple", alpha = 0.2) +
      geom_smooth(col = "mediumpurple", alpha = 1.1, se = FALSE, span = 0.7) +
      scale_x_continuous(name = xref1[j], limits = c(x1[j], x2[j]), breaks = seq(x3[j], x2[j], by = xbreak[j])) +
      scale_y_continuous(name = "Death Rate (Per 100k)", limits = c(y1[j], y2[j]),
                         breaks = seq(y3[j], y2[j], by = ybreak[j])) +
      theme_bw() +
      theme(panel.grid = element_blank())
    ggsave(paste0("../Figure/effect-plot/effect-plot_", model_name[i], "_", xref[j], ".png", sep = ""), 
           width = 4, height = 3.2, units = "in", dpi = 300)
    print(j)
  }
}

## machine learning for other covariates
model_name = c("rf", "xgboost")
xtest <- c( "NO2", "Resp_Risk", "NPL_Prox", "Traffic_Prox", "Noise_Level", "SVI_Overall")

for (i in 1:length(model_name)) {
  shap_df <- read_csv(paste0("Output/shap-effect_", model_name[i], "_other-covariates.csv", sep = ""))
  shap_df[, 2:7] <- shap_df[, 2:7]*100000
  shap_df <- select(read_csv("Output/data_ref_unscaled.csv"), c("OBJECT_ID", "Town_Death", xtest, "Population")) %>%
    left_join(shap_df, by = "OBJECT_ID")   
  
  for (j in 1:length(xtest)) {
    data_plot <- shap_df[, which(names(shap_df) %in% c(xtest[j], paste0(xtest[j], "_ShapEffect", sep = "")))]
    names(data_plot) <- c("x", "y")
    
    if (xtest[j] %in% c('NPL_Prox', 'Traffic_Prox','PD')) {
      data_plot <- mutate(data_plot, x = case_when(x == 0 ~ "0-20%", x == 1 ~ "20%-40%", 
                                                   x == 2 ~ "40%-60%", x == 3 ~ "60%-80%", TRUE ~ "80%-100%"))
      data_mean <- group_by(data_plot, x) %>%
        summarise(Mean = mean(y))
      ggplot(data = data_plot, aes(x = x, y = y)) +
        geom_jitter(position = position_jitter(0.15), col = "black", alpha = 0.2) +
        geom_bar(data = data_mean, aes(x, Mean), stat = "identity", fill = "black", 
                 alpha = 0.2 , width = 0.4, inherit.aes = FALSE) +
        # scale_x_discrete(name = "NO2 Average Concentration (ppb)") +
        # scale_y_continuous(name = "Death Rate (Per 100k)", limits = c(y1[j], y2[j]),
        #                    breaks = seq(y3[j], y2[j], by = ybreak[j])) +
        theme_bw() +
        theme(panel.grid = element_blank())
      ggsave(paste0("../Figure/effect-plot/effect-plot_", model_name[i], "_", xtest[j], ".png", sep = ""),
             width = 4.5, height = 3.6, units = "in", dpi = 300)
      next
    }
    
    ggplot(data = data_plot, aes(x = x, y = y)) +
      geom_point(col = "mediumpurple", alpha = 0.2) +
      geom_smooth(col = "mediumpurple", alpha = 1.1, se = FALSE, span = 0.7) +
      # scale_x_continuous(name = "SVI", limits = c(0, 0.8), breaks = seq(0, 0.8, by = 0.2)) +
      # scale_y_continuous(name = "SVI", limits = c(0, 400), breaks = seq(0, 400, by = 50)) +
      theme_bw() +
      theme(panel.grid = element_blank())
    ggsave(paste0("../Figure/effect-plot/effect-plot_", model_name[i], "_", xtest[j], ".png", sep = ""),
           width = 4, height = 3.2, units = "in", dpi = 300)
    print(j)
  }
}