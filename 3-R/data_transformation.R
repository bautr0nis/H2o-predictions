#install.packages("tidyverse")
library(tidyverse)
library(visdat)
library(dplyr)
getwd()

file_path <- ("1-data/1-sample_data.csv")
data <- read_csv(file_path)
data_additional <- read_csv("1-data/2-additional_data.csv")
data_feat <- read_csv("1-data/3-additional_features.csv")


#Joins
combined_df <- dplyr::bind_rows(data_additional, data)
df <- inner_join(combined_df, data_feat, by = "id")



#Writing procedure
write_csv(df, "1-data/train_data.csv")
