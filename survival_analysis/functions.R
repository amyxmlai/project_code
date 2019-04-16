# Objective: Functions to perform various operations & computations in
# survival_analyses.rmd file

# Get column indices 
# Args: cols is a vector of column names; df is a dataframe 
# Returns: a vector of indices corresponding to specified column names
## ---- get_indices
get_indices = function(cols, df){
  indices = c()
  for (c in 1:length(cols)){
    indices[c] = which(colnames(df) == cols[c])
  }
  return(indices)
}
## ---- end-of-get_indices

#test_indices = get_indices(factor_vars, final_df)

# Compute contingency table 
# Args: x is row variable, y is column variable; df is a dataframe 
# Prints: contingency table
## ---- contingency_table
contingency_table = function(x, y, df){ 
  indices = get_indices(c(x, y), df)
  print(table(df[, indices[1]], df[, indices[2]], dnn = list(x, y)))
}
## ---- end-of-contingency_table

#contingency_table(factor_vars, y = 'mooddis_status_cox', df = final_df)

# See number of missing values 
# Args: col is a column
# Returns: number of missing values in the column 
## ---- get_missing
get_missing = function(col){
  return(sum(is.na(col)))
}
## ---- end-of-get_missing

# Estimate Kaplan-Meir survival function (unadjusted)
# Args: start & end are start & end times of the study; event is censorship status; 
# group is variable to be used for comparing survival functions; df is a dataframe
# Returns: list of results, where each result is a dataframe of times & probabilities
# from Kaplan-Meir survival model for a specific level of group variable
## ---- km_mod
km_mod = function(start, end, event, group, df){
  
  # Get indices 
  indices = get_indices(c(start, end, event, group), df)
  
  # Estimate survival function
  km = Surv(time = df[, indices[1]], time2 = df[, indices[2]],
            event = df[, indices[3]])
  mod = summary(survfit(km ~ df[, indices[4]], data = df))
  
  # Store results
  results = list()
  groups = levels(mod$strata)
  for (i in 1:length(groups)){
    time = mod$time[mod$strata == groups[i]]
    prob = mod$surv[mod$strata == groups[i]]
    results[[i]] = data.frame(time, prob)
  }
  return(results)
}
## ---- end-of-km_mod

#test_km_mod = km_mod(start = 'age_start_round', end = 'age_end_mooddis_round', 
                     #event = 'mooddis_status_cox', df = final_df)

# Plot (1 - survival function)
# Args: km_df is list of dataframes, where each dataframe contains times & probabilities 
# from Kaplan-Meir survival model; ylim is y-axis range; position is legend position;
# groups is legend text; title is plot title 
# Shows: plot
## ---- plot_survival
plot_survival = function(km_df, xlim, ylim, position, groups, title){
  
  # Set-up plot 
  colors = c('blue', 'red', 'dark green', 'black')
  line_types = c('solid', 'dashed', 'dotdash')
  plot(range(km_df[[1]]$time), range(1 - km_df[[1]]$prob), type = 'n', 
       xlim = xlim, ylim = ylim, las = 1, xlab ='Age (years)', 
       ylab = 'Probability', main = paste('Probability of', title,
                                          'by dementia status', sep = ' '))
  
  # Add lines
  for (i in 1:length(km_df)){
    lines(km_df[[i]]$time, 1 - km_df[[i]]$prob, lty = line_types[i], 
          col = colors[i], lwd = 2)
  }

  # Add Legend
  legend(position, legend = groups, lty = line_types, col = colors, lwd = 2)
}
## --- end-of-plot_survival

#plot_survival(test_km_mod, ylim = c(0,1), position = 'bottomright', 
              #groups = c('no dementia', 'dementia', title = 'mood')

# Fit Cox proportional hazards model 
# Args: start & end is start & end times of study; event is censorship status;
# df is a dataframe
# Returns: summary of model object 
## --- cox_mod
cox_mod = function(start, end, event, df){
  
  # Get indices
  indices = get_indices(c(start, end, event), df)
  
  # Fit model 
  km = Surv(time = df[, indices[1]], time2 = df[, indices[2]],
            event = df[, indices[3]])
  mod = coxph(km ~ anydem_comorbidity + gender + race_csrd_3cat + 
                marital_status + education_census2013 + 
                income_census2013 + hyperten_comorbidity + mi_comorbidity + 
                cvd_comorbidity + csrdtbi_comorbidity + pd_comorbidity + 
                dm_comorbidity + obesity_comorbidity + sleepissue_comorbidity + 
                pain_comorbidity, ties = 'efron', data = df)
  #mod = coxph(km ~ dementia_time, ties = 'efron', data = df)
  
  # Return results
  return(summary(mod))
}
## ---- end-of-cox_mod

#test_mod = cox_mod('age_start_round', end = 'age_end_mooddis_round', 
                   #event = 'mooddis_status_cox', df = final_df)
#test_mod

# Subset dataframe 
# Args: col is column name; df is dataframe
# Returns: list of dataframes, where each dataframe contains data for specific 
# level of column variable
## ---- subset_data
subset_data = function(col, df){
  dfs = list()
  index = which(colnames(df) == col)
  for (i in 1:length(levels(df[,index]))){
    dfs[[i]] = df[df[,index] == as.numeric(levels(df[,index]))[i],] 
  }
  return(dfs)
}
## --- end-of-subset_data
#gender_df = subset_data('gender', df = final_df)