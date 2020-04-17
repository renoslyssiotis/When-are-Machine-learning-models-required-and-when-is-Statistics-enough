def retrieve_best_ML_and_stats_model(nested_results):
    """
    Retrieves the performanc of the best performing ML model
    and the performance of logistic regression, in order to find
    the performance gain of ML over statistics
    """
    logit_performances = {}
    ML_performances = {}
    performance_gain = {}
    
    for i in range(len(nested_results)):
        
        logit_performances[i] = list(nested_results[i].values())[0]
        ML_performances[i] = max(list(nested_results[i].values())[1:])
        performance_gain[i] = ML_performances[i] - logit_performances[i]
        
    return logit_performances, ML_performances, performance_gain