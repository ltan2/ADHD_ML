from tsfresh.feature_selection.relevance import calculate_relevance_table

def tsfresh_pvalue(dataX, dataY):
    p_table = calculate_relevance_table(dataX, dataY, n_jobs=2, ml_task='classification',show_warnings=True)
    relevant_features = p_table[p_table.p_value<0.05].feature
    dataX = dataX.loc[:, relevant_features]

    return dataX