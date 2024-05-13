def make_predictions(dataframe):
    import joblib
    import numpy as np
    from category_encoders import OneHotEncoder
    from skimpy import clean_columns
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    pred_model = joblib.load('./artifacts/model_2.pkl')

    #pred_df = pd.read_csv(dataframe)
    pred_df = dataframe
    #removing the irrelevent feature
    pred_df.drop(
        columns='CustomerID',
        inplace=True
    )

    num_col = ['Income','NumOfPurchases'] #creating a list of the numeric features with missing values
    cat_col = ['Gender','Location','MaritalStatus'] #creating a list of categorical features with missing values

    for col in num_col: #for each of the columns in the list replace the missing values with the mean of the column
        pred_df[col].fillna(
            pred_df[col]
            .dropna()
            .mean(),
            inplace= True
        )

    for col1 in cat_col:
        pred_df[col1].fillna( #replace the missing categorical values with the mode of the feature
            pred_df[col1]
            .mode()[0],
            inplace= True
        )

    pred_df.drop(
        columns= 'Age',
        inplace= True
    )

    pred_df = clean_columns(pred_df)

    label_enc = LabelEncoder()

    pred_df['gender'] = label_enc.fit_transform(pred_df['gender'])
    pred_df['location'] = label_enc.fit_transform(pred_df['location'])
    pred_df['marital_status'] = label_enc.fit_transform(pred_df['marital_status'])
    pred_df['education'] = label_enc.fit_transform(pred_df['education'])
    pred_df['subscription_plan'] = label_enc.fit_transform(pred_df['subscription_plan'])

    # Bin income into brackets
    bins = [0, 30000, 50000, 70000, float('inf')]
    labels = ['Low Income', 'Medium Income', 'High Income', 'Very High Income']
    pred_df['income_bin'] = pd.cut(pred_df['income'], bins=bins, labels=labels, right=False)

    # Added average feature
    pred_df['average_purchase'] = round(pred_df['total_purchase'] / pred_df['num_of_purchases'],0)

    pred_df['income'] = round(pred_df['income'],0)  

    # Reorder columns
    pred_df = pred_df[['gender', 'income', 'income_bin', 'total_purchase', 'num_of_purchases', 'average_purchase', 'location', 'marital_status', 'education', 'subscription_plan']]

    pred_df = OneHotEncoder(handle_unknown='ignore', use_cat_names=True).fit_transform(pred_df)

    predictions = pred_model.predict(pred_df)

    predictions = np.where(predictions == 1, 'yes', 'no')

    return predictions