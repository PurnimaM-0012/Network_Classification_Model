#!/usr/bin/env python
# coding: utf-8

# In[48]:

def load_and_preprocess_data(path="drive.csv"):
    import pandas as pd
    df = pd.read_csv("drive.csv")
    flow_cols = ["BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV", "REV_MORE"]
    keep_patterns = [
        r"^PKT_LENGTHS_",     # all columns starting with PKT_LENGTHS_
        r"^INTERVALS_",       # all columns starting with INTERVALS_
        r"^BRST_COUNT$",      # exactly BRST_COUNT
        r"^BRST_BYTES_",      # all columns starting with BRST_BYTES_
        r"^BRST_PACKETS_",    # all columns starting with BRST_PACKETS_
        r"^BRST_INTERVALS_",  # all columns starting with BRST_INTERVALS_
        r"^BRST_DURATION_"    # all columns starting with BRST_DURATION_
    ]
    cols_to_keep = df.filter(regex="|".join(keep_patterns)).columns.tolist()
    cols_to_keep = flow_cols + cols_to_keep
    df_kept = df[cols_to_keep]
    df_kept.head()


    # In[49]:


    df.drop_duplicates(inplace=True)


    # In[50]:


    import numpy as np
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "null": np.nan})


    # In[51]:


    df['DBI_BRST_BYTES_mean'] = df['DBI_BRST_BYTES'].apply(lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan)
    df['DBI_BRST_BYTES_std']  = df['DBI_BRST_BYTES'].apply(lambda x: np.std(x)  if isinstance(x, list) and len(x) > 0 else np.nan)
    df['DBI_BRST_BYTES_min']  = df['DBI_BRST_BYTES'].apply(lambda x: np.min(x)  if isinstance(x, list) and len(x) > 0 else np.nan)
    df['DBI_BRST_BYTES_max']  = df['DBI_BRST_BYTES'].apply(lambda x: np.max(x)  if isinstance(x, list) and len(x) > 0 else np.nan)
    df['DBI_BRST_BYTES_len']  = df['DBI_BRST_BYTES'].apply(lambda x: len(x) if isinstance(x, list) else 0)


    # In[52]:


    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())


    # In[53]:


    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")


    # In[78]:


    from sklearn.preprocessing import LabelEncoder
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    df_encoded['TYPE'] = label_encoder.fit_transform(df_encoded['TYPE'])
    print("Label encoding complete for target column (TYPE).")
    print("Encoded traffic classes:", list(label_encoder.classes_))


    # In[90]:


    import numpy as np

    def extract_features(series):
        return pd.DataFrame({
            f"{series.name}_mean": series.apply(lambda x: np.mean(x) if isinstance(x, list) and len(x)>0 else np.nan),
            f"{series.name}_max":  series.apply(lambda x: np.max(x) if isinstance(x, list) and len(x)>0 else np.nan),
            f"{series.name}_min":  series.apply(lambda x: np.min(x) if isinstance(x, list) and len(x)>0 else np.nan),
            f"{series.name}_std":  series.apply(lambda x: np.std(x) if isinstance(x, list) and len(x)>0 else np.nan),
            f"{series.name}_len":  series.apply(lambda x: len(x) if isinstance(x, list) else 0),
        })


    # In[91]:


    import numpy as np
    import pandas as pd

    # Function to convert list-like strings into numeric lists
    def safe_eval(x):
        try:
            if isinstance(x, str):
                x = x.strip()
                if x.startswith("[") and x.endswith("]"):
                    return [float(i) for i in x.strip("[]").split(",") if i.strip() != ""]
            return x
        except:
            return []

    # Apply to all list-like columns
    list_cols = ['DBI_BRST_BYTES', 'DBI_BRST_PACKETS', 'PKT_LENGTHS',
                'PPI_PKT_DIRECTIONS', 'PKT_TIMES', 'DBI_BRST_TIME_START',
                'DBI_BRST_TIME_STOP', 'DBI_BRST_DURATION',
                'DBI_BRST_INTERVALS', 'TIME_INTERVALS']

    for col in list_cols:
        df[col] = df[col].apply(safe_eval)

    # Extract numerical features
    def extract_features(series):
        return pd.DataFrame({
            f"{series.name}_mean": series.apply(lambda x: np.mean(x) if isinstance(x, list) and len(x)>0 else np.nan),
            f"{series.name}_max":  series.apply(lambda x: np.max(x) if isinstance(x, list) and len(x)>0 else np.nan),
            f"{series.name}_min":  series.apply(lambda x: np.min(x) if isinstance(x, list) and len(x)>0 else np.nan),
            f"{series.name}_std":  series.apply(lambda x: np.std(x) if isinstance(x, list) and len(x)>0 else np.nan),
            f"{series.name}_len":  series.apply(lambda x: len(x) if isinstance(x, list) else 0),
        })

    # Create new feature DataFrames
    feature_dfs = []
    for col in list_cols:
        feature_dfs.append(extract_features(df[col]))

    # Concatenate new features
    df_features = pd.concat(feature_dfs, axis=1)

    # Combine with target column
    df_model = pd.concat([df_features, df['TYPE']], axis=1)

    print("Final dataset shape:", df_model.shape)


    # In[86]:


    import numpy as np
    import pandas as pd

    # Function to convert list-like strings into numeric lists
    def safe_eval(x):
        try:
            if isinstance(x, str):
                x = x.strip()
                if x.startswith("[") and x.endswith("]"):
                    return [float(i) for i in x.strip("[]").split(",") if i.strip() != ""]
            return x
        except:
            return []

    # Apply to all list-like columns
    list_cols = ['DBI_BRST_BYTES', 'DBI_BRST_PACKETS', 'PKT_LENGTHS',
                'PPI_PKT_DIRECTIONS', 'PKT_TIMES', 'DBI_BRST_TIME_START',
                'DBI_BRST_TIME_STOP', 'DBI_BRST_DURATION',
                'DBI_BRST_INTERVALS', 'TIME_INTERVALS']

    for col in list_cols:
        df[col] = df[col].apply(safe_eval)

    # Extract numerical features
    def extract_features(series):
        return pd.DataFrame({
            f"{series.name}_mean": series.apply(lambda x: np.mean(x) if isinstance(x, list) and len(x)>0 else np.nan),
            f"{series.name}_max":  series.apply(lambda x: np.max(x) if isinstance(x, list) and len(x)>0 else np.nan),
            f"{series.name}_min":  series.apply(lambda x: np.min(x) if isinstance(x, list) and len(x)>0 else np.nan),
            f"{series.name}_std":  series.apply(lambda x: np.std(x) if isinstance(x, list) and len(x)>0 else np.nan),
            f"{series.name}_len":  series.apply(lambda x: len(x) if isinstance(x, list) else 0),
        })

    # Create new feature DataFrames
    feature_dfs = []
    for col in list_cols:
        feature_dfs.append(extract_features(df[col]))

    # Concatenate new features
    df_features = pd.concat(feature_dfs, axis=1)

    # Combine with target column
    df_model = pd.concat([df_features, df['TYPE']], axis=1)

    print("Final dataset shape:", df_model.shape)


    # In[92]:


    import ast

    def to_list(x):
        """Convert string '[1, 2, 3]' -> [1,2,3], keep [] if empty."""
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)   # safely evaluate string to list
            except:
                return []
        elif isinstance(x, list):
            return x
        else:
            return []

    list_cols = ['DBI_BRST_BYTES', 'DBI_BRST_PACKETS', 'PKT_LENGTHS',
                'PPI_PKT_DIRECTIONS', 'PKT_TIMES', 'DBI_BRST_TIME_START',
                'DBI_BRST_TIME_STOP', 'DBI_BRST_DURATION',
                'DBI_BRST_INTERVALS', 'TIME_INTERVALS']

    # Apply conversion
    for col in list_cols:
        df[col] = df[col].apply(to_list)


    # In[87]:


    for col in X_train.select_dtypes(include="object").columns:
        print(col, X_train[col].unique()[:10])


    # In[93]:


    for col in list_cols:
        print(col, df[col].apply(type).value_counts().to_dict())


    # In[94]:


    import numpy as np
    import pandas as pd

    def extract_features(df, list_cols):
        feature_dfs = []
        for col in list_cols:
            feature_dfs.append(pd.DataFrame({
                f"{col}_mean": df[col].apply(lambda x: np.mean(x) if len(x)>0 else np.nan),
                f"{col}_max":  df[col].apply(lambda x: np.max(x) if len(x)>0 else np.nan),
                f"{col}_min":  df[col].apply(lambda x: np.min(x) if len(x)>0 else np.nan),
                f"{col}_std":  df[col].apply(lambda x: np.std(x) if len(x)>0 else np.nan),
                f"{col}_len":  df[col].apply(len)
            }))
        return pd.concat(feature_dfs, axis=1)

    # List of columns with lists
    list_cols = [
        'DBI_BRST_BYTES', 'DBI_BRST_PACKETS', 'PKT_LENGTHS',
        'PPI_PKT_DIRECTIONS', 'PKT_TIMES', 'DBI_BRST_TIME_START',
        'DBI_BRST_TIME_STOP', 'DBI_BRST_DURATION', 'DBI_BRST_INTERVALS',
        'TIME_INTERVALS'
    ]

    df_features = extract_features(df, list_cols)

    # Final dataset for ML (only numeric features + label)
    df_model = pd.concat([df_features, df['TYPE']], axis=1)
    print(df_model.dtypes)


    # In[100]:


    import re
    import numpy as np
    import pandas as pd

    # Function: convert "[4172, 123]" → list of numbers
    def extract_numbers(val):
        if isinstance(val, str):
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)  # match ints and floats
            return [float(n) for n in nums]
        elif isinstance(val, list):
            return [float(x) for x in val]
        else:
            return []

    # Columns that contain lists
    list_cols = [
        'DBI_BRST_BYTES', 'DBI_BRST_PACKETS', 'PKT_LENGTHS',
        'PPI_PKT_DIRECTIONS', 'PKT_TIMES', 'DBI_BRST_TIME_START',
        'DBI_BRST_TIME_STOP', 'DBI_BRST_DURATION', 'DBI_BRST_INTERVALS',
        'TIME_INTERVALS'
    ]

    # Apply conversion
    for col in list_cols:
        df[col] = df[col].apply(extract_numbers)

    # Extract features without errors
    def extract_features(df, list_cols):
        feature_dfs = []
        for col in list_cols:
            feature_dfs.append(pd.DataFrame({
                f"{col}_mean": df[col].apply(lambda x: np.mean(x) if len(x)>0 else np.nan),
                f"{col}_max":  df[col].apply(lambda x: np.max(x) if len(x)>0 else np.nan),
                f"{col}_min":  df[col].apply(lambda x: np.min(x) if len(x)>0 else np.nan),
                f"{col}_std":  df[col].apply(lambda x: np.std(x) if len(x)>0 else np.nan),
                f"{col}_len":  df[col].apply(len)
            }))
        return pd.concat(feature_dfs, axis=1)

    # Build clean numeric dataset
    df_features = extract_features(df, list_cols)
    df_model = pd.concat([df_features, df['TYPE']], axis=1)


    # In[107]:


    print(y.value_counts())


    # In[109]:


    df_features = extract_features(df, list_cols)
    df_model = pd.concat([df_features, df['TYPE']], axis=1)
    X = df_model.drop("TYPE", axis=1)
    y = df_model["TYPE"]


    # In[124]:

def train_model(df_model):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report

    # Example with Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))


    # In[123]:


    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.show()


    # In[36]:


    import numpy as np

    # Save feature names before scaling
    feature_names = X.columns  # X is your dataframe before train-test split

    # Get feature importance from Random Forest
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 10 Features:")
    for i in range(10):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")


    # In[37]:


    import joblib

    joblib.dump(rf, "traffic_classifier.pkl")
    joblib.dump(scaler, "scaler.pkl")


    # In[125]:

import joblib

def load_model(model_path="traffic_classifier.pkl"):
    return joblib.load("traffic_classifier.pkl")


    # In[133]:


    get_ipython().system('pip freeze | findstr /v "@ file" > requirements.txt')


    # In[132]:


    get_ipython().system('type requirements.txt')


    # In[ ]:




