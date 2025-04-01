import pandas as pd

def print_basic_statistics(df):
    print("Number of missing values in each column:")
    missing_values = df.isnull().sum()
    print(missing_values)

    total_missing = missing_values.sum()
    print(f"Total number of missing values: {total_missing}")

    missing_percentage = (total_missing / (df.shape[0] * df.shape[1])) * 100
    print(f"Percentage of missing values: {missing_percentage:.2f}%")

    print("Basic statistics:")
    print(df.describe())

    duplicate_rows = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_rows}")



def clean_and_validate_numeric_data(df):

    CORRECT_RANGE = (0, 15)
    COLUMNS = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

    def get_median(data):
        return data.median()
    
    def get_data_within_range(data, range):
        return data[(data > range[0]) & (data < range[1])]
    

    df_cleaned = df.copy()
    for column in COLUMNS:
        # Convert values to float - if conversion not possible, set to NaN
        df_cleaned[column]= pd.to_numeric(df_cleaned[column], errors='coerce')
        
        # Calculate median of valid values (not NaN and within range)
        valid_values = get_data_within_range(df_cleaned[column], CORRECT_RANGE)
        if len(valid_values) > 0:
            median_value = get_median(valid_values)
            
            # Replace NaN and out-of-range values with the median
            invalid_mask = (df_cleaned[column].isna() | 
                           (df_cleaned[column] <= CORRECT_RANGE[0]) | 
                           (df_cleaned[column] >= CORRECT_RANGE[1]))
              
            # Replace invalid values with median
            df_cleaned.loc[invalid_mask, column] = median_value
            
        else:
            print(f'  Warning: No valid values found in column {column}')

    print("Basic statistics after cleaning data:")
    print(df_cleaned.describe())
    return df_cleaned

def clean_and_validate_category_data(df):
    COLUMN = 'variety'
    VALID_CATEGORIES = ['Setosa', 'Versicolor', 'Virginica']

    CORRECTIONS = [
        ('Versicolour', 'Versicolor'),
    ]

    def validate_category(data):
        return data.isin(VALID_CATEGORIES)

    df_cleaned = df.copy()

    if df_cleaned[COLUMN].isna().any():
        most_common = df_cleaned[COLUMN].mode()[0]
        df_cleaned[COLUMN] = df_cleaned[COLUMN].fillna(most_common)
        print(f"  Replaced {df_cleaned[COLUMN].isna().sum()} NaN values with '{most_common}'")


    invalid_mask = ~validate_category(df_cleaned[COLUMN])

    if invalid_mask.any():
        invalid_indices = df_cleaned[invalid_mask].index
        for idx in invalid_indices:
            original = df_cleaned.at[idx, COLUMN]
            capitalized = original.capitalize()
            df_cleaned.at[idx, COLUMN] = capitalized
            # print(f"  Corrected '{original}' to '{capitalized}'")

        for value, correction in CORRECTIONS:
            if value in df_cleaned[COLUMN].values:
                df_cleaned[COLUMN] = df_cleaned[COLUMN].replace(value, correction)
                # print(f"  Corrected '{value}' to '{correction}'")
    


    still_invalid_mask = ~validate_category(df_cleaned[COLUMN])
    still_invalid_count = still_invalid_mask.sum()
    if still_invalid_count > 0:
        print(f"  Warning: {still_invalid_count} invalid values found in column {COLUMN} after corrections:")
        print(df_cleaned.loc[still_invalid_mask])
    else:
        print(f"  All values in column {COLUMN} are now valid.")

    

def main():
    df = pd.read_csv("iris_with_errors.csv")
    print_basic_statistics(df)

    clean_and_validate_numeric_data(df)
    clean_and_validate_category_data(df)

if __name__ == "__main__":
    main()