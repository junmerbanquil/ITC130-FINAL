import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Title of the web app
st.title('Crop Recommendation System')

# Load your dataset
@st.cache_data
def load_data():
    # Load your data from a CSV file (replace with the path to your CSV file)
    data = pd.read_csv('Crop_Recommendation.csv')
    return data

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('crop_recommendation_model.joblib', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model file is in the correct location.")
    except pickle.UnpicklingError:
        st.error("Error unpickling the model. The file might be corrupted.")
        return None

df = load_data()
model = load_model()

# Sidebar header
st.sidebar.markdown("""
## ITC130 Final Exam
## Streamlit Project by:
## Junmer Banquil & Kristine Notarion
""")

# Sidebar menu for navigation
menu = st.sidebar.selectbox(
    'Select an option',
    ('Dataset', 'Crop Distribution', 'Soil Nutrients Distribution', 
     'Temperature and Humidity Distribution', 'Temperature vs. Humidity', 
     'Correlation Heatmap', 'Crop Recommendation')
)

if menu == 'Dataset':
    # Display the dataset as a table
    st.write('## Dataset')
    st.write(df)

elif menu == 'Crop Distribution':
    # Visualizing crop distribution using a pie chart
    st.write('### Crop Distribution')
    st.write("This pie chart shows the proportion of each crop type in the dataset. It helps us understand the distribution of different crops and their relative frequencies.")
    fig, ax = plt.subplots()
    # Check if 'Crop' column exists before creating the pie chart
    if 'Crop' in df.columns:
        crop_counts = df['Crop'].value_counts()
        ax.pie(crop_counts, labels=crop_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title('Proportion of Each Crop Type')
        st.pyplot(fig)
    else:
        st.write('Crop column is not available in the dataset.')

elif menu == 'Soil Nutrients Distribution':
    # Nitrogen, Phosphorus, and Potassium Distribution
    st.write('### Distribution of Soil Nutrients')
    st.write("These histograms show the distribution of Nitrogen, Phosphorus, and Potassium levels in the soil. Understanding these distributions can help in determining the soil's fertility and nutrient availability.")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(df['Nitrogen'], bins=20, kde=True, ax=ax[0], color='blue')
    ax[0].set_title('Nitrogen Distribution')
    sns.histplot(df['Phosphorus'], bins=20, kde=True, ax=ax[1], color='green')
    ax[1].set_title('Phosphorus Distribution')
    sns.histplot(df['Potassium'], bins=20, kde=True, ax=ax[2], color='red')
    ax[2].set_title('Potassium Distribution')
    st.pyplot(fig)

elif menu == 'Temperature and Humidity Distribution':
    # Temperature and Humidity Distribution
    st.write('### Distribution of Temperature and Humidity')
    st.write("These histograms illustrate the distribution of temperature and humidity values in the dataset. This information is crucial for understanding the climate conditions under which different crops are grown.")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df['Temperature'], bins=20, kde=True, ax=ax[0], color='orange')
    ax[0].set_title('Temperature Distribution')
    sns.histplot(df['Humidity'], bins=20, kde=True, ax=ax[1], color='purple')
    ax[1].set_title('Humidity Distribution')
    st.pyplot(fig)

elif menu == 'Temperature vs. Humidity':
    # Scatter plot of Temperature vs. Humidity
    st.write('### Temperature vs. Humidity')
    st.write("This scatter plot shows the relationship between temperature and humidity for different crops. It helps in identifying how different crops are distributed across various temperature and humidity conditions.")
    fig, ax = plt.subplots()

    # Check if 'Crop' column exists before using it
    if 'Crop' in df.columns:
        sns.scatterplot(data=df, x='Temperature', y='Humidity', hue='Crop', palette='viridis')
    else:
        sns.scatterplot(data=df, x='Temperature', y='Humidity', palette='viridis')

    ax.set_title('Scatter plot of Temperature vs. Humidity')
    st.pyplot(fig)

elif menu == 'Correlation Heatmap':
    # Correlation heatmap
    st.write('### Correlation Heatmap')
    st.write("This heatmap shows the correlation between different parameters in the dataset. High correlation values (positive or negative) indicate a strong relationship between the parameters, which can be important for understanding how different factors influence each other.")
    fig, ax = plt.subplots(figsize=(10, 8))
    # Select only numeric columns for the correlation matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation between Parameters')
    st.pyplot(fig)

elif menu == 'Crop Recommendation':
    # Adding sliders for input parameters
    st.write('## Select Input Parameters for Crop Recommendation')

    nitrogen = st.slider('Select Nitrogen value', int(df['Nitrogen'].min()), int(df['Nitrogen'].max()), step=1)
    phosphorus = st.slider('Select Phosphorus value', int(df['Phosphorus'].min()), int(df['Phosphorus'].max()), step=1)
    potassium = st.slider('Select Potassium value', int(df['Potassium'].min()), int(df['Potassium'].max()), step=1)
    temperature = st.slider('Select Temperature value', float(df['Temperature'].min()), float(df['Temperature'].max()), step=0.1)
    humidity = st.slider('Select Humidity value', float(df['Humidity'].min()), float(df['Humidity'].max()), step=0.1)
    ph_value = st.slider('Select pH Value', float(df['pH_Value'].min()), float(df['pH_Value'].max()), step=0.1)
    rainfall = st.slider('Select Rainfall value', float(df['Rainfall'].min()), float(df['Rainfall'].max()), step=0.1)

    # Function to recommend a crop based on input parameters
    def recommend_crop(model, nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall):
        try:
            input_data = pd.DataFrame({
                'Nitrogen': [nitrogen],
                'Phosphorus': [phosphorus],
                'Potassium': [potassium],
                'Temperature': [temperature],
                'Humidity': [humidity],
                'pH_Value': [ph_value],
                'Rainfall': [rainfall]
            })
            prediction = model.predict(input_data)
            return prediction[0]
        except Exception as e:
            st.error(f"Error in predicting crop: {e}")
            return None

    # Displaying selected values and crop recommendation
    st.write('## Selected Input Parameters')
    st.write(f'- Nitrogen: {nitrogen}')
    st.write(f'- Phosphorus: {phosphorus}')
    st.write(f'- Potassium: {potassium}')
    st.write(f'- Temperature: {temperature}')
    st.write(f'- Humidity: {humidity}')
    st.write(f'- pH Value: {ph_value}')
    st.write(f'- Rainfall: {rainfall}')

    # Recommend crop
    if model:
        crop = recommend_crop(model, nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall)
        if crop:
            st.write('## Recommended Crop')
            st.write(f'The recommended crop is: {crop}')
    else:
        st.write("Unable to recommend crop due to model loading error.")

# Narrative explaining the context of the data
st.sidebar.write("""
### Context and Usage of the Data
This dataset contains various agricultural parameters such as Nitrogen, Phosphorus, Potassium levels in the soil,
Temperature, Humidity, pH value, and Rainfall. These parameters are critical for determining the optimal crop type
for a given set of conditions. The goal of this application is to recommend the best crop based on user inputs for these parameters.

### Metadata
- **Nitrogen**: Amount of Nitrogen in the soil (in ppm)
- **Phosphorus**: Amount of Phosphorus in the soil (in ppm)
- **Potassium**: Amount of Potassium in the soil (in ppm)
- **Temperature**: Ambient temperature (in Celsius)
- **Humidity**: Humidity level (in %)
- **pH Value**: pH level of the soil
- **Rainfall**: Rainfall received (in mm)

### Insights
The following visualizations provide insights into the distribution and relationships of these parameters.
""")
