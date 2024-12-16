import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("Housing.csv")

# Select relevant columns
X = data[['area', 'bedrooms']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Simplified output
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Save the model as a pickle file
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Function for price prediction
def predict_price(area, bedrooms):
    example = pd.DataFrame({'area': [area], 'bedrooms': [bedrooms]})
    predicted_price = model.predict(example)
    return int(predicted_price[0])

# Example usage
area = 3000
bedrooms = 3
predicted_price = predict_price(area, bedrooms)
print(f"Predicted price for area={area} and bedrooms={bedrooms}: {predicted_price}")
