import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#Create a dataframe and select target and features
data = pd.read_csv('/Users/mamduhhalawa/Desktop/Churn_predictor/churn_prediction.csv')
data.info()

features = ['previous_month_end_balance','average_monthly_balance_prevQ',
            'average_monthly_balance_prevQ2','current_month_credit','previous_month_credit',
            'current_month_debit','previous_month_debit','current_month_balance',
            'previous_month_balance']
X = data[features]

y = data.churn
y

#Train the first model

First_model = RandomForestRegressor(random_state=1)
First_model.fit(X,y)
first_pred = First_model.predict(X)
first_pred

print(f'The first predictions are{first_pred}')
print(f'The actual first values are{y}')

mae = mean_absolute_error(first_pred,y)
print(f'The MAE is {mae}')


# Split the data and see MAE again
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

First__split_model = RandomForestRegressor(random_state=1)
First__split_model.fit(train_X,train_y)
second_pred = First__split_model.predict(val_X)
mae_split = mean_absolute_error(second_pred,val_y)

print(f'The second predictions are{second_pred[:10]}')
print(f'The actual second values are{val_y.iloc[:10]}')

print(f'The MAE is {mae_split}')

# Naturally the MAE is higher as you decrease the data size

pred_df = pd.DataFrame(second_pred)

plt.plot(Figsize=(50,5))
pred_df.iloc[:50].plot()

val_y.iloc[:50].plot(kind='bar',color='green')








