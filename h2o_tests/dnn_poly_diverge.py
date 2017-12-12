import h2o
import pandas as pd
import matplotlib.pyplot as plt
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


# Define columns and types
columns = ['x', 'y']
column_types = ['real', 'real']

# Generate dataset for y = x^2
df = pd.DataFrame(columns=columns)
df['x'] = [0.0001 * i for i in range(-100000, 100000)]
df['y'] = df.apply(lambda row: row['x'] ** 2, axis=1)

# Start h2o
h2o.init(ip='192.168.0.41', port=65432, max_mem_size_GB=128, nthreads=70)

# Create H2OFrame
hf = h2o.H2OFrame(df, column_types=column_types)
train, val = hf.split_frame(ratios=[0.8])

# Create model
predictors = 'x'
response = 'y'
model = H2ODeepLearningEstimator(
    model_id='dnn_poly',
    epochs=5000,
    hidden=[300, 300],
    activation='maxout_with_dropout',
    hidden_dropout_ratios=[0.5, 0.5],
    l1=1e-4,
    l2=1e-4,
    max_w2=0.55,
    stopping_rounds=10,
    # stopping_tolerance=1e-4,
    stopping_metric='rmse',

    # Control scoring epochs
    score_interval=0,
    score_duty_cycle=1,
    shuffle_training_data=False,
    replicate_training_data=True,
    train_samples_per_iteration=int(5 * len(df) / 1.258),

    # Controlling momentum
)
model.train(x=predictors, y=response, training_frame=train, validation_frame=val)

# Create test set with domain outside training
test_df = pd.DataFrame(columns=columns)
test_df['x'] = [0.0001 * i for i in range(-150000, 150000)]
test_df['y'] = test_df.apply(lambda row: row['x'] ** 2, axis=1)
test = h2o.H2OFrame(test_df, column_types=column_types)
test_df['predict'] = model.predict(test).as_data_frame()
print model.model_performance(test)

# Plot results
driverless_predict = pd.DataFrame(columns=['x', 'y'])
driverless_predict['x'] = test_df['x']
driverless_predict['y'] = pd.read_csv('/home/kimmmx/Downloads/test_preds.csv')

plt.plot(test_df['x'], test_df['y'])
plt.plot(test_df['x'], test_df['predict'])
plt.plot(driverless_predict['x'], driverless_predict['y'])
plt.show()
