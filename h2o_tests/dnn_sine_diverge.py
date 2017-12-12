import h2o
import math
import pandas as pd
import matplotlib.pyplot as plt
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


def sine_df(periods, density):
    """Generates a sine wave.

    Args:
        periods (int): Number of periods on each side of 0.
        density (int): Number of data points per period.

    Returns:
        DataFrame with 'x' and 'y' columns sampling a sine wave with period 1.

    """
    num_points = 2 * periods * density
    df = pd.DataFrame(columns=['x', 'y'])
    df['x'] = [1.0 / density * i for i in range(-num_points/2, num_points/2)]
    df['y'] = df.apply(lambda row: math.sin(2*math.pi*row['x']), axis=1)

    return df


def main():

    # Generate dataset for y = x^2
    df = sine_df(glob_train_periods, glob_density)

    # Start h2o
    h2o.init(ip='192.168.0.41', port=65432, max_mem_size_GB=128)

    # Create H2OFrame
    column_types = ['real', 'real']
    hf = h2o.H2OFrame(df, column_types=column_types)
    train, val = hf.split_frame(ratios=[0.8])

    # Create model
    predictors = 'x'
    response = 'y'
    model = H2ODeepLearningEstimator(
        model_id='dnn_sine',
        epochs=5000,
        hidden=[800],
        activation='rectifier',
        # hidden_dropout_ratios=[0.0],
        l1=1e-4,
        l2=1e-4,
        max_w2=0.55,
        stopping_rounds=8,
        # stopping_tolerance=1e-4,
        stopping_metric='rmse',

        # Control scoring epochs
        score_interval=0,
        score_duty_cycle=1,
        shuffle_training_data=False,
        replicate_training_data=True,
        train_samples_per_iteration=int(0.5 * len(df) / 1.258),
    )
    model.train(x=predictors, y=response, training_frame=train, validation_frame=val)

    # Create test set with domain outside training
    test_df = sine_df(glob_test_periods, glob_density)
    test = h2o.H2OFrame(test_df, column_types=column_types)
    test_df['predict'] = model.predict(test).as_data_frame()

    # Plot results
    plt.plot(test_df['x'], test_df['y'])
    plt.plot(test_df['x'], test_df['predict'])
    plt.xlim(-glob_test_periods, glob_test_periods)
    plt.show()


if __name__ == '__main__':

    # Define globals
    glob_train_periods = 5
    glob_test_periods = 10
    glob_density = 1000

    # Run main function
    main()
