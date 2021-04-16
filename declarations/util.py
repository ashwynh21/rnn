"""
We are certainly going to require a set of functions that are going to help us run any conversions that we require.
"""
# first we are going to need a function that will convert our input data set to a normalized format.


def result(r):
    """ Displays training results
    """
    print(
        f'\nEpisode {r[0]}/{r[1]} - Profit: {r[2]} Minimization Loss: {int(r[3] * 100)}% - Accuracy: {int(r[4])}%) '
        f'Accounts - {r[5]}')
