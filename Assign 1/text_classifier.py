def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    Input:
        test_x: A list of reviews
        test_y: the corresponding labels for the list of reviews
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of reviews classified correctly)/(total # of reviews)
    """
    accuracy = 0  

    
    y_hats = []
    for review in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(review, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    error = np.mean(np.absolute(y_hats - test_y))

    accuracy = 1 - error

    return y_hats

if __name__ == '__main__':
    
    file = open('model_params.txt', 'r')
    logprior, loglikelihood = file.read()
    file.close()
    
    while(1):
        my_review = input("Enter your review or enter X to quit:")
        if my_review == 'X'or my_review == 'x':
            sys.exit()
        else:
            p = naive_bayes_predict(my_review, logprior, loglikelihood)
            print('The expected output is', p)
