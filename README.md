# finance_machine_learning
Attempt at some finance tools and using machine learning to try and predict data

Latest commit is a version I've been working on for about a week, built using Sentdex's finance data analysis lessons initially.
Now repurposed, feeding in UK stocklists from a downloadable list from the LSE website.
Data drawn from 2010, reconfigurable as needed.
Machine learning conducted based on a selected stock, or all stocks
Lag variable can shift the stock up whatever number of days so machine learning predicts whether the stock will rise or fall based on other stocks for that day shift
The train and test data is processed based on whether it rose or fell from reconfigurable variables. Data converted to list of 1, 0 and -1 for up, same, down in value respectively
