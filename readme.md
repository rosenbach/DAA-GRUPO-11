## Goal
The objective of each working group is to develop a model capable of predicting the traffic flow at a given time point. For this competition, several streets in the city of Porto were grouped in order to obtain a metric of the state of traffic in the city itself.

The dataset used in this competition contains a set of features, the average_speed_diff being worth mentioning. This feature indicates, on a qualitative scale (None, Low, Medium, High and Very_High), whether there is traffic or not! This feature (speed difference) corresponds to the difference between (1.) the maximum speed that vehicles can reach in scenarios without traffic and (2.) the speed that actually takes place at that time point. Thus, if the speed difference is "none", it means that the speed of cars in the city of Porto at that particular time point is the maximum speed they are allowed to reach in scenarios without traffic (ie, there is no traffic). If, on the other hand, the speed difference is "very high" it means that the speed of cars on a certain street is much lower than the maximum speed they are allowed to reach in scenarios without traffic (ie, there is traffic).

You will be provided with two datasets. One of them, the learning dataset, should be used to train and tune the Machine Learning model. The second should be used as a test dataset. Thus, for each record of the test dataset, they must predict the transit that occurs, using the scale None, Low, Medium, High and Very_High.

## Metrics
As a metric, the correctness percentage will be used, i.e., the accuracy.

## Submission File Format
You can submit up to 3 valid files per day (invalid submissions do not count towards the daily limit) and, in the end, only one (chosen by you) will count towards the competition ranking. They must submit a file, in CSV format, with exactly 1500 records (exactly the same as the test dataset) plus the header (RowId, Speed_Diff). The submission will give an error if the file is badly formatted or has more lines and/or columns. You can find an example submission file in the Data tab for this competition.

Please don't build the file by hand!

The file must have two columns and the following header, i.e.: