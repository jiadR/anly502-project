# anly502-project

# Group members: 
Ning Hu
Xiangyu Hu
Dongru Jia
Xiwen Zhang
 
# Group name:
Winner Winner Chicken Dinner

# Executive Summary:
This project mainly talks about the game Player Unknown’s Battlegrounds. According to the exploring description analysis, Spark machine learning and other methods to try to  predict the team placement of the game, recommended the weapons in the game and helped players arrange their path and locations. Hope that players can find easier ways to win the game and game companies can improve the game more wisely. 

# Introduction
Playing video games is a good way of relaxing and escaping from pressure. But losing every game certainly does not serve any of the two purposes. So how to become a more competitive player in video games, in our case PUBG, is the focus of the project. By applying a couple machine learning models to the game data to analyze and extract the best way of winning and coming on top of other players, and try to give suggestions to game companies on how to improve the gameplay is the goals of the project.
The game that is used in this project is Player Unknown’s Battlegrounds (PUBG), an online multiplayer battle game developed and published by PUBG Corporation, a subsidiary of South Korean video game company Bluehole. Maximum one hundred players will parachute onto an island and search weapons and equipment to kill each other and the last one or team who survived will win the game.
The dataset was directly downloaded from Kaggle.com, a Kaggle user extracted over 720,000 competitive matches of the PUBG game from pubg.op.gg, a game tracker website.
This dataset provides two zips: “aggregate” and “deaths”.
In aggregate, each match's meta information and player statistics are summarized (as provided by PUBG). It includes various aggregate statistics such as each player’s survival time, damage conducted, distance walked, etc. as well as metadata on the match itself such as queue size, fpp/tpp, date, etc.   
In deaths, the files record every death that occurred within the 720k matches. That is, each row documents an event when a player is taken out from the game in the match. Deaths datasets include the locations, names, and placements of killer and victim, and also how they were taken out.

# Code files
https://github.com/jiadR/anly502-project

# Analysis/Method Selection
## Data Cleaning
### Aggregate Dataset
Combining the file together is the first step.
After dropping the missing value, the aggregate file contains 67369231 rows of data. 

![](anly502-project/pic/1.1.png)
(Figure 1.1 the Schema of aggregate data)

The unused variables like date, macth_id, player_name and team_id were all dropped because they are not useful for the analysis part.
Some variables’ data type should be changed according to figure 1.1, the schema of the data. ‘player_dist_ride’ and ‘player_dist_walk’ means the distance the player walks or rides in the game, so they should be changed into integer type. ‘player_survive_time’ here also should be changed into integer type because it is the time the player survived in the game.







