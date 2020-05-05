# anly502-project

# Group members: 
Ning Hu,
Xiangyu Hu,
Dongru Jia,
Xiwen Zhang,
 
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

![](pic/1.1.PNG)

(Figure 1.1 the Schema of aggregate data)

The unused variables like date, macth_id, player_name and team_id were all dropped because they are not useful for the analysis part.
Some variables’ data type should be changed according to figure 1.1, the schema of the data. ‘player_dist_ride’ and ‘player_dist_walk’ means the distance the player walks or rides in the game, so they should be changed into integer type. ‘player_survive_time’ here also should be changed into integer type because it is the time the player survived in the game.

For ‘game_size’ and ‘party_size’ and ‘macth_mode’, they are qualitative and categorical variables, so using the StringIndexer method to transform them into double type in order to fit the models.

The ‘teamplacement’ is the predict label in this data, so it is also transformed as a double type.
According to the summary description of the data, the variable ‘player_survive_time’ has an extremely large value that was unusual. Based on the game’s condition, the survival time cannot be longer than 5000 seconds, so in this column, the value that more than 5000 will be dropped. After this procedure, it now has 67369186 rows, 45 rows were removed.
After containing all the predictors into a vector by VectorAssembler, the features also need to be scaled. Using StandardScaler to transform the features’ vector into a normalize and standard vector. 

![](pic/1.2.PNG)

(Figure 1.2 the Schema of clean aggregate data)

Figure 1.2 shows the schema of the aggregate data after the cleaning process.

### Deaths Dataset
For the ‘deaths’ dataset, the procedure is almost the same.

![](pic/1.3.png)

(figure 1.3 the schema of deaths data)

Figure 1.3 shows the original Schema of this dataset. It has 65370475 rows, after dropping the unused columns ‘killer_name’, ‘match_id’ and ‘victim_name’ and removing the missing value, it now has 58923976 rows of data.
Then, change the data type of the positions of both killer and victim from double to integer type.  A new feature ‘DIST’ was created to measure the distance of killer and victim in order to find the relationship between distance and weapons effect.
For ‘killed_by’, this variable represents the weapons the killer used, so transform it into a double type and name it ‘weapon index’. And the map is categorical data, so change it into double type, too.
Then, like the aggregate dataset, containing all the predictors into a vector by VectorAssembler, the features also need to be scaled. Using StandardScaler to transform the features’ vector into a normalize and standard vector. 

![](pic/1.4.png)

(Figure 1.4 the Schema of clean deaths data)

Figure 1.4 is the Schema that shows the deaths data after cleaning.

## EDA
![](pic/vis1.png)

(Figure 2.1 Histogram plot of gamesize)

Figure 2.1 shows the distribution and frequency of the gamesize in the aggregate dataset. The plot was separated into two parts, the first part indicates that the relatively smaller game size, 25-30 players or 45-50 players are the most common condition. The second part is relatively larger in size, it contains mainly 90-100 players, and 100 players is the maximum number of the game.  

![](pic/vis2.png)

(Figure 2.2 Histogram plot of teamplacment)

Figure 2.2 shows the distribution and frequency of the teamplacement in the aggregate dataset. In this dataset, nearly 60% of the players get the team placement in the top 20. It is mainly because of the limitation of game size.  The highest frequency appears in the 5-10 interval, this stage may be the fierce part of the game, players will try their best to survive.  

![](pic/vis3.png)

(Figure 2.3 Boxplot of party_size vs player_survive_time)

From Figure 2.3,  the box plot reflects the relationship between party_size and player survive time.  In this game, the party size has three possibilities, a team for one player only,  a team for a pair of players or 4 players.  In this plot, the party_size of 4 has the largest average value of survival time, thus, a 4 players’ team may be the best choice for the game. However, the longest survival time for the party size 4 is shorter than the others. Although 4 players’ team can take advantage, it also has the risk of losing the game at the same time.  

![](pic/vis5.png)

(Figure 2.4 Density plot of time)

Figure 2.4 is about the time variable in the deaths dataset. More than half of the players can only survive for 0-500 seconds, points out that the rhythm of the game is very fast, players need to be very concentrated. As the time increases, the frequency decreases, but when the time interval from 1250 to 1750, the frequency increases a little, many players can survive until that time may win the round of the game.    

![](pic/vis4.png)

Figure 2.5 Boxplot of weapons and time)

Figure 2.5 shows different survival time of different weapons. The difference of time between weapons is obvious from this plot. In the machine learning part,  the recommendation of weapons will be discussed in more detail. 







