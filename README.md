# NBAMVPProject

NBA Most Valuable Players and Which Stat Most Defines Them

What makes an NBA player great?  Oftentimes, one can watch a player play the game and a special player stands out.  
Growing up watching Michael Jordan in the 90’s, I didn’t know anything about box scores, MVP votes, or even the basics of the game aside from “put the ball in the hoop.” 
But even as a young kid watching Michael play, I could tell he was special, and different from the other players in the league.  
Is there statistical data to back up the eye test that a player is special?  What statistic is most important in showing if a player should be an MVP Winner or not?

Using a dataset including season long statistics of every player in the NBA from 1982 - 2022, I used Python to analyze the data to show NBA players that were of MVP caliber in a given season.  I filtered this by only including players that received at least one MVP vote.  With the cleaned data, I checked correlation between variables and decided to base a theory on the Player Efficiency Rating, or PER, statistic.  My theory was:  the PER stat is the most important and predictive statistic of a player's overall value to his team, and thus the player with the highest PER should be awarded the NBA MVP.

Using linear regression, I compared NBA player’s PER over 40 years, and 58% of the MVPs in the last 40 years had the highest PER.  This statistic was first introduced by its creator, ESPN writer John Hollinger, in April 2007 at the end of the regular season and after MVP voting had already taken place. (Hollinger, John. “What Is per?” ESPN, ESPN Internet Ventures, 26 Apr. 2007, https://www.espn.com/nba/columns/story?columnist=hollinger_john&id=2850240.)  The MVP is voted on by a panel of sportswriters and broadcasters. Starting with the 2008 season, they would have been voting on MVP knowing each player’s PER value. (“NBA Most Valuable Player Award.” Wikipedia, Wikimedia Foundation, 24 Dec. 2022, https://en.wikipedia.org/wiki/NBA_Most_Valuable_Player_Award.)  Since 2008, 80% of MVP winners have had the highest PER.

