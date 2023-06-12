# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:49:08 2023

@author: Dan Johnson
"""

# Import Python libraries Pandas
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
import scipy.stats, statsmodels.api as sm

# Create Pandas dataframe for Dataset
nba = pd.read_csv('C:/Users/danjo/OneDrive/Python Project Files/Johnson_Dan_DACapTask_3_nbadataset.csv')

# Find the max of values in specific columns that have bad data
max_per = nba['per'].max()
max_ts = nba['ts_pct'].max()

# Locate the index of per columns with bad data
bad_per_max = nba[nba['per'] > 32.85].index.values
bad_per_max = list(bad_per_max)
bad_per_min = nba[nba['per'] < 0].index.values
bad_per_min = list(bad_per_min)

# Use the lists created to drop the bad rows from DataFrame nba
nba2 = nba.drop(bad_per_max, axis=0, inplace=False)
nba2 = nba.drop(bad_per_min, axis=0, inplace=False)

# Locate the index of ts_pct columns with bad data
bad_ts = nba[nba['ts_pct'] > 1].index.values
bad_ts = list(bad_ts)

# Use the list created to drop the bad rows from DataFrame nba
nba2 = nba.drop(bad_ts, axis=0, inplace=False)

# Locate the index of players who played 10 games or less in a single season
too_few_games = nba[nba['g'] <= 10].index.values
too_few_games = list(too_few_games)

# Use the list created to drop the bad rows from DataFrame nba
nba2 = nba.drop(too_few_games, axis=0, inplace=False)

# Find and Drop any rows if a player did not recieve MVP votes
non_mvp = nba[nba['award_share'] <= 0.0].index.values
non_mvp = list(non_mvp)
nba.drop(non_mvp, axis=0, inplace=True)

# Drop columns that won't be needed or are redundant
nba.drop(['orb_per_g', 'drb_per_g', 'fg3a_per_fga_pct', 'fta_per_fga_pct', 
          'orb_pct', 'drb_pct', 'ows', 'dws', 'ws_per_48', 'obpm', 'dbpm', 
          'mov_adj', 'gs', 'mp_per_g', 'fg_per_g', 'fga_per_g', 
          'fg_pct', 'fg3_per_g', 'fg3a_per_g', 'fg2_per_g', 'fg2a_per_g', 
          'fg2_pct', 'efg_pct', 'ft_per_g', 'fta_per_g', 
          'ft_pct', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g',
          'mp', 'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct',
          'vorp', 'mov', 'fg3_pct', 'trb_pct'], axis=1, inplace=True)
nba2.drop(['orb_per_g', 'drb_per_g', 'fg3a_per_fga_pct', 'fta_per_fga_pct', 
          'orb_pct', 'drb_pct', 'ows', 'dws', 'ws_per_48', 'obpm', 'dbpm', 
          'mov_adj', 'gs', 'mp_per_g', 'fg_per_g', 'fga_per_g', 
          'fg_pct', 'fg3_per_g', 'fg3a_per_g', 'fg2_per_g', 'fg2a_per_g', 
          'fg2_pct', 'efg_pct', 'ft_per_g', 'fta_per_g', 
          'ft_pct', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g',
          'mp', 'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct',
          'vorp', 'mov', 'fg3_pct', 'trb_pct'], axis=1, inplace=True)

# Create a category column for PER, to see how players place in roles
nba['per_categories'] = pd.cut(
    x=nba['per'],
    bins=[-1.0, 9.0, 11.0, 13.0, 15.0, 16.5, 18.0, 20.0, 22.5, 25.0, 27.5,
          30.0, 35.0, np.inf],
    labels=['Player who won\'t stick in the leauge', 'Fringe roster player',
            'Non-rotation player', 'Rotation Player', 
            'Slightly above-average player', 'Third Offensive Option',
            'Second Offensive Option', 'Borderline All-Star',
            'Definitie All-Star', 'Weak MVP Candidate',
            'Strong MVP Candidate', 'Runaway MVP Candidate',
            'All-Time Great Season']
    )

# Create Describe table for NBA Dataset
desc = nba.describe().round(2).to_clipboard()

# Create a heatmap and correlation matrix to show any correlation between columns
cm = nba.corr().round(2)
sns.heatmap(nba.corr()**2, cmap='inferno', fmt='.2f').set(
    title='Correlation heatmap of columns of interest')

# Box plot of won mvp and PER
plt.clf()
sns.boxplot(x=nba['won_mvp'], y=nba['per'], data=nba).set(
    title='Player Efficiency Rating of MVP winners and non-winners',
    ylabel='Player Efficiency Rating in units', 
    xlabel='0 if non-MVP winner, 1 if MVP winner')
plt.show()

# Box plot of Pts per game and Won MVP
plt.clf()
sns.boxplot(x=nba['won_mvp'], y=nba['pts_per_g'], data=nba).set(
    title='Points Per Game of MVP winners and non-winners',
    ylabel='Points Per Game', 
    xlabel='0 if non-MVP winner, 1 if MVP winner')
plt.show()

# Box plot of Assists per game and won mvp
plt.clf()
sns.boxplot(x=nba['won_mvp'], y=nba['ast_per_g'], data=nba).set(
    title='Assists Per Game of MVP winners and non-winners',
    ylabel='Assists Per Game', 
    xlabel='0 if non-MVP winner, 1 if MVP winner')
plt.show()

# Box plot of True Shooting % and Won MVP
plt.clf()
sns.boxplot(x=nba['won_mvp'], y=nba['ts_pct'], data=nba).set(
    title='True Shooting % of MVP winners and non-winners',
    ylabel='True Shooting %', 
    xlabel='0 if non-MVP winner, 1 if MVP winner')
plt.show()

# Regression and Residual Plots for PER and Points Per Game
plt.clf()
sns.regplot(x=nba['pts_per_g'], y=nba['per'], data=nba)
sns.residplot(x=nba['pts_per_g'], y=nba['per'], data=nba).set(
        title='Player Efficiency Rating and Points Per Game Linear Regression and Residuals',
        xlabel='Average Points per game', ylabel='Player Efficiency Rating')
plt.show()

# Regression and Residual plots for PER and True Shooting Percentage
plt.clf()
sns.regplot(x=nba['ts_pct'], y=nba['per'], data=nba)
sns.residplot(x=nba['ts_pct'], y=nba['per'], data=nba).set(
    title= 'Player Efficiency Rating and True Shooting Percentage Linear Regression and Residuals',
    xlabel= 'True Shooting Percentage', ylabel='Player Efficiency Rating')
plt.show()

# Regression and Residual plots for PER and Assists Per Game
plt.clf()
sns.regplot(x=nba['ast_per_g'], y=nba['per'], data=nba)
sns.residplot(x=nba['ast_per_g'], y=nba['per'], data=nba).set(
    title= 'Player Efficiency Rating and Assists Per Game Linear Regression and Residuals',
    xlabel= 'Assists Per Game', ylabel='Player Efficiency Rating')
plt.show()

# Regression and Residual plots for PER and Rebounds Per Game
plt.clf()
sns.regplot(x=nba['trb_per_g'], y=nba['per'], data=nba)
sns.residplot(x=nba['trb_per_g'], y=nba['per'], data=nba).set(
    title= 'Player Efficiency Rating and Rebounds Per Game Linear Regression and Residuals',
    xlabel= 'Rebounds Per Game', ylabel='Player Efficiency Rating')
plt.show()

# Regression and Residual plots for PER and Win Shares
plt.clf()
sns.regplot(x=nba['ws'], y=nba['per'], data=nba)
sns.residplot(x=nba['ws'], y=nba['per'], data=nba).set(
    title= 'Player Efficiency Rating and Win Shares Linear Regression and Residuals',
    xlabel= 'Win Shares', ylabel='Player Efficiency Rating')
plt.show()

# Create Frequency Table showing how many people Won MVP
MVP_table = nba['won_mvp'].value_counts()
print(f'Total MVP Winners: {MVP_table[1]}')
print(f'Total MVP Non-Winners: {MVP_table[0]}')

# Create DataFrame showing MVP winners
mvps = nba.loc[nba['won_mvp'] == 1]
mvps.to_clipboard()

# Create DataFrame showing non-MVP winners
non_mvps = nba.loc[nba['won_mvp'] == 0]

# Multiple Linear Regression comparing PER to Points per game for MVP winners
X = mvps[['pts_per_g','bpm', 'ws']]
Y = mvps['per']
Xc = sm.add_constant(X)
model = sm.OLS(Y, Xc).fit()
print(model.params)
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 1, ax=ax)
plt.xlabel('Points Per Game')
plt.ylabel('Player Efficiency Rating')
plt.title('Fitted PER versus Points Per Game for MVP Winners')
plt.grid()
plt.show()

# Multiple Linear Regression comparing PER to Points per game for non-MVP winners
X = non_mvps[['pts_per_g','bpm', 'ws']]
Y = non_mvps['per']
Xc = sm.add_constant(X)
model = sm.OLS(Y, Xc).fit()
print(model.params)
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 1, ax=ax)
plt.xlabel('Points Per Game')
plt.ylabel('Player Efficiency Rating')
plt.title('Fitted PER versus Points Per Game for non-MVP Winners')
plt.grid()
plt.show()

# Multiple Linear Regression comparing PER to Box Plus/Minus for MVP winners
X = mvps[['pts_per_g','bpm', 'ws']]
Y = mvps['per']
Xc = sm.add_constant(X)
model = sm.OLS(Y, Xc).fit()
print(model.params)
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 2, ax=ax)
plt.xlabel('Box Plus/Minus')
plt.ylabel('Player Efficiency Rating')
plt.title('Fitted PER versus Box Plus/Minus For MVP Winners')
plt.grid()
plt.show()

# Multiple Linear Regression comparing PER to Box Plus/Minus for non-MVP winners
X = non_mvps[['pts_per_g','bpm', 'ws']]
Y = non_mvps['per']
Xc = sm.add_constant(X)
model = sm.OLS(Y, Xc).fit()
print(model.params)
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 2, ax=ax)
plt.xlabel('Box Plus/Minus')
plt.ylabel('Player Efficiency Rating')
plt.title('Fitted PER versus Box Plus/Minus for non-MVP Winners')
plt.grid()
plt.show()

# Multiple Linear Regression comparing PER to Points per game for MVP winners
X = mvps[['pts_per_g','bpm', 'ws']]
Y = mvps['per']
Xc = sm.add_constant(X)
model = sm.OLS(Y, Xc).fit()
print(model.params)
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 3, ax=ax)
plt.xlabel('Win Shares in total wins')
plt.ylabel('Player Efficiency Rating')
plt.title('Fitted PER versus Win Shares for MVP Winners')
plt.grid()
plt.show()

# Multiple Linear Regression comparing PER to Points per game for MVP non-winners
X = non_mvps[['pts_per_g','bpm', 'ws']]
Y = non_mvps['per']
Xc = sm.add_constant(X)
model = sm.OLS(Y, Xc).fit()
print(model.params)
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 3, ax=ax)
plt.xlabel('Win Shares in total wins')
plt.ylabel('Player Efficiency Rating')
plt.title('Fitted PER versus Win Shares for non-MVP winners')
plt.grid()
plt.show()

# Histogram of MVP PER numbers
plt.clf()
mvp_per_hist = mvps.hist(column='per')
plt.title('PER of MVP Winners')
plt.xlabel('MVP PER\'s')
plt.ylabel('Frequency')
plt.show()

# Histogram of MVP Non-PER numbers
plt.clf()
non_mvp_per_hist = non_mvps.hist(column='per')
plt.title('PER of Non-MVP Winners')
plt.xlabel('Non-MVP PER\'s')
plt.ylabel('Frequency')

# Histogram showing games play by MVP Winners
plt.clf()
mvp_games_hist = mvps.hist(column='g')
plt.title('Games played by MVP Winners')
plt.xlabel('Total Games Played')
plt.ylabel('Frequency')

# Histogram showing games played by Non-MVP
plt.clf()
non_mvp_games_hist = non_mvps.hist(column='g')
plt.title('Games played by Non-MVP Winners')
plt.xlabel('Total Games Played')
plt.ylabel('Frequency')

# Find minimum and maximum per of MVP Winners
max_per_mvp = mvps['per'].max()
min_per_mvp = mvps['per'].min()
Nikola_Jokic_2022 = mvps['player'].loc[mvps['per'] == max_per_mvp]
Steve_Nash_2005 = mvps['player'].loc[mvps['per'] == min_per_mvp]

# Create a table of all PER numbers from 2005 season
per_2005season = nba[['per', 'player']].loc[nba['season'] == 2005]
per_2005season = per_2005season.sort_values(by='per', ascending=False)

# Create a table of all PER numbers from 2022 season
per_2022season = nba[['per', 'player']].loc[nba['season'] == 2022]
per_2022season = per_2022season.sort_values(by='per', ascending=False)

# Create a table showing only the highest PER in each season
m = nba.groupby('season')['per'].transform('max') == nba['per']
max_per_season = nba.loc[m]
# Drop columns from max_per_season table we aren't using
max_per_season.drop(['pos', 'age', 'team_id', 'bpm', 'g', 'trb_per_g',
                     'pts_per_g', 'ts_pct', 'ws', 'award_share', 'win_loss_pct',
                     'ast_per_g'], axis=1, inplace=True)
max_per_season.to_clipboard()

# Find out Season year of Min and Max PER MVP Winners
print(mvps['season'].loc[mvps['per'] == max_per_mvp])
print(mvps['season'].loc[mvps['per'] == min_per_mvp])

# Find Minimum and Maximun PER of non-MVP Winners
max_per_non_mvp = non_mvps['per'].max()
min_per_non_mvp = non_mvps['per'].min()
Giannis_2022 = non_mvps['player'].loc[non_mvps['per'] == max_per_non_mvp]
Mark_Eaton_1989 = non_mvps['player'].loc[non_mvps['per'] == min_per_non_mvp]

# Find out Season year of Min and Max PER MVP Winners
print(non_mvps['season'].loc[non_mvps['per'] == max_per_non_mvp])
print(non_mvps['season'].loc[non_mvps['per'] == min_per_non_mvp])

# Scatterplot showing MVP winners and their PER
plt.clf()
sns.scatterplot(x='per', y='season', hue='player', data=mvps, 
                palette='tab20b', legend='brief')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol=2)
plt.title('PER numbers by MVP Winners by Season')
plt.xlim(10, 35)
plt.show()

# Scatterplot showing Non-MVP's and their PER
plt.clf()
sns.scatterplot(x='per', y='season', data=non_mvps, 
                palette='tab20b', legend='brief')
plt.title('PER numbers by Non-MVP Winners by Season')
plt.xlim(10, 35)
plt.show()

# Scatterplot showing MVP winners and their Points Per Game
plt.clf()
sns.scatterplot(x='pts_per_g', y='player', hue='season', data=mvps, 
                palette='tab20b', legend='brief')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol=2)
plt.title('Points Per Game numbers by MVP Winners by Season')
plt.show()

# Scatterplot showing Non-MVP winners and their Points Per Game
plt.clf()
sns.scatterplot(x='pts_per_g', y='season', hue='season', data=non_mvps, 
                palette='tab20b', legend='brief')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol=2)
plt.title('Points Per Game numbers by MVP Winners by Season')
plt.show()

# Scatterplot show MVP winners and their True Shooting Percentage
plt.clf()
sns.scatterplot(x='ts_pct', y='player', hue='season', data=mvps, 
                palette='tab20b', legend='brief')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol=2)
plt.title('True Shooting Percentage by MVP Winners by Season')
plt.show()

# Scatterplot showing non-MVP winners and their True Shooting Percentage
plt.clf()
sns.scatterplot(x='per', y='season', hue='season', data=non_mvps, 
                palette='tab20b', legend='brief')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol=2)
plt.title('True Shooting Percentage by Non-MVP Winners by Season')
plt.show()
