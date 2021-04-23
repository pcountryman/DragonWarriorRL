# This initial section imports the dataset containing both sold and unsold houses from a region
import pandas as pd
import os
from datetime import date
from CalcVIF import calc_vif
# from LatLongConversion import latlong
from BayesOptimizeGlobal import bayesoptimizemodel
from HouseCleanup import houseclean
from GlobalCV import knncv
import matplotlib.pyplot as plt
import numpy as np
from AlchemyConnect import alchemyconnect
from sqlalchemy import create_engine
from NeighborhoodElasticNet import neighborhood_elastic_net
from sqlalchemy.types import VARCHAR
from sklearn.metrics import mean_squared_error
import folium

# Make a new folder with today's date to store all outputs
today = str(date.today())
if not os.path.exists(today):
    os.makedirs(today)

stratified = 1
number_of_comps = 5
number_of_comps_neighborhood = 10
neighborhood = 'yes'  # either 'yes' or 'no' to also generate prices based off linear regression of closest homes
number_of_unsold_houses = 2000000
number_of_folds = 5
keeplist = ['latitude', 'longitude', 'year_built', 'num_beds', 'num_baths', 'sqft', 'acres', 'GarageSqft',
            'Adj_sale_amount', 'muni_code', 'sale_datetime']

# FIPS code
fips_code = '029'

# for each month, generate predictions using KNN
year_start = 2020
year_end = 2021
month_start = 7
month_end = 1
date_start = pd.Timestamp(year_start, month_start, 1)
date_end = pd.Timestamp(year_end, month_end, 1)

# how many months worth of previous sold homes to use as comps
month_time_window = 8

# Import the learning dataset, swis = muni_code, std_use_code = property_class
string_import = ("SELECT int_id, latitude, longitude, year_built, num_baths, num_beds, square_feet, acres, "
                 "garage_sqft, census_cousub, std_use_code")
string_database = f"FROM prop_{fips_code}"
house_all = alchemyconnect(string_import, string_database)
house_all = house_all.rename(columns={'census_cousub': 'muni_code'})

# Import the adjusted sale price dataset
string_import = "SELECT int_id, Adj_sale_amount, val_date, sale_date, model, price_estimate_lim"
string_database = f"FROM price_estimate_{fips_code}"
house_all_adj_sale_prices = alchemyconnect(string_import, string_database)
house_all_adj_sale_prices['sale_datetime'] = pd.to_datetime(
    house_all_adj_sale_prices.sale_date, format='%Y/%m/%d')

# create various lists for managing price predictions within for loops
all_predicted_price_list = []
all_predicted_index = []
val_date_list = []
all_neighborhood_predicted_list = []
all_neighborhood_index_list = []
all_neighborhood_coefs_list = []
all_neighborhood_intercepts_list = []
val_date_neigh_list = []
val_month_all = pd.DataFrame()

'''
make predictions on all homes sold in val_month based off homes sold in month_time_window from previous
months
'''
p = 0
for val_month in pd.date_range(start=date_start, end=date_end, freq='m'):
    p += 1  # todo change this to year & month from val_month
    date_list = []  # todo should this be inside for loop?
    date_list.append(val_month.strftime('%Y-%m-%d'))
    '''
    setup X month window for time analysis, to predict prices for val_month.
    date_cutoff is the last month used to train the model
    early_date_cutoff is the first month used to train the model
    All homes are price adjusted up to val_month for model training
    '''
    date_cutoff = val_month - pd.DateOffset(months=1)
    early_date_cutoff = date_cutoff - pd.DateOffset(months=month_time_window)

    house_adj_sale_prices = house_all_adj_sale_prices[
        house_all_adj_sale_prices.val_datetime == val_month]
    house_adj_sale_prices = house_adj_sale_prices.drop(['val_datetime', 'val_date'], axis=1)
    house_adj_sale_prices = house_adj_sale_prices.drop_duplicates(subset=['int_id'])
    house_adj_sale_prices = house_adj_sale_prices.set_index('int_id')

    house = house_adj_sale_prices.join(house_all)

    house.to_csv(f'{today}/housetest{p}.csv')

    house = house[(house.sale_datetime <= date_cutoff) &
                  (house.sale_datetime > early_date_cutoff)]  # todo double check homes sold on 1 exist

    # house.head().to_csv(f'{today}/housetest.csv')
    # house_adj_sale_prices.head().to_csv(f'{today}/houseadjsaleprice.csv')

    # Create dataframe for houses sold in val_month for later prediction and comparison
    # todo fix issue for overlapping days from Oct 31 into Nov

    house_sold_in_month_to_predict = house_all_adj_sale_prices[
        (house_all_adj_sale_prices['val_datetime'] == val_month) &
        (house_all_adj_sale_prices['sale_datetime'] <= val_month) &
        (house_all_adj_sale_prices['sale_datetime'] > (date_cutoff + pd.DateOffset(days=1)))
        ]
    house_sold_in_month_to_predict = house_sold_in_month_to_predict.set_index('int_id')
    house_sold_in_month_to_predict = house_sold_in_month_to_predict.join(house_all)

    # Clean up and separate dataframes into sold and unsold
    cleansold_valmonth, cleanunsold_valmonth, dead_df = houseclean(
        house_sold_in_month_to_predict, date_cutoff, early_date_cutoff, today, keeplist)
    cleansold, cleanunsold, house_master_id = houseclean(
        house, date_cutoff, early_date_cutoff, today, keeplist)


    # todo instead of dropping, consider adjust GarageSqft to 0
    cleansoldcutoff = cleansold.fillna(
        value={'GarageSqft': 0}).dropna(axis=0)
    cleansold = cleansold.fillna(
        value={'GarageSqft': 0}).dropna(axis=0)
    cleansold_valmonth = cleansold_valmonth.fillna(
        value={'GarageSqft': 0}).dropna(axis=0)

    # cleanunsold.to_csv(f'{today}/cleanunsold.csv')
    cleansold.to_csv(f'{today}/cleansold{p}.csv')
    # cleansoldcutoff.to_csv(f'{today}/cleansoldcutoff.csv')
    cleansold_valmonth.to_csv(f'{today}/cleansoldvalmonth{p}.csv')

    # %%

    sold_clean = cleansold[['latitude', 'longitude', 'years_ago_built', 'num_beds', 'num_baths',
                            'sqft', 'acres',
                            'GarageSqft', 'Adj_sale_amount', 'muni_code']]
    sold_clean_valmonth = cleansold_valmonth[['latitude', 'longitude', 'years_ago_built', 'num_beds',
                                              'num_baths',
                            'sqft', 'acres',
                            'GarageSqft', 'Adj_sale_amount', 'muni_code']]

    # VIF on standardized data removing unhelpful input parameters

    # Separate droplists for X and y

    droplist = ['Adj_sale_amount']

    # Not sold dataframe
    unsold_clean_dropped_nulls = cleanunsold[['latitude', 'longitude', 'years_ago_built',
                                              'num_beds', 'num_baths',
                                              'sqft', 'acres', 'GarageSqft', 'muni_code']].fillna(value=0)

    # todo incorporate VIF again into model
    # calc_vif(sold_clean_train_standardized, today)

    # create a list for unique muni_code for bayes optimization based on each muni_code
    muni_list = pd.unique(sold_clean_valmonth['muni_code']).tolist()

    # drop muni less than 10 sold houses
    # todo aggregate all dropped muni_code into one batch to analyze w/o distance
    small_muni = []
    for muni in muni_list:
        if len(sold_clean.loc[sold_clean['muni_code'] == muni]) < 10:
            small_muni.append(muni)
    for i in small_muni:
        try:
            muni_list.remove(i)
        except ValueError:
            pass

    # create empty dictionary to store key and value pairs for muni code and Bayes optimzed weights
    muni_dict = {}
    for muni in muni_list:
        # select all sold homes in muni muni_code
        sold_muni = sold_clean.loc[sold_clean['muni_code'] == muni]
        X = sold_muni.drop(droplist, axis=1)
        y = sold_muni['Adj_sale_amount']

        # select all unsold homes in muni muni_code
        unsold_muni = unsold_clean_dropped_nulls.loc[unsold_clean_dropped_nulls['muni_code'] == muni]

        # use bayes to optimize weights assigned to input parameters for each muni in muni_code
        optimal_weights = bayesoptimizemodel(X, y,
                                             unsold_muni,
                                             house_master_id,
                                             number_of_unsold_houses,
                                             number_of_comps, number_of_folds, today)

        # store muni and optimal_weights as key/value pairs in dictionary
        muni_dict[muni] = optimal_weights

    muni_df = pd.DataFrame(muni_dict).T
    muni_df.to_csv(f'{today}/muni_code_optimal_values.csv')

    # new dataframe to store weight adjusted values

    # Optimal weights returned as acres, bath, bed, garagesqft, latitude, longitude, sqft, years_ago
    # Prox model weights accepted as latitude, longitude, yearsago, bed, bath, sqft, acres, GarageSqft

    for muni in muni_list:
        # select all sold homes in muni muni_code
        '''
        Now that the optimal weights have been determine for the house parameters, we need to make
        predictions in each muni code for each val month. We only care about sold houses in val month
        for the time being.'''
        sold_muni = sold_clean.loc[sold_clean['muni_code'] == muni]
        sold_muni_valmonth = sold_clean_valmonth.loc[sold_clean_valmonth['muni_code'] == muni]
        X = sold_muni.drop(droplist, axis=1)
        Xvalmonth = sold_muni_valmonth.drop(droplist, axis=1)
        y = sold_muni['Adj_sale_amount']
        yvalmonth = sold_muni_valmonth['Adj_sale_amount']

        # select all unsold homes in muni muni_code
        unsold_muni = unsold_clean_dropped_nulls.loc[unsold_clean_dropped_nulls['muni_code'] == muni]
        optimal_weights = muni_df.loc[muni].tolist()

        valmonth_predict = knncv(X, y, unsold_muni, Xvalmonth, house_master_id, number_of_folds,
                                 optimal_weights[4], optimal_weights[5], optimal_weights[7], optimal_weights[2],
                                 optimal_weights[1], optimal_weights[6], optimal_weights[0], optimal_weights[3],
                                 number_of_unsold_houses, number_of_comps, today, muni_list, muni_df,
                                 model_train_or_predict='Predict')
        # all_predicted_price = pd.concat([all_predicted_price, muni_predicted_price])
        all_predicted_index = all_predicted_index + valmonth_predict.index.tolist()
        valmonth_predict['avepredict'] = valmonth_predict.mean(axis=1)
        valmonth_predict.to_csv(f'{today}/valmonthpredicttest.csv')
        all_predicted_price_list = all_predicted_price_list + valmonth_predict['avepredict'].tolist()
        val_date_list = val_date_list + (date_list * len(valmonth_predict.index.tolist()))
        val_month_all = pd.concat([val_month_all, valmonth_predict])
    # optional neighborhood elastic net price model, does not need to be run within muni code
    if neighborhood == 'yes':
        neigh_id, neigh_price, neigh_sqft, neigh_predict, neigh_coefs, neigh_intercepts = \
            neighborhood_elastic_net(
            sold_clean, sold_clean['Adj_sale_amount'],
                sold_clean_valmonth, sold_clean_valmonth['Adj_sale_amount'],
                number_of_comps_neighborhood, today)
        all_neighborhood_index_list = all_neighborhood_index_list + neigh_predict.index.tolist()
        all_neighborhood_predicted_list = all_neighborhood_predicted_list + neigh_predict.values.tolist()
        all_neighborhood_coefs_list = all_neighborhood_coefs_list + neigh_coefs.values.tolist()
        all_neighborhood_intercepts_list = all_neighborhood_intercepts_list + neigh_intercepts.values.tolist()
        val_date_neigh_list = val_date_neigh_list + (date_list * len(neigh_predict.index.tolist()))

# todo figure out why KNN and neighborhood output the same price

all_predicted_price = pd.DataFrame(zip(all_predicted_index, all_predicted_price_list))
# reassemble lists into a dataframe
all_predicted_price = all_predicted_price.rename(columns={0: 'int_id', 1: 'price_estimate'})
'''
idx = pd.Index(all_predicted_price.index.to_list(), name='int_id')
all_predicted_price['int_id'] = idx
all_predicted_price = all_predicted_price.set_index('int_id')  # todo remove, replace with index_label in to_sql
pierre_date_cutoff = date_cutoff - pd.Timedelta('1 days')  # todo link with AlchemyConnect code
val_date_list = [pierre_date_cutoff.strftime('%Y-%m-%d')] * len(all_predicted_price)
'''

# pull adjusted sale amounts
all_homes = house_all_adj_sale_prices.copy()
all_homes = all_homes.set_index(['int_id', 'val_date'])
all_homes = all_homes[~all_homes.index.duplicated()]

all_predicted_price['val_date'] = val_date_list
all_predicted_price['model'] = 'knn'
all_predicted_price = all_predicted_price.set_index(['int_id', 'val_date'])
all_predicted_price['Adj_sale_amount'] = all_homes['Adj_sale_amount']
all_predicted_price['compare'] = ((
    all_predicted_price['Adj_sale_amount'] - all_predicted_price['price_estimate'])
                                  / all_predicted_price['Adj_sale_amount'] * 100)
all_predicted_price['knn_accurate'] = np.where(abs(all_predicted_price['compare']) <= 10, 1, 0)
all_predicted_price['sale_date'] = all_homes['sale_date']

# assemble dataframe for all neighborhood prices
if neighborhood == 'yes':
    all_neighborhood_prices = pd.DataFrame(zip(all_neighborhood_index_list, all_neighborhood_predicted_list,
                                               all_neighborhood_coefs_list, all_neighborhood_intercepts_list))
    all_neighborhood_prices = all_neighborhood_prices.rename(columns={0: 'int_id', 1: 'price_estimate',
                                                                      2: 'PricePerSqft', 3: 'Location_Cost'})
    all_neighborhood_prices['val_date'] = val_date_neigh_list
    all_neighborhood_prices['model'] = 'neighborhood'
    all_neighborhood_prices = all_neighborhood_prices.set_index(['int_id', 'val_date'])

    all_neighborhood_prices['knn_predict'] = all_predicted_price['price_estimate']
    all_neighborhood_prices['Adj_sale_amount'] = all_homes['Adj_sale_amount']
    all_neighborhood_prices['price_estimate_lim'] = all_homes['price_estimate_lim']
    all_neighborhood_prices['sale_date'] = all_homes['sale_date']

    all_neighborhood_prices['compare'] = ((all_neighborhood_prices['Adj_sale_amount'] -
                                           all_neighborhood_prices['price_estimate']) /
                                          all_neighborhood_prices['Adj_sale_amount'] * 100)
    all_neighborhood_prices['neigh_accuracy'] = np.where(
        abs(all_neighborhood_prices['compare']) <= 10, 1, 0)

    # error based fixes
    # todo keep track of how many houses are eliminated by these methods
    all_neighborhood_prices = all_neighborhood_prices.dropna(axis=0, subset=['knn_predict', 'price_estimate_lim',
                                                                             'Adj_sale_amount'])
    all_neighborhood_prices = all_neighborhood_prices[all_neighborhood_prices.price_estimate_lim != 0]

    # include knn results for comparison
    all_neighborhood_prices['knn_accuracy'] = all_predicted_price['knn_accurate']
    all_predicted_price['neigh_accuracy'] = all_neighborhood_prices['neigh_accuracy']

    # write to Julius table
    engine = create_engine(
        'mysql+pymysql://preston:tRT8Rq2X23xe@ec2-54-205-186-164.compute-1.amazonaws.com:3306/Julius'
    )
    all_neighborhood_prices.to_sql(f'neighborhood_{fips_code}', con=engine, index_label=['int_id', 'val_date'],
                                   if_exists='replace', dtype={'val_date':VARCHAR(length=30)})

    all_neighborhood_prices.to_csv(f'{today}/allneighborhoodpricestest.csv')
    '''
    # examine error of methods
    neighborhood_rmse = mean_squared_error(
        all_neighborhood_prices['Adj_sale_amount'], all_neighborhood_prices['price_estimate'], squared=False)
    knn_rmse = mean_squared_error(
        all_neighborhood_prices['Adj_sale_amount'], all_neighborhood_prices['knn_predict'], squared=False)
    lim_rmse = mean_squared_error(
        all_neighborhood_prices['Adj_sale_amount'], all_neighborhood_prices['price_estimate_lim'], squared=False)

    print(f'Neighborhood RMSE {neighborhood_rmse}')
    print(f'KNN RMSE {knn_rmse}')
    print(f'lim RMSE {lim_rmse}')
    '''

'''
all_predicted_price.to_csv(f'{today}/PGVim_RMSE.csv')  # todo automate to what type of Bayes parameter is optimized
duplicate = pd.DataFrame(all_predicted_price.index.duplicated(), index=all_predicted_price.index)
duplicate.to_csv(f'{today}/duplicate.csv')
'''
# write to Julius table
engine = create_engine(
    'mysql+pymysql://preston:tRT8Rq2X23xe@ec2-54-205-186-164.compute-1.amazonaws.com:3306/Julius'
)
all_predicted_price.to_sql(f'knnpredict_{fips_code}', con=engine, index_label=['int_id', 'val_date'],
                                   if_exists='replace', dtype={'val_date':VARCHAR(length=30)})
all_predicted_price.to_csv(f'{today}/AllPredictedPrice.csv')
val_month_all.to_csv(f'{today}/valmonthalltest.csv')

'''
# Create a dataframe to examine accuracy of predicted prices
Accuracy = house_master_id.copy()
Accuracy = Accuracy.dropna(axis=0, subset=['latitude', 'longitude'])
Accuracy['ave_prediction'] = all_predicted_price['price_estimate']

Accuracy = Accuracy.dropna(axis=0, subset=['Adj_sale_amount'])

Accuracy['ave_prediction'] = Accuracy[['prediction1', 'prediction2', 'prediction3',
                                       'prediction4', 'prediction5']].mean(axis=1)

Accuracy['Price_Diff'] = Accuracy['Adj_sale_amount'] - Accuracy['ave_prediction']
Accuracy['Abs_Price_Diff'] = abs(Accuracy['Adj_sale_amount'] - Accuracy['ave_prediction'])
Accuracy['Std_Abs_Price_Diff'] = Accuracy['Abs_Price_Diff'] / Accuracy['Adj_sale_amount'] * 100

Accuracy.to_csv(f'{today}/Accuracy.csv')

# Create dataframes to looks at accuracy statistics breakdown by every 10% interval
for i in range(1, 5):
    AccuracyDescribe = Accuracy[(Accuracy['Std_Abs_Price_Diff'] <= i * 10)
                                & (Accuracy['Std_Abs_Price_Diff'] >= (i - 1) * 10)]
    AccuracyDescribe.describe().to_csv(
        f'{today}/Accuracy{i * 10}.csv')

plt.cla()
plt.scatter(range(0, len(Accuracy)), Accuracy.Std_Abs_Price_Diff, label='Price_Diff')
plt.legend(loc=0)
plt.xlabel("Arbitrary House")
plt.ylabel("Accuracy")
plt.ylim(0, 100)
plt.savefig(f'{today}/Accuracy.png')
plt.cla()
plt.scatter(Accuracy.num_beds, Accuracy.Std_Abs_Price_Diff, label='Beds_vs_Accuracy')
plt.legend(loc=0)
plt.xlabel("Number of Beds")
plt.ylabel("Accuracy")
plt.ylim(0, 100)
plt.savefig(f'{today}/Beds_vs_Accuracy.png')
plt.cla()
plt.scatter(Accuracy.num_baths, Accuracy.Std_Abs_Price_Diff, label='Bathrooms_vs_Accuracy')
plt.legend(loc=0)
plt.xlabel("Number of Bathrooms")
plt.ylabel("Accuracy")
plt.ylim(0, 100)
plt.savefig(f'{today}/Bathrooms_vs_Accuracy.png')
plt.cla()
plt.scatter(Accuracy.sqft, Accuracy.Std_Abs_Price_Diff, label='Sqft_vs_Accuracy')
plt.legend(loc=0)
plt.xlabel("Sqft")
plt.ylabel("Accuracy")
plt.ylim(0, 100)
plt.savefig(f'{today}/Sqft_vs_Accuracy.png')
plt.cla()
plt.scatter(Accuracy.acres, Accuracy.Std_Abs_Price_Diff, label='Acres_vs_Accuracy')
plt.legend(loc=0)
plt.xlabel("Acres")
plt.ylabel("Accuracy")
plt.ylim(0, 100)
plt.savefig(f'{today}/Acres_vs_Accuracy.png')
plt.cla()
plt.scatter(Accuracy.GarageSqft, Accuracy.Std_Abs_Price_Diff, label='GarageSqft_vs_Accuracy')
plt.legend(loc=0)
plt.xlabel("GarageSqft")
plt.ylabel("Accuracy")
plt.ylim(0, 100)
plt.savefig(f'{today}/GarageSqft_vs_Accuracy.png')
plt.cla()

plt.scatter(Accuracy.years_ago_built, Accuracy.Std_Abs_Price_Diff, label='YearsAgoBuilt_vs_Accuracy')
plt.legend(loc=0)
plt.xlabel("YearsAgoBuilt")
plt.ylabel("Accuracy")
plt.ylim(0, 100)
plt.savefig(f'{today}/YearsAgoBuilt_vs_Accuracy.png')

plt.cla()
plt.scatter(Accuracy.ave_prediction, Accuracy.Std_Abs_Price_Diff, label='ave_prediction_vs_Accuracy')
plt.legend(loc=0)
plt.xlabel("ave_prediction")
plt.ylabel("Accuracy")
plt.ylim(0, 100)
plt.savefig(f'{today}/ave_prediction_vs_Accuracy.png')
plt.cla()
Accuracy_Bins = np.linspace(0, 100, 11)
plt.hist(Accuracy['Std_Abs_Price_Diff'], bins=Accuracy_Bins)
plt.savefig(f'{today}/AccuracyHistogram')

SoldLat = Accuracy['latitude'].tolist()
SoldLong = Accuracy['longitude'].tolist()


# create the map.
map_pickup = folium.Map(location=[43.0203404, -78.9016948])
SoldLatLong = list(zip(SoldLat, SoldLong))
for i in range(0, len(Accuracy)):
    if Accuracy['Std_Abs_Price_Diff'].iloc[i] <= 10:
        folium.Marker(
            location=SoldLatLong[i],
            popup=f'{Accuracy.index[i]}',
            icon=folium.Icon(icon_color='green')
        ).add_to(map_pickup)
    elif Accuracy['Std_Abs_Price_Diff'].iloc[i] <= 20:
        folium.Marker(
            location=SoldLatLong[i],
            popup=f'{Accuracy.index[i]}',
            icon=folium.Icon(icon_color='yellow')
        ).add_to(map_pickup)
    else:
        folium.Marker(
            location=SoldLatLong[i],
            popup=f'{Accuracy.index[i]}',
            icon=folium.Icon(icon_color='red')
        ).add_to(map_pickup)
map_pickup.save(
    f'{today}/Location_vs_Abs_Price_Diff.html')
'''
# knncv(sold_clean_train_standardized, sold_amount_clean_train_standardized, today)
